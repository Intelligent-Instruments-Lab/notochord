import math
from typing import List, Tuple, Dict, Union, Any
from numbers import Number
from collections import namedtuple, defaultdict
from pathlib import Path
import hashlib
import json

import mido
from tqdm import tqdm
import appdirs
import numpy as np
import joblib

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as D

from .rnn import GenericRNN
from .distributions import CensoredMixtureLogistic, categorical_sample

from .util import arg_to_set, download_url

class Query:
    def __init__(self, modality, cases=None, value=None, then=None, **kw):
        assert (cases is None) or (value is None)
        self.cases = cases
        self.value = value
        self.modality = modality
        self.then = then
        self.kw = kw
    def __repr__(self):
        return f"{self.modality} \n{self.then}"# {self.kw}"
    
# TODO: refactor use of Range into a 'cases' keyword, rather than a 'then' case
# add case index to the event
class Range:
    def __init__(self, lo=-torch.inf, hi=torch.inf, weight=1, sample_lo=None, sample_hi=None):
        """use lo, hi when computing weights of each branch; but actually sample between sample_lo and sample_hi. for example, you could let lo,hi cover the full range to compute the true model on/off ratio, but sample from a narrow range of allowed velocities in the noteOn case"""
        self.lo = lo
        self.hi = hi
        self.weight = weight
        self.sample_lo = lo if sample_lo is None else sample_lo
        self.sample_hi = hi if sample_hi is None else sample_hi

class Subset:
    def __init__(self, values=None, weight=1, sample_values=None):
        self.values = values
        self.weight = weight
        self.sample_values = values if sample_values is None else sample_values

def get_from_scalar_or_dict(x, default=None):
    if isinstance(x, dict):
        return lambda i: x.get(i, default)
    else:
        return lambda _: default if x is None else x

def _user_data_dir():
    d = Path(appdirs.user_data_dir('Notochord', 'IIL'))
    d.mkdir(exist_ok=True, parents=True)
    return d

mem = joblib.Memory(_user_data_dir())

class SineEmbedding(nn.Module):
    def __init__(self, n, hidden, w0=1e-3, w1=10, scale='log'):
        """
        Args:
            n (int): number of sinusoids
            hidden (int): embedding size
            w0 (float): minimum wavelength
            w1 (float): maximum wavelength
            scale (str): if 'log', more wavelengths close to w0
        """
        super().__init__()
        if scale=='log':
            w0 = math.log(w0)
            w1 = math.log(w1)
        ws = torch.linspace(w0, w1, n)
        if scale=='log':
            ws = ws.exp()
        self.register_buffer('fs', 2 * math.pi / ws)
        self.proj = nn.Linear(n,hidden)

    def forward(self, x):
        x = x[...,None] * self.fs
        return self.proj(x.sin())

class MixEmbedding(nn.Module):
    def __init__(self, n, domain=(0,1)):
        """
        Args:
            n (int): number of channels
            domain (Tuple[float])
        """
        super().__init__()
        self.domain = domain
        self.lo = nn.Parameter(torch.randn(n))
        self.hi = nn.Parameter(torch.randn(n))
    def forward(self, x):
        """
        Args:
            x: Tensor[...]
        Returns:
            Tensor[...,n]
        """
        x = (x - self.domain[0])/(self.domain[1] - self.domain[0])
        x = x[...,None]
        return self.hi * x + self.lo * (1-x)

class GLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        a, b = x.chunk(2, -1)
        return a * b.sigmoid()

class GLUMLP(nn.Module):
    def __init__(self, input, hidden, output, layers, dropout=0, norm=None):
        super().__init__()
        h = input
        def get_dropout():
            if dropout > 0:
                return (nn.Dropout(dropout),)
            return tuple()
        def get_norm():
            if norm=='layer':
                return (nn.LayerNorm(hidden),)
            return tuple()
        self.net = []
        for _ in range(layers):
            self.net.append(nn.Sequential(
                *get_dropout(), nn.Linear(h, hidden*2), GLU(), *get_norm()))
            h = hidden
        self.net.append(nn.Linear(hidden, output))
        self.net = nn.Sequential(*self.net)

        with torch.no_grad():
            self.net[-1].weight.mul_(1e-2)

    def forward(self, x):
        return self.net(x)


class Notochord(nn.Module):
    # note: use named arguments only for benefit of training script
    def __init__(self, 
            emb_size=256, 
            rnn_hidden=2048, rnn_layers=1, kind='gru', 
            mlp_layers=0,
            dropout=0.1, norm=None,
            num_pitches=128, 
            num_instruments=320,
            time_sines=128, vel_sines=128,
            time_bounds=(0,10), time_components=32, time_res=1e-2,
            vel_components=16
            ):
        """
        """
        super().__init__()

        self.step = 0
        self.current_time = 0
        self.held_notes = {}

        self.note_dim = 4 # instrument, pitch, time, velocity

        self.instrument_start_token = 0
        self.instrument_domain = num_instruments+1

        self.pitch_start_token = num_pitches
        self.pitch_domain = num_pitches+1

        self.max_dt = time_bounds[1]
        self.time_dist = CensoredMixtureLogistic(
            time_components, time_res, 
            sharp_bounds=(1e-4,2e3),
            lo=time_bounds[0], hi=time_bounds[1], init='time')
        self.vel_dist = CensoredMixtureLogistic(
            vel_components, 1.0,
            sharp_bounds=(1e-3,128),
            lo=0, hi=127, init='velocity')
        
        # embeddings for inputs
        self.instrument_emb = nn.Embedding(self.instrument_domain, emb_size)
        self.pitch_emb = nn.Embedding(self.pitch_domain, emb_size)
        self.time_emb = (#torch.jit.script(
            SineEmbedding(
            time_sines, emb_size, 1e-3, 30, scale='log'))
        # self.vel_emb = MixEmbedding(emb_size, (0, 127))
        self.vel_emb = (#torch.jit.script(
            SineEmbedding(
            vel_sines, emb_size, 2, 512, scale='lin'))

        # RNN backbone
        self.rnn = GenericRNN(kind, 
            emb_size, rnn_hidden, 
            num_layers=rnn_layers, batch_first=True, dropout=dropout)

        # learnable initial RNN state
        self.initial_state = nn.ParameterList([
             # layer x batch x hidden
            nn.Parameter(torch.randn(rnn_layers,1,rnn_hidden)*rnn_hidden**-0.5)
            for _ in range(2 if kind=='lstm' else 1)
        ])

        mlp_cls = GLUMLP#lambda *a: torch.jit.script(GLUMLP(*a))
        # projection from RNN state to distribution parameters
        self.h_proj = mlp_cls(
                rnn_hidden, emb_size, emb_size, 
                mlp_layers, dropout, norm)
        self.projections = nn.ModuleList([
            mlp_cls(
                emb_size, emb_size, self.instrument_domain, 
                mlp_layers, dropout, norm),
            mlp_cls(
                emb_size, emb_size, self.pitch_domain, 
                mlp_layers, dropout, norm),
            mlp_cls(
                emb_size, emb_size, self.time_dist.n_params,
                mlp_layers, dropout, norm),
            mlp_cls(
                emb_size, emb_size, self.vel_dist.n_params, 
                mlp_layers, dropout, norm),
        ])

        self.end_proj = nn.Linear(rnn_hidden, 2)

        with torch.no_grad():
            for p in self.projections:
                p.net[-1].weight.mul_(1e-2)
            self.end_proj.weight.mul(1e-2)

        # persistent RNN state for inference
        for n,t in zip(self.cell_state_names(), self.initial_state):
            self.register_buffer(n, t.clone())
        self.step = 0

        # volatile hidden states for caching purposes
        self.h = None
        self.h_query = None      

    def cell_state_names(self):
        return tuple(f'cell_state_{i}' for i in range(len(self.initial_state)))

    @property
    def cell_state(self):
        return tuple(getattr(self, n) for n in self.cell_state_names())

    @property
    def embeddings(self):
        return (
            self.instrument_emb,
            self.pitch_emb,
            self.time_emb,
            self.vel_emb
        )
        
    def forward(self, instruments, pitches, times, velocities, ends,
            validation=False, ar_mask=None):
        """
        teacher-forced probabilistic loss and diagnostics for training.

        Args:
            instruments: LongTensor[batch, time]
            pitches: LongTensor[batch, time]
            times: FloatTensor[batch, time]
            velocities: FloatTensor[batch, time]
            ends: LongTensor[batch, time]
            validation: bool (computes some extra diagnostics)
            ar_mask: Optional[Tensor[note_dim x note_dim]] if None, generate random
                masks for training
        """
        batch_size, batch_len = pitches.shape

        self.checkpoint_path = None

        # embed data to input vectors
        inst_emb = self.instrument_emb(instruments) # batch, time, emb_size
        pitch_emb = self.pitch_emb(pitches) # batch, time, emb_size
        time_emb = self.time_emb(times) # batch, time, emb_size
        vel_emb = self.vel_emb(velocities) # batch, time, emb_size

        embs = (inst_emb, pitch_emb, time_emb, vel_emb)

        # feed to RNN backbone
        x = sum(embs)
        ## broadcast initial state to batch size
        initial_state = tuple(
            t.expand(self.rnn.num_layers, x.shape[0], -1).contiguous() # 1 x batch x hidden
            for t in self.initial_state)
        h, _ = self.rnn(x, initial_state) #batch, time, hidden_size

        # fit all event factorizations 
        # e.g. inst->pitch->time->vel vs vel->time->inst->pitch
        trim_h = h[:,:-1]
        # always include hidden state, never include same modality,
        # other dependencies are random per time and position
        n = self.note_dim
        if ar_mask is None:
            # random binary mask
            ar_mask = torch.randint(2, (*trim_h.shape[:2],n,n), dtype=torch.bool, device=h.device)
            # zero diagonal
            ar_mask &= ~torch.eye(n,n, dtype=torch.bool, device=h.device)
        # include hidden state
        ar_mask = torch.cat((ar_mask.new_ones(*ar_mask.shape[:-2],1,n), ar_mask), -2).float()

        to_mask = torch.stack((
            self.h_proj(trim_h),
            *(emb[:,1:] for emb in embs)
        ), -1)
        # TODO: try without this tanh?
        mode_hs = (to_mask @ ar_mask).tanh().unbind(-1)
        
        # final projections to raw distribution parameters
        inst_params, pitch_params, time_params, vel_params = [
            proj(h) for proj,h in zip(self.projections, mode_hs)]

        # get likelihood of data for each modality
        inst_logits = F.log_softmax(inst_params, -1)
        inst_targets = instruments[:,1:,None] #batch, time, 1
        inst_log_probs = inst_logits.gather(-1, inst_targets)[...,0]

        pitch_logits = F.log_softmax(pitch_params, -1)
        pitch_targets = pitches[:,1:,None] #batch, time, 1
        pitch_log_probs = pitch_logits.gather(-1, pitch_targets)[...,0]

        time_targets = times[:,1:] # batch, time
        time_result = self.time_dist(time_params, time_targets)
        time_log_probs = time_result.pop('log_prob')

        vel_targets = velocities[:,1:] # batch, time
        vel_result = self.vel_dist(vel_params, vel_targets)
        vel_log_probs = vel_result.pop('log_prob')

        # end prediction
        # skip the first position for convenience 
        # (so masking is the same for end as for note parts)
        end_params = self.end_proj(h[:,1:])
        end_logits = F.log_softmax(end_params, -1)
        end_log_probs = end_logits.gather(-1, ends[:,1:,None])[...,0]

        r = {
            'end_log_probs': end_log_probs,
            'instrument_log_probs': inst_log_probs,
            'pitch_log_probs': pitch_log_probs,
            'time_log_probs': time_log_probs,
            'velocity_log_probs': vel_log_probs,
            **{'time_'+k:v for k,v in time_result.items()},
            **{'velocity_'+k:v for k,v in vel_result.items()}
        }
        # this just computes some extra diagnostics which are inconvenient to do in the
        # training script. should be turned off during training for performance.
        if validation:
            with torch.no_grad():
                r['time_acc_30ms'] = (
                    self.time_dist.cdf(time_params, time_targets + 0.03)
                    - torch.where(time_targets - 0.03 >= 0,
                        self.time_dist.cdf(time_params, time_targets - 0.03),
                        time_targets.new_zeros([]))
                )
        return r


    # 0 - start token
    # 1-128 - melodic
    # 129-256 - drums
    # 257-288 - anon melodic
    # 289-320 - anon drums
    def is_drum(self, inst):
        # TODO: add a constructor argument to specify which are drums
        # hardcoded for now
        return inst > 128 and inst < 257 or inst > 288
    def is_anon(self, inst):
        return inst > 256
    def first_anon_like(self, inst):
        # TODO: add a constructor argument to specify how many anon
        # hardcoded for now
        return 289 if self.is_drum(inst) else 257
    
    def feed(self, inst:int, pitch:int, time:Number, vel:Number, **kw):
        """consume an event and advance hidden state
        
        Args:
            inst: int. instrument of current note.
                0 is start token
                1-128 are General MIDI instruments
                129-256 are drumkits (MIDI 1-128 on channel 13)
                257-288 are 'anonymous' melodic instruments
                289-320 are 'anonymous' drumkits
            pitch: int. MIDI pitch of current note.
                0-127 are MIDI pitches / drums
                128 is start token
            time: float. elapsed time in seconds since previous event.
            vel: float. (possibly dequantized) MIDI velocity from 0-127 inclusive.
                0 indicates a note-off event
            **kw: ignored (allows doing e.g. noto.feed(**noto.query(...)))
        """
        # print(f'FEED from {threading.get_ident()}') 
        # print('feed', inst, pitch, time, vel)

        # track elapsed time and ongoing notes
        key = (inst,pitch)
        for k in self.held_notes:
            self.held_notes[k] += time
        self.current_time += time
        self.step += 1

        if vel > 0:
            self.held_notes[key] = 0
        elif key in self.held_notes:
            self.held_notes.pop(key)

        # print(self.held_notes)

        # update RNN state

        with torch.inference_mode():
            inst = torch.LongTensor([[inst]]) # 1x1 (batch, time)
            pitch = torch.LongTensor([[pitch]]) # 1x1 (batch, time)
            time = torch.FloatTensor([[time]]) # 1x1 (batch, time)
            vel = torch.FloatTensor([[vel]]) # 1x1 (batch, time)

            embs = [
                self.instrument_emb(inst), # 1, 1, emb_size
                self.pitch_emb(pitch), # 1, 1, emb_size
                self.time_emb(time),# 1, 1, emb_size
                self.vel_emb(vel)# 1, 1, emb_size
            ]
            x = sum(embs)

            self.h, new_state = self.rnn(x, self.cell_state)
            for t,new_t in zip(self.cell_state, new_state):
                t[:] = new_t

            self.h_query = None
            
    def deep_query(self, query, predict_end=True):
        """flexible querying with nested Query objects.
        see query_vtip for an example.

        Args:
            query: Query object
        """
        with torch.inference_mode():
            if self.h_query is None:
                self.h_query = self.h_proj(self.h)
            event = self._deep_query(
                query, hidden=self.h_query[:,0], event={})
            
            if predict_end:
                # print('END')
                # print(f'{self.h}')
                end_params = self.end_proj(self.h)
                event['end'] = end_params.softmax(-1)[...,1].item()
                # event['end'] = D.Categorical(logits=end_params).sample().item()
            else:
                event['end'] = 0#torch.zeros(self.h.shape[:-1])

        return event
    
    def _deep_query(self, query, hidden, event):
        if hasattr(query, '__call__'):
            query = query(event)
        m = query.modality
        sq = query.then
        try:
            idx = ('inst','pitch','time','vel').index(m)
        except ValueError:
            raise ValueError(f'unknown modality "{m}"')
        
        project = self.projections[idx]
        embed = self.embeddings[idx]

        if query.value is not None:
            result = torch.tensor(query.value)
        else:
            if m=='time':
                dist = self.time_dist
                sample = dist.sample
            elif m=='vel':
                dist = self.vel_dist
                sample = dist.sample
            else:
                sample = categorical_sample

            params = project(hidden.tanh())

            if query.cases is None:
                result = sample(params, **query.kw)
            elif m in ('inst', 'pitch'):
                # weighted subsets case
                assert all(isinstance(s, Subset) for s in query.cases), query.cases
                if len(query.cases) > 1:
                    all_probs = params.softmax(-1)
                    probs = [
                        all_probs[...,s.values].sum(-1) * s.weight
                        for s in query.cases]
                    idx = categorical_sample(torch.tensor(probs).log())
                else:
                    idx = 0
                # sample from subset
                # TODO: handle case where user supplies cases and whitelist
                result = sample(
                    params, whitelist=query.cases[idx].sample_values, **query.kw)
            else:
                # weiighted ranges case
                assert all(isinstance(r, Range) for r in query.cases), query.cases
                if len(query.cases) > 1:
                    probs = [
                        (dist.cdf(params, r.hi) - dist.cdf(params, r.lo)
                        ) * r.weight
                        for r in query.cases]
                    # print(f'deep_query {m} {probs=}')
                    idx = categorical_sample(torch.tensor(probs).log())
                else:
                    idx = 0
                r = query.cases[idx]
                # sample from range
                # TODO: handle case where user supplies cases and truncate
                result = sample(
                    params, truncate=(r.sample_lo, r.sample_hi), **query.kw)


        if not result.isfinite().all():
            print('WARNING: nonfinite value {result=} {m=}')
            result.nan_to_num_(0)

        try:
            event[m] = result.item()
        except Exception:
            event[m] = result

        # print(f'{result=}')
        # embed, add to hidden, recurse into subquery
        if isinstance(query.then, Query) or hasattr(query.then, '__call__'):
            emb = embed(result)
            hidden = hidden + emb
            if (~hidden.isfinite()).any():
                raise Exception(f'{m=} {result=} {emb=}')
            return self._deep_query(query.then, hidden, event)
        else:
            event['path'] = query.then
            return event
            
    def query_tipv_onsets(self,
        min_time=None, max_time=None, 
        include_inst=None,
        include_pitch=None,
        truncate_quantile_time=None,
        truncate_quantile_pitch=None,
        rhythm_temp=None, timing_temp=None,
        min_vel=None, max_vel=None
        ):
        """
        for onset-only_models
        """
        q = Query(
            'time',
            truncate=(min_time or -torch.inf, max_time or torch.inf), 
            truncate_quantile=truncate_quantile_time,
            weight_top_p=rhythm_temp, component_temp=timing_temp,
            then=Query(
                'inst',
                whitelist=include_inst,
                then=Query(
                    'pitch',
                    whitelist=include_pitch,
                    truncate_quantile=truncate_quantile_pitch,
                    then=Query(
                        'vel',
                        truncate=(min_vel or 0.5, max_vel or torch.inf),
                    )
                )
            )
        )
        return self.deep_query(q)
    
    def query_itpv_onsets(self,
        min_time=None, max_time=None, 
        include_inst=None,
        include_pitch=None,
        truncate_quantile_time=None,
        truncate_quantile_pitch=None,
        rhythm_temp=None, timing_temp=None,
        min_vel=None, max_vel=None
        ):
        """
        for onset-only_models
        """
        q = Query(
            'inst',
            whitelist=include_inst,
            then=Query(
                'time',
                truncate=(min_time or -torch.inf, max_time or torch.inf), 
                truncate_quantile=truncate_quantile_time,
                weight_top_p=rhythm_temp, component_temp=timing_temp,
                then=Query(
                    'pitch',
                    whitelist=include_pitch,
                    truncate_quantile=truncate_quantile_pitch,
                    then=Query(
                        'vel',
                        truncate=(min_vel or 0.5, max_vel or torch.inf),
                    )
                )
            )
        )
        return self.deep_query(q)


    def query_vtip(self,
            note_on_map:Dict[int,List[int]], 
            note_off_map:Dict[int,List[int]], 
            min_time=None, max_time=None, 
            min_vel=None, max_vel=None, 
            rhythm_temp=None, timing_temp=None,
            truncate_quantile_time=None,
            truncate_quantile_pitch=None,
            steer_density=None, # truncate_quantile_type ? 
            inst_weights=None,
            no_steer=None, # TODO
            ):
        """ Query in velocity-time-instrument-pitch order,
            efficiently truncating the joint distribution to just allowable
            (velocity>0)/instrument/pitch triples.

            Args:
                note_on_map: possible note-ons as {instrument: [pitch]} 
                note_off_map: possible note-offs as {instrument: [pitch]} 
        """
        no_on = all(len(ps)==0 for ps in note_on_map.values())
        no_off = all(len(ps)==0 for ps in note_off_map.values())
        if no_on and no_off:
            raise ValueError(f"""
                no possible notes {note_on_map=} {note_off_map=}""")
        
        def get_subquery(ipm, path, weights=None):
            return Query(
                'time',
                truncate=(min_time or -torch.inf, max_time or torch.inf),
                truncate_quantile=truncate_quantile_time,
                weight_top_p=rhythm_temp, component_temp=timing_temp,
                then=Query(
                    'inst', [(
                        Subset([i], 1 if weights is None else weights[i]), 
                        Query('pitch', path, 
                            whitelist=list(ps), 
                            truncate_quantile=None if self.is_drum(i) else truncate_quantile_pitch)
                    ) for i,ps in ipm.items() if len(ps)]
                ),
            )
        w = 1 if steer_density is None else 2**(steer_density*2-1)
        
        w_on = 0 if no_on else w
        w_off = 0 if no_off else 1/w

        min_vel = max(0.5, 0 if min_vel is None else min_vel)
        max_vel = torch.inf if max_vel is None else max_vel
        
        return self.deep_query(Query('vel', [
            (Range(-torch.inf,0.5,w_off), get_subquery(note_off_map, 'note off')),
            (Range(0.5,torch.inf,w_on,min_vel,max_vel), get_subquery(note_on_map, 'note on', inst_weights))
        ]))
    
    # TODO: should be possible to implement velocity steering now
    # need syntax for keywords to vary by case;
    # then can put truncate_quantile in the noteon case
    def query_vipt(self,
        note_on_map=None, note_off_map=None,
        min_time=None, max_time=None, # affecting interevent time (not durations)
        min_vel=None, max_vel=None, # affecting note on only
        min_polyphony=None, max_polyphony=None, # scalar or per instrument
        min_duration=None, max_duration=None, # scalar or per instrument
        rhythm_temp=None, timing_temp=None,
        truncate_quantile_time=None,
        truncate_quantile_pitch=None,
        steer_density=None,
        inst_weights=None,
        no_steer=None,
        ):
        """
        Query in velocity-instrument-pitch-time order,
            efficiently truncating the joint distribution to just allowable
            (velocity>0)/instrument/pitch triples.

            Args:
                note_on_map: possible note-ons as {instrument: [pitch]} 
                note_off_map: possible note-offs as {instrument: [pitch]} 
        """
        min_time = min_time or 0
        max_time = max_time or torch.inf

        inst_weights = inst_weights or {}

        # convert {(i,p):t} to {i:[p]}
        held_map = {}
        for i,p in self.held_notes:
            if i not in held_map:
                held_map[i] = set()
            held_map[i].add(p)

        # get default note_on_map (anything)
        if note_on_map is None:
            note_on_map = {
                i:range(128) for i in range(1,self.instrument_domain+1)}
            
        # note offs can be any from the note_on instruments by default
        # but users can also supply this themselves
        if note_off_map is None:
            note_off_map = {
                i: held_map[i] 
                for i in note_on_map 
                if i in held_map}
            
        # exclude held notes for note on
        for i in held_map:
            note_on_map[i] = set(note_on_map[i]) - held_map[i]

        # exclude non-held notes for note off
        note_off_map = {
            i: set(note_off_map[i]) & held_map[i]
            for i in note_off_map
            if i in held_map
        }

        # TODO: break out application of these constraints to helper functions;
        # allow breaking each constraint when it results in no options

        max_poly = get_from_scalar_or_dict(max_polyphony, torch.inf)
        min_poly = get_from_scalar_or_dict(min_polyphony, 0)

        # prevent note on if polyphony exceeded
        for i in list(note_on_map):
            if len(held_map.get(i, [])) >= max_poly(i):
                note_on_map.pop(i)

        # prevent note off if below minimum polyphony
        for i in list(note_off_map):
            if len(held_map[i]) <= min_poly(i):
                note_off_map.pop(i)

        max_dur = get_from_scalar_or_dict(max_duration, torch.inf)
        min_dur = get_from_scalar_or_dict(min_duration, 0)

        soonest_off = {
            (i,p):max(min_time, min_dur(i) - self.held_notes[(i,p)]) 
            for i,ps in note_off_map.items()
            for p in ps}
        
        # latest possible event is minimum max remaining duration over all held notes  
        latest_event = max_time
        for (i,p),t in self.held_notes.items():
            latest_event = min(latest_event, max_dur(i) - t)
        # slip to accomodate global constraint
        latest_event = max(min_time, latest_event)

        # illegal note offs:
        #   the soonest possible note-off is later than the latest possible event
        new_note_off_map = {}
        for i,ps in note_off_map.items():
            for p in ps:
                if soonest_off[(i,p)] <= latest_event:
                    if i not in new_note_off_map:
                        new_note_off_map[i] = []
                    new_note_off_map[i].append(p)
        old_note_off_map = note_off_map
        note_off_map = new_note_off_map

        no_on = all(len(ps)==0 for ps in note_on_map.values())
        no_off = all(len(ps)==0 for ps in note_off_map.values())
        no_off_old = all(len(ps)==0 for ps in old_note_off_map.values())

        if no_on and no_off:
            if no_off_old:
                raise ValueError(f"""
                    no possible notes {note_on_map=} {note_off_map=}""")
            # let the duration constraint slip
            note_off_map = old_note_off_map
            no_off = no_off_old
            for k in soonest_off:
                soonest_off[k] = latest_event

        def note_map(e):
            if e['vel'] > 0:
                m = note_on_map
            else:
                m = note_off_map
            i = e.get('inst')
            if i is not None:
                m = m[i]
            return m
                    
        w = 1 if steer_density is None else 2**(steer_density*2-1)
        
        w_on = 0 if no_on else w
        w_off = 0 if no_off else 1/w

        min_vel = max(0.5, 0 if min_vel is None else min_vel)
        max_vel = torch.inf if max_vel is None else max_vel
        
        return self.deep_query(Query(
            'vel', 
            cases=(
                Range(-torch.inf,0.5,w_off), 
                Range(0.5,torch.inf,w_on,min_vel,max_vel)),
            then=lambda e: Query(
                'inst', 
                whitelist={
                    k:inst_weights.get(k,1) if e['vel'] > 0 else 1 
                    for k in note_map(e)},
                then=lambda e: Query(
                    'pitch', 
                    whitelist=note_map(e),
                    truncate_quantile=(
                        None if self.is_drum(e['inst']) 
                        else truncate_quantile_pitch),
                    then=lambda e: Query(
                        'time', #'note on' if e['vel']>0 else 'note off',         
                        truncate=(
                            min_time if e['vel']>0 
                            else soonest_off[(e['inst'],e['pitch'])],
                            latest_event
                        ),
                        truncate_quantile=None if (
                            no_steer is not None and e['inst'] in no_steer
                            ) else truncate_quantile_time,
                        weight_top_p=rhythm_temp, 
                        component_temp=timing_temp
        )))))
    
    # def query_ipvt(self,
    #     note_map, 
    #     min_time=-torch.inf, max_time=torch.inf, 
    #     min_vel=-torch.inf, max_vel=torch.inf,
    #     truncate_quantile_time=None,
    #     truncate_quantile_pitch=None,
    #     ):
    #     """
    #     """
      
    #     return self.deep_query(Query(
    #         'inst', then=[(
    #             Subset([i]), Query(
    #                 'pitch', 
    #                 whitelist=list(ps), 
    #                 truncate_quantile=truncate_quantile_pitch,
    #                 then=Query(
    #                     'vel',
    #                     truncate=(min_vel or -torch.inf, max_vel or torch.inf),
    #                     then=Query(
    #                         'time',         
    #                         truncate=(min_time or -torch.inf, max_time or torch.inf), truncate_quantile=truncate_quantile_time
    #                     )
    #                 )
    #             )
    #         ) for i,ps in note_map.items() if len(ps)]
    #     ))

    # TODO: remove pitch_topk and sweep_time?
    # TODO: rewrite this to build queries and dispatch to deep_query
    def query(self,
            next_inst:int=None, next_pitch:int=None, 
            next_time:float=None, next_vel:int=None,

            allow_end:bool=False,

            include_inst:List[int]=None, exclude_inst:List[int]=None,
            allow_anon:bool=True, 
            instrument_temp:float=None, 

            include_pitch:List[int]=None, exclude_pitch:List[int]=None,
            include_drum:List[int]=None,
            truncate_quantile_pitch:Tuple[float,float]=None,
            pitch_temp:float=None, 
            index_pitch:int=None,

            min_time:float=None, max_time:float=None,
            truncate_quantile_time:Tuple[float, float]=None,
            rhythm_temp:float=None, timing_temp:float=None,

            min_vel:int=None, max_vel:int=None,
            velocity_temp:float=None,

            pitch_topk:int=None, sweep_time:bool=False, 

            handle:str=None, return_params:bool=False
            ) -> dict:
        """
        return a prediction for the next note.

        various constraints on the the next note can be requested.

        Args:
            # hard constraints

            next_inst: fix a particular instrument for the predicted note.
                sampled values will always condition on fixed values, so passing
                `next_inst=1`, for example, will make the event appropriate
                for the piano (instrument 1) to play.
            next_pitch: fix a particular MIDI number for the predicted note.
                sampled values will always condition on fixed values, so passing
                `next_pitch=60`, for example, will make the event appropriate
                for a middle C.
            next_time: fix a particular delta time for the predicted note.
                sampled values will always condition on fixed values, so passing
                `next_time=0`, for example, will make the event appropriate
                for one which is concurrent with the previous event.
            next_vel: fix a particular velocity for the predicted note.
                sampled values will always condition on fixed values, so passing
                `next_inst=0`, for example, will make the event appropriate
                for a noteOff (i.e., something which is currently playing).
                
            # partial constraints

            include_inst: instrument id(s) to include in sampling.
                (if not None, all others will be excluded)
            exclude_inst: instrument id(s) to exclude from sampling.
            allow_anon: bool. if False, zero probability of anon instruments

            include_pitch: pitch(es) to include in sampling.
                (if not None, all others will be excluded)
            exclude_pitch: pitch(es) to exclude from sampling.
            include_drum: like `include_pitch`, but only in effect when 
                instrument is a drumkit

            min_time: if not None, truncate the time distribution below
            max_time: if not None, truncate the time distribution above

            min_vel: if not None, truncate the velocity distribution below
                e.g., `min_vel=1` prevents NoteOff events
            max_vel: if not None, truncate the velocity distribution above

            allow_end: if False, zero probability of sampling the end marker

            # sampling strategies
            
            instrument_temp: if not None, apply top_p sampling to instrument. 0 is
                deterministic, 1 is 'natural' according to the model

            pitch_temp: if not None, apply top_p sampling to pitch. 0 is
                deterministic, 1 is 'natural' according to the model
            truncate_quantile_pitch: applied after include_pitch, exclude_pitch
                truncate the remaining pitch distribution by quantile.
                e.g. truncate_quantile_pitch=(0.25, 0.75)
                excludes the lowest- and highest- pitch 25% of probability mass
            index_pitch: if not None, deterministically take the
                nth most likely pitch instead of sampling.

            timing_temp: if not None, apply temperature sampling to the time
                component. this affects fine timing; 0 is deterministic and 
                precise, 1 is 'natural' according to the model.
            rhythm_temp: if not None, apply top_p sampling to the weighting
                of mixture components. this affects coarse rhythmic patterns;
                0 is deterministic, 1 is 'natural' according to the model
            truncate_quantile_time: applied after min_time, max_time
                truncate the remaining delta time distribution by quantile.
                e.g. truncate_quantile_time=(0.25, 0.75)
                excludes the soonest- and furthest- 25% of probability mass

            velocity_temp: if not None, apply temperature sampling to the velocity
                component.

            # multiple predictions

            pitch_topk: Optional[int]. if not None, instead of sampling pitch, 
                stack the top k most likely pitches along the batch dimension
            sweep_time: if True, instead of sampling time, choose a diverse set
            of times and stack along the batch dimension

            # other

            handle: metadata to be included in the returned dict, if not None
            return_params: if True, return tensors of distribution parameters
                under the keys `inst_params`, `pitch_params`, `time_params`,
                and `vel_params`.

        Returns:
            'inst': int. id of predicted instrument.
                1-128 are General MIDI standard melodic instruments
                129-256 are drumkits for MIDI programs 1-128
                257-288 are 'anonymous' melodic instruments
                289-320 are 'anonymous' drumkits
            'pitch': int. predicted MIDI number of next note, 0-128.
            'time': float. predicted time to next note in seconds.
            'vel': float. unquantized predicted velocity of next note.
                0-127; hard 0 indicates a note-off event.
            'end': int. value of 1 indicates the *current* event (the one 
                passed as arguments to `predict`) was the last event, and the
                predicted event should *not* be played. if `allow end` is false, 
                this will always be 0.
            'step': int. number of steps since calling `reset`.
            '*_params': tensor. distribution parameters for visualization
                and debugging purposes. present if `return_params` is True.

        NOTE: `instrument`, `pitch`, `time`, `velocity` may return lists,
            when using `sweep_time` or `pitch_topk`. that part of the API 
            is very experimental and likely to break.
        """
         # validate options:
        if (index_pitch is not None) and (pitch_temp is not None):
            print("warning: `index pitch` overrides `pitch_temp`")

        inst_intervention = any(p is not None for p in (
            instrument_temp, include_inst, exclude_inst))

        pitch_intervention = (pitch_topk or any(p is not None for p in (
            pitch_temp, include_pitch, exclude_pitch, include_drum)))

        time_intervention = any(p is not None for p in (
            min_time, max_time, rhythm_temp, timing_temp))

        vel_intervention = any(p is not None for p in (
            min_vel, max_vel, velocity_temp))

        exclude_inst = arg_to_set(exclude_inst)
        if not allow_anon:
            exclude_inst |= set(range(257, 321))
        constrain_inst = list((
            set(range(self.instrument_domain)) - {self.instrument_start_token}
            if include_inst is None 
            else arg_to_set(include_inst)
        ) - exclude_inst)
        if len(constrain_inst)==0:
            raise ValueError("""
            every instrument has been excluded. check values of 
            `include_inst` and `exclude_inst`
            """)
        # elif len(constrain_inst)==1:
        #     print("""
        #     warning: recommended to use `next_inst`, not 
        #     `include_inst` to allow only one specific instrument
        #     """)
        
        constrain_pitch = list((
            set(range(self.pitch_domain)) - {self.pitch_start_token}
            if include_pitch is None 
            else arg_to_set(include_pitch)
        ) - arg_to_set(exclude_pitch))
        if len(constrain_pitch)==0:
            raise ValueError("""
            every pitch has been excluded. check values of 
            `include_pitch` and `exclude_pitch`
            """)
        elif len(constrain_pitch)==1:
            print("""
            warning: recommended to use `next_pitch`, not 
            `include_pitch` to allow only one specific pitch
            """)

        # TODO: this got really complicated to support include_drum...
        # really want to edit the whole joint distribution of pitch,inst in 
        # cases where certain pitches or drums need to be excluded...
        # would that be practical? if there are ~40000 inst x pitch combos?
        # would need to run the instrument head for a whole batch of all
        # allowable pitches or vice-versa...
        def sample_instrument(x):
            # if include_drum is supplied, make sure to exclude drum instruments
            # when no pitch is in the allowed drums
            if include_drum is not None:
                pit = predicted_by_name('pitch')
                pits = [pit] if pit is not None else constrain_pitch
                if pits is not None and all(pit not in include_drum for pit in pits):
                    nonlocal constrain_inst
                    if constrain_inst is None:
                        constrain_inst = range(1,self.instrument_domain)
                    constrain_inst = [
                        i for i in constrain_inst if not self.is_drum(i)]

            # if constrain_inst is not None:
            #     preserve_x = x[...,constrain_inst]
            #     x = torch.full_like(x, -torch.inf)
            #     x[...,constrain_inst] = preserve_x
            # probs = x.softmax(-1)
            # if instrument_temp is not None:
            #     probs = reweight_top_p(probs, instrument_temp)
            # return D.Categorical(probs).sample()

            return categorical_sample(x, 
                whitelist=constrain_inst,
                top_p=instrument_temp)

        def sample_pitch(x):
            # conditional constraint
            if include_drum is not None:
                # if this event is / must be a drum,
                # use include_drum instead of constrain_inst
                inst = predicted_by_name('instrument')
                insts = [inst] if inst is not None else constrain_inst
                if insts is not None and all(self.is_drum(i) for i in insts):
                    nonlocal constrain_pitch
                    constrain_pitch = include_drum

            if pitch_topk is not None:
                raise NotImplementedError

            return categorical_sample(x,
                whitelist=constrain_pitch, 
                index=index_pitch,
                top_p=pitch_temp,
                truncate_quantile=truncate_quantile_pitch
                )
            # if constrain_pitch is not None:
            #     preserve_x = x[...,constrain_pitch]
            #     x = torch.full_like(x, -torch.inf)
            #     x[...,constrain_pitch] = preserve_x
            # # x is modified logits

            # if index_pitch is not None:
            #     return x.argsort(-1, True)[...,index_pitch]
            # elif pitch_topk is not None:
            #     return x.argsort(-1, True)[...,:pitch_topk].transpose(0,-1)
            
            # probs = x.softmax(-1)
            # if pitch_temp is not None:
            #     probs = reweight_top_p(probs, pitch_temp)

            # if steer_pitch is not None:
            #     return steer_categorical(probs, steer_pitch)
            # else:
            #     return D.Categorical(probs).sample()

        def sample_time(x):
            # TODO: respect trunc_time when sweep_time is True
            if sweep_time:
                if min_time is not None or max_time is not None:
                    raise NotImplementedError("""
                    min_time/max_time with sweep_time needs implementation
                    """)
                assert x.shape[0]==1, "batch size should be 1 here"
                log_pi, loc, s = self.time_dist.get_params(x)
                idx = log_pi.squeeze().argsort()[:9]
                loc = loc.squeeze()[idx].sort().values[...,None] 
                # multiple times in batch dim
                # print(loc.shape)
                return loc
            
            trunc = (
                -torch.inf if min_time is None else min_time,
                torch.inf if max_time is None else max_time)

            return self.time_dist.sample(x, 
                truncate=trunc,
                component_temp=timing_temp, 
                weight_top_p=rhythm_temp,
                truncate_quantile=truncate_quantile_time
                )

        def sample_velocity(x):
            trunc = (
                -torch.inf if min_vel is None else min_vel,
                torch.inf if max_vel is None else max_vel)
            return self.vel_dist.sample(
                x, component_temp=velocity_temp, truncate=trunc,
                # truncate_quantile=truncate_quantile_vel
                )

        with torch.inference_mode():
            if self.h_query is None:
                self.h_query = self.h_proj(self.h)

            modalities = list(zip(
                self.projections,
                (sample_instrument, sample_pitch, sample_time, sample_velocity),
                self.embeddings,
                ))

            context = [self.h_query] # embedded outputs for autoregressive prediction
            predicted = [] # raw outputs
            params = [] # distribution parameters for visualization

            fix = [
                None if item is None else torch.tensor([[item]], dtype=dtype)
                for item, dtype in zip(
                    [next_inst, next_pitch, next_time, next_vel],
                    [torch.long, torch.long, torch.float, torch.float])]

            # if any modalities are determined, embed them
            # sort constrained modalities before unconstrained
            # TODO: option to skip modalities
            det_idx, cons_idx, uncons_idx = [], [], []
            for i,(item, embed) in enumerate(zip(fix, self.embeddings)):
                if item is None:
                    if (
                        i==0 and inst_intervention or
                        i==1 and pitch_intervention or
                        i==2 and time_intervention or
                        i==3 and vel_intervention):
                        cons_idx.append(i)
                    else:
                        uncons_idx.append(i)
                else:
                    det_idx.append(i)
                    context.append(embed(item))
                    predicted.append(item)
                    params.append(None)
            undet_idx = cons_idx + uncons_idx
            perm = det_idx + undet_idx # permutation from the canonical order
            iperm = np.argsort(perm) # inverse permutation back to canonical order

            mode_names = ['instrument', 'pitch', 'time', 'velocity']
            name_to_idx = {k:v for k,v in zip(mode_names, iperm)}
            def predicted_by_name(name):
                idx = name_to_idx[name]
                if len(predicted) > idx:
                    return predicted[idx]
                return None
            # print('sampling order:', [mode_names[i] for i in perm])

            # for each undetermined modality, 
            # sample a new value conditioned on already determined ones
            
            running_ctx = sum(context)
            # print(running_ctx)
            # perm_h_tgt = [h_tgt[i] for i in perm]
            while len(undet_idx):
                # print(running_ctx.norm())
                i = undet_idx.pop(0) # index of modality to determine
                # j = len(det_idx) # number already determined
                project, sample, embed = modalities[i]
                # determine value for the next modality
                hidden = running_ctx.tanh()
                params.append(project(hidden))
                pred = sample(params[-1])
                predicted.append(pred)
                # prepare for next iteration
                if len(undet_idx):
                    # context.append(embed(pred))
                    running_ctx += embed(pred)
                det_idx.append(i)

            pred_inst = predicted_by_name('instrument')
            pred_pitch = predicted_by_name('pitch')
            pred_time = predicted_by_name('time')
            pred_vel = predicted_by_name('velocity')

            if allow_end:
                end_params = self.end_proj(self.h)
                # print(end_params)
                end = D.Categorical(logits=end_params).sample()
            else:
                end = torch.zeros(self.h.shape[:-1])

            if sweep_time or pitch_topk:
                # return lists of predictions
                pred_inst = [x.item() for x in pred_inst]
                pred_pitch = [x.item() for x in pred_pitch]
                pred_time = [x.item() for x in pred_time]
                pred_vel = [x.item() for x in pred_vel]
                end = [x.item() for x in end]
                # print(pred_time, pred_pitch, pred_vel)
            else:
                # return single predictions
                pred_inst = pred_inst.item()
                pred_pitch = pred_pitch.item()
                pred_time = pred_time.item()
                pred_vel = pred_vel.item()
                end = end.item()

            r = {
                'inst': pred_inst,
                'pitch': pred_pitch, 
                'time': pred_time,
                'vel': pred_vel,

                'end': end,
                'step': self.step,
            }

            if handle is not None:
                r['handle'] = handle

            if return_params:
                r |= {
                    'inst_params': params[iperm[0]],
                    'pitch_params': params[iperm[1]],
                    'time_params': params[iperm[2]],
                    'vel_params': params[iperm[3]]
                }

            return r

    def predict(self, inst, pitch, time, vel, **kw):
        """
        DEPRECATED: alias for feed_query
        """
        self.feed(inst, pitch, time, vel)
        return self.query(**kw)

    def feed_query(self, inst:int, pitch:int, time:Number, vel:Number, 
 **kw):
        """
        feed an event to the model, 
        then query for the next predicted event and return it.
        """
        self.feed(inst, pitch, time, vel)
        return self.query(**kw)

    def query_feed(self, *a, **kw):
        """
        query for the next predicted event and immediately feed it to the model,
        also returning the predicted event.
        """
        r = self.query(*a, **kw)
        self.feed(r['inst'], r['pitch'], r['time'], r['vel'])
        return r

    def feed_query_feed(self, 
            inst:int, pitch:int, time:Number, vel:Number, 
            **kw):
        """
        given an event, return the next predicted event, 
        feeding both to the model.
        """ 
        self.feed(inst, pitch, time, vel)
        return self.query_feed(**kw)
    
    def reset(self, start=None, state=None):
        """
        resets internal model state.
        Args:
            start: if True, send start tokens through the model
                default behavior is True when state=None, False otherwise
            state: set the state from a result of `get_state`,
                instead of the initial state
        """
        self.current_time = 0
        self.held_notes.clear()
        self.step = 0
        if start is None:
            start = state is None
        if state is None: 
            named_states = zip(self.cell_state_names(), self.initial_state)
        else:
            named_states = state.items()
        with torch.inference_mode():
            for n,t in named_states:
                getattr(self, n)[:] = t
            if start:
                self.feed(
                    self.instrument_start_token, self.pitch_start_token, 0., 0.)
        # for n,t in zip(self.cell_state_names(), self.initial_state):
        #     getattr(self, n)[:] = t.detach()
        # if start:
        #     self.feed(
        #         self.instrument_start_token, self.pitch_start_token, 0., 0.)

    def get_state(self) -> Dict[str, torch.Tensor]:
        """return a dict of {str:Tensor} representing the model state"""
        return {n:getattr(self, n).clone() for n in self.cell_state_names()}
                

    @classmethod
    def user_data_dir(cls):
        return _user_data_dir()

    @classmethod
    def from_checkpoint(cls, path):
        """
        create a Notochord from a checkpoint file containing 
        hyperparameters and model weights.

        Args:
            path: file path to Notochord model
        """
        if path=="notochord-latest.ckpt":
            url = 'https://github.com/Intelligent-Instruments-Lab/iil-python-tools/releases/download/notochord-v0.4.0/notochord_lakh_50G_deep.pt'
        elif path=="txala-latest.ckpt":
            url = 'https://github.com/Intelligent-Instruments-Lab/notochord/releases/download/notochord-v0.5.4/noto-txala-011-0020.ckpt'
        else:
            url = None

        if url is not None:
            d = Notochord.user_data_dir()
            path = d / path
            # maybe download
            if not path.is_file():
                while True:
                    answer = input("Do you want to download a notochord model? (y/n)")
                    if answer.lower() in ["y","yes"]:
                        download_url(url, path)
                        print(f'saved to {path}')
                        break
                    if answer.lower() in ["n","no"]:
                        break
        # path = 
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model = cls(**checkpoint['kw']['model'])
        model.load_state_dict(checkpoint['model_state'], strict=False)
        model.checkpoint_path = path
        model.reset()
        model.eval()
        return model
    
    def prompt(self, midi_file):
        """Read a MIDI file and feed events to this Notochord model.

        When possible, the hidden states will be cached so re-using the same prompt will be fast.

        Args:
            midi_file: path of a midi file to read
        Returns:
            state: hidden state dict of the Notochord encoding the MIDI prompt
            channel_inst: dict mapping MIDI channel (0-index) to Notochord instrument (1-256)
        """
        return prompt(self, Path(midi_file), state_hash=hash_states(self.get_state()))
    
def hash_states(s):
    if isinstance(s, dict):
        return {k:hash_states(v) for k,v in s.items()}
    elif isinstance(s, torch.Tensor):
        return hash_tensor(s)
    return s

def hash_tensor(t):
    return hashlib.md5(json.dumps(t.tolist()).encode('utf-8')).digest()

@mem.cache(ignore=('noto',))
def prompt(noto:Notochord, midi_file:str|Path, state_hash:int|None=None): 
    # state_hash is used for disk cache only
    """Read a MIDI file and feed events to a Notochord model.

    Args:
        noto: a Notochord
        midi_file: path of a midi file to read
        state_hash: representation of model hidden state to use for caching results
    Returns:
        state: hidden state dict of the Notochord encoding the MIDI prompt
        channel_inst: dict mapping MIDI channel (0-index) to Notochord instrument (1-256)
    """
    # TODO: deduplicate this code?
    class AnonTracks:
        def __init__(self):
            self.n = 0
        def __call__(self):
            self.n += 1
            return 256+self.n
    next_anon = AnonTracks()
    mid_channel_inst = defaultdict(next_anon)

    mid = mido.MidiFile(midi_file)
    ticks_per_beat = mid.ticks_per_beat
    us_per_beat = 500_000
    time_seconds = 0
    prev_time_seconds = 0
    event_count = defaultdict(int)
    print(f'MIDI file: {ticks_per_beat} ticks, {us_per_beat} μs per beat')

    for msg in tqdm(mid, desc='ingesting MIDI prompt'):
        # when iterating over a track this is ticks,
        # when iterating the whole file it's seconds
        time_seconds += msg.time

        if msg.type=='program_change':
            mid_channel_inst[msg.channel] = msg.program + 1 + 128*int(msg.channel==9)
            tqdm.write(str(msg))
            
        elif msg.type=='set_tempo':
            us_per_beat = msg.tempo
            tqdm.write(f'MIDI file: set tempo {us_per_beat} μs/beat at {time_seconds} seconds')
            
        elif msg.type in ('note_on', 'note_off'):
            # make channel 10 with no PC standard drumkit
            if msg.channel not in mid_channel_inst and msg.channel==9:
                mid_channel_inst[msg.channel] = 129

            event_count[mid_channel_inst[msg.channel]] += 1
            noto.feed(
                mid_channel_inst[msg.channel],
                msg.note,
                time_seconds - prev_time_seconds,
                msg.velocity if msg.type=='note_on' else 0)
            prev_time_seconds = time_seconds

        else: continue

    print(f'MIDI file: {sum(event_count.values())} events in {time_seconds} seconds')
    # print(event_count)
    # print(mid_channel_inst)  
    return noto.get_state(), dict(mid_channel_inst)