import math
from typing import List, Tuple, Dict, Union, Any, Sequence, Mapping, MutableMapping, TypeVar
from numbers import Number
from collections import namedtuple, defaultdict
from pathlib import Path
import hashlib
import json
import copy
import traceback
import time
import random

import mido
from tqdm import tqdm
import appdirs
import joblib

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as D

from .rnn import GenericRNN
from .distributions import CensoredMixtureLogistic, categorical_sample

from .util import arg_to_set, download_url, argsort
from .event import *

class NoPossibleEvents(Exception):
    pass

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
    def __init__(self, lo:float=-torch.inf, hi:float=torch.inf, weight:float=1, sample_lo:float=None, sample_hi:float=None, **kw):
        """use lo, hi when computing weights of each branch; but actually sample between sample_lo and sample_hi. for example, you could let lo,hi cover the full range to compute the true model on/off ratio, but sample from a narrow range of allowed velocities in the noteOn case.

        **kw gets passed to sample once the case is selected
        """
        self.lo = lo
        self.hi = hi
        self.weight = weight
        self.sample_lo = lo if sample_lo is None else sample_lo
        self.sample_hi = hi if sample_hi is None else sample_hi
        self.kw = kw

class Subset:
    def __init__(self, values=None, weight=1, sample_values=None, **kw):
        """**kw gets passed to sample once the case is selected"""
        self.values = values
        self.weight = weight
        self.sample_values = values if sample_values is None else sample_values
        self.kw = kw

# def get_from_scalar_or_dict(x:Any, default:Any=None):
ScalarT = TypeVar('ScalarT')
def get_from_scalar_or_dict(
        x:None|ScalarT|Mapping[int,ScalarT], default:ScalarT):
    """
    """
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
    def __init__(self, input, hidden, output, layers, dropout=0.0, norm:str=None):
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
        net = []
        for _ in range(layers):
            net.append(nn.Sequential(
                *get_dropout(), nn.Linear(h, hidden*2), GLU(), *get_norm()))
            h = hidden
        net.append(nn.Linear(hidden, output))
        self.net = nn.Sequential(*net)

        with torch.no_grad():
            self.net[-1].weight.mul_(1e-2) # type: ignore

    def forward(self, x):
        return self.net(x)


class Notochord(nn.Module):
    # note: use named arguments only for benefit of training script
    def __init__(self, 
            emb_size:int=512, 
            rnn_hidden:int=1024, 
            rnn_layers:int=2, 
            kind:str='gru', 
            mlp_layers:int=2,
            dropout:float=0.1,
            norm:str='layer',
            num_pitches:int=128, 
            num_instruments:int=320,
            time_sines:int=128, 
            vel_sines:int=128,
            time_bounds:tuple[float, float]=(0.,10.), 
            time_components:int=32, 
            time_res:float=1e-2,
            vel_components:int=16
            ):
        """
        Args:
            emb_size: Embedding dimension and size parameter for most layers
            rnn_hidden: size parameter for the backbone RNN
            rnn_layers: number of layers in backbone RNN
            kind: type of backbone RNN ('gru', 'lstm' or 'elman')
            mlp_layers: layers in each prediction head plus shared projection
            dropout: dropout probability during training
            norm: normalization layer type ('layer' or None)
            num_pitches: number of discrete pitches (leave at 128 for MIDI)
            num_instruments: number of discrete instruments 
                default is 128 melodic + 128 drums + 32 anon mel + 32 anon drum 
                don't change this unless you know what you are doing.
            time_sines: number of sinusoids in time embedding
            vel_sines: number of sinusoids in velocity embedding
            time_bounds: min and max inter-event time
            time_components: number of mixture components in time likelihood
            time_res: resolution parameter for discretized time likelihood
            vel_components: number of mixture components in velocity likelihood
        """
        super().__init__()

        self.step = 0
        self.current_time = 0.0
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
            lo=0., hi=127., init='velocity')
        
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
                p.net[-1].weight.mul_(1e-2) # type: ignore
            self.end_proj.weight.mul(1e-2)

        # persistent RNN state for inference
        for n,t in zip(self.cell_state_names(), self.initial_state):
            self.register_buffer(n, t.clone())
        self.step = 0

        # volatile hidden states for caching purposes
        self.h = None
        self.h_query = None

    def device(self):
        return self.end_proj.weight.device

    def default_note_map(self):
        return {
            i:set(range(128)) 
            for i in range(1,self.instrument_domain)
            }      

    def cell_state_names(self):
        return tuple(f'cell_state_{i}' for i in range(len(self.initial_state)))
    
    def held_map(self):
        """
            currently held notes as a map from instrument to pitch set
        """
        held_map = {}
        for i,p in self.held_notes:
            if i not in held_map:
                held_map[i] = set()
            held_map[i].add(p)
        return held_map
    
    def get_note_maps(self, 
        note_on_map:NoteMap=None, 
        note_off_map:NoteMap=None, 
        min_polyphony=None, max_polyphony=None,
        no_steer=None
        ) -> Tuple[Dict[int,List[int]], Dict[int,List[int]]]:
        """common logic for v-first sampling"""
        # convert {(i,p):t} to {i:[p]}
        held_map = self.held_map()

        # get default note_on_map (anything)
        if note_on_map is None:
            note_on_map = self.default_note_map()
        else:
            note_on_map = {i:set(ps) for i,ps in note_on_map.items()}
        # exclude held notes for note on
        for i in held_map:
            if i in note_on_map:
                note_on_map[i] = set(note_on_map[i]) - held_map[i]

        # note offs can be any from the note_on instruments by default
        # but users can also supply this themselves
        # exclude non-held notes for note off
        if note_off_map is None:
            note_off_map = {
                i: held_map[i] 
                for i in note_on_map
                if i in held_map}
        else:
            note_off_map = {
                i:set(ps) & held_map[i] 
                for i,ps in note_off_map.items()
                if i in held_map}

        max_poly = get_from_scalar_or_dict(max_polyphony, torch.inf)
        min_poly = get_from_scalar_or_dict(min_polyphony, 0)
        if no_steer is None: no_steer = []

        # prevent note on if polyphony exceeded
        for i in list(note_on_map):
            if i in no_steer: continue
            if len(held_map.get(i, [])) >= max_poly(i): #type:ignore
                note_on_map.pop(i)

        # prevent note off if below minimum polyphony
        for i in list(note_off_map):
            if i in no_steer: continue
            if len(held_map[i]) <= min_poly(i): #type:ignore
                note_off_map.pop(i)

        return note_on_map, note_off_map #type:ignore

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
    
    def embedding(self, m:str):
        if m=='i': return self.instrument_emb
        elif m=='p': return self.pitch_emb
        elif m=='t': return self.time_emb
        elif m=='v': return self.vel_emb
        else: raise ValueError

    def projection(self, m:str):
        if m=='i': return self.projections[0]
        elif m=='p': return self.projections[1]
        elif m=='t': return self.projections[2]
        elif m=='v': return self.projections[3]
        else: raise ValueError

        
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
        # batch, time, emb_size
        inst_emb:torch.Tensor = self.instrument_emb(instruments) 
        pitch_emb:torch.Tensor = self.pitch_emb(pitches) 
        time_emb:torch.Tensor = self.time_emb(times) 
        vel_emb:torch.Tensor = self.vel_emb(velocities) 

        embs = (inst_emb, pitch_emb, time_emb, vel_emb)

        # feed to RNN backbone
        x = sum(embs)
        ## broadcast initial state to batch size
        initial_state = tuple(
            # 1 x batch x hidden 
            t.expand(self.rnn.num_layers, x.shape[0], -1).contiguous() # type: ignore
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

    # TODO: add a constructor argument to specify which are drums
    # hardcoded for now
    # 0 - start token
    # 1-128 - melodic
    # 129-256 - drums
    # 257-288 - anon melodic
    # 289-320 - anon drums
    def is_drum(self, inst):
        return inst > 128 and inst < 257 or inst > 288
    def is_anon(self, inst):
        return inst > 256
    def first_anon_like(self, inst):
        # TODO: add a constructor argument to specify how many anon
        # hardcoded for now
        return 289 if self.is_drum(inst) else 257
    def anon_like(self, i):
        n_anon = (self.instrument_domain - 257)//2
        i = self.first_anon_like(i)
        return range(i, i+n_anon)
        
    def feed(self, 
            inst:int|torch.LongTensor, 
            pitch:int|torch.LongTensor,
            time:float|torch.FloatTensor, 
            vel:float|torch.FloatTensor, 
            **kw):
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
            **kw: ignored (allows doing e.g. noto.feed(**noto.sample(...)))
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
            inst = torch.LongTensor([[inst]]).to(self.device()) # 1x1 (batch, time)
            pitch = torch.LongTensor([[pitch]]).to(self.device()) # 1x1 (batch, time)
            time = torch.FloatTensor([[time]]).to(self.device()) # 1x1 (batch, time)
            vel = torch.FloatTensor([[vel]]).to(self.device()) # 1x1 (batch, time)

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
                # print(f'{query.kw=}')
                result = sample(params, **query.kw)
                # print(f'{result=}, {query.kw=}, {event=}') ##DEBUG

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
                s = query.cases[idx]
                result = sample(
                    params, whitelist=s.sample_values, **query.kw, **s.kw) #type:ignore
            else:
                # weighted ranges case
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
                    params, truncate=(r.sample_lo, r.sample_hi), **query.kw, **r.kw) #type:ignore


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
            # print(f'{result.dtype} {result.dtype==torch.double}')
            # TODO prevent result from being double in the first place?
            if result.dtype==torch.double: result = result.float()
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
     
    def init_event(self, cons:EventConstraints) -> NotochordEvent:
        """
        create Support based on held notes and user-supplied note map, 
        as well as global time and velocity constraints
        """
        event = NotochordEvent()
        held_map = self.held_map()

        # get default note_on_map (anything)
        if cons.note_on_map is None:
            note_on_map = self.default_note_map()
        else:
            note_on_map = {i: set(ps) for i,ps in cons.note_on_map.items()}

        for i,ps in note_on_map.items():
            ps = ps - held_map.get(i,set())
            if len(ps):
                # print(f'init noteOn support for {ps}')
                # exclude held notes for note on
                event.support.add(SupportAtom(
                    inst={i}, 
                    pitch=ps, 
                    time=SupportRange(cons.min_time),
                    vel=SupportRange(
                        max(0.5, cons.get('min_vel', i) or 0),
                        cons.get('max_vel', i))
                    ))
                   

        # note offs can be any from the note_on instruments by default
        # but users can also supply this themselves
        if cons.note_off_map is None:
            note_off_map = {i: held_map.get(i, set()) for i in note_on_map}
        else:
            note_off_map = {i: set(ps) for i,ps in cons.note_off_map.items()}
            
        for i,ps in note_off_map.items():
            ps = ps & held_map.get(i, set())
            if len(ps):
                # print(f'init noteOff support for {ps}')
                event.support.add(SupportAtom(
                    inst={i}, 
                    pitch=ps, 
                    time=SupportRange(cons.min_time),
                    vel=SupportRange(hi=0.5)
                    ))
                
        if cons.vel is not None:
            event.set('v', cons.vel)
                
        if cons.time is not None:
            event.set('t', cons.time)
        elif cons.max_time < torch.inf:
            # make max interevent time a soft constraint
            event.support.penalize_later(
                cons.max_time, 
                cons.penalty[Constraint.MAX_TIME], 
                Constraint.MAX_TIME)
        
        return event
          
    def constrain_dur(self, 
            event:NotochordEvent, 
            cons:EventConstraints
        ):
        """
        """
        # max_dur = get_from_scalar_or_dict(cons.max_duration, torch.inf)
        # min_dur = get_from_scalar_or_dict(cons.min_duration, 0)

        # hard constraints:

        # if any notes are >= max dur, they must have immediate noteOff
        #     sampling can be skipped
        latest_event = cons.max_time or float('inf')
        for (i,p),t in self.held_notes.items():
            # don't try to end externally controlled notes
            if i in cons.external: continue
            # delta = max_dur(i) - t
            delta = cons.get('max_duration', i) - t
            latest_event = min(latest_event, delta)
            if delta <= max(cons.min_time, 10e-3):# and i not in no_steer:
                event.set('i', i)
                event.set('p', p)
                event.set('v', 0)
                event.set('t', cons.min_time)
                print(f'ending long note with {event=}')
                return
            
        # dur/poly interaction:
        #     if the instrument is <= min poly (and has available pitches), it must instead have immediate *noteOn* (so old note can then end)
        #     it is simpler to just prioritize the max dur here, and let constrain_poly deal with starting a new note immediately, although that would not guarantee legato playing, which might affect synth behavior? # TODO
        
        # soft constraints:

        # should we exclude external instruments from soft constraints?
        # max dur: no, this is a constraint on all events; predictions of 
        # external instruments must still respect it, otherwise it loses effect
        # min dur: i think it doesn't matter much?

        # whatever note must end soonest (max dur) is the time of latest possible event, for all atoms -- split atoms on time and penalize late
        # print(f'clipping max time to {latest_event=} while {cons.min_time=}')
        event.support.penalize_later(
            latest_event, 
            cons.penalty[Constraint.MAX_DUR], 
            Constraint.MAX_DUR)

        # penalize noteOffs which violate min dur for held notes
        # NOTE complexity is held notes * atoms...
        # if atoms were indexed by note, it could be faster
        for (i,p),t in self.held_notes.items():
            if i in cons.external: continue
            # delta = min_dur(i) - t
            delta = cons.get('min_duration', i) - t
            if delta > 10e-3:
                # print(f'clipping min time for {i=} {p=} to {delta}')
                event.support.penalize_earlier_noteoff(
                    delta, penalty=cons.penalty[Constraint.MIN_DUR], 
                    pitch=p, inst=i)


    def constrain_poly(self, 
            event:NotochordEvent, 
            cons:EventConstraints
        ):
        """
        apply polyphony constraints to the Support.

        hard:
        - any instruments > max poly must have NoteOff immediately.
            pitch must still be sampled
        - any instruments < min poly must have NoteOn immediately.
            pitch and velocity must still be sampled

        soft:
        - any instruments == min poly, penalize NoteOff
        - any instruments == max poly, penalize NoteOn
        """
        # max_poly = get_from_scalar_or_dict(cons.max_polyphony, torch.inf)
        # min_poly = get_from_scalar_or_dict(cons.min_polyphony, 0)

        poly = defaultdict(int)
        for (i,_) in self.held_notes:
            poly[i] += 1

        # TODO: external instruments must be excluded, otherwise it will
        # get stuck waiting whenever external polyphony is outside constraints
        insts = event.support.marginal_inst() - cons.external

        for i in insts:
            # if poly[i] > max_poly(i):
            if poly[i] > cons.get('max_polyphony',i):
                event.set('i', i)
                event.set('t', cons.min_time)
                event.set('v', 0)
                print(f'ending because over polyphony with {event=}')
                return

        for i in insts:
            # if poly[i] < min_poly(i):
            if poly[i] < cons.get('min_polyphony', i):
                event.set('i', i)
                event.set('t', cons.min_time)
                event.support.remove_off()
                print(f'starting because under polyphony with {event=}')
                return

        for i in insts:
            # if poly[i] >= max_poly(i):
            if poly[i] >= cons.get('max_polyphony',i):
                event.support.penalize_on(
                    inst=i, penalty=cons.penalty[Constraint.MAX_POLY])
            # if poly[i] <= min_poly(i):
            if poly[i] <= cons.get('min_polyphony',i):
                event.support.penalize_off(
                    inst=i, penalty=cons.penalty[Constraint.MIN_POLY])

    def _sample(self, 
            modality:str,
            ctx:torch.Tensor, 
            event:NotochordEvent, 
            cons:EventConstraints
        ):
        params = self.projection(modality)(ctx.tanh())

        no_steer = (
            event.inst is not None 
            and event.inst in cons.external)
        if modality=='i':
            # apply instrument weights
            for i, w in cons.inst_weights.items():
                params[...,i] += math.log(w)
            return categorical_sample(
                params, 
                whitelist=event.support.marginal_inst(),
                )
        elif modality=='p':
            is_drum = (
                event.inst is not None 
                and self.is_drum(event.inst))
            tqp = (
                None if no_steer or is_drum
                else cons.truncate_quantile_pitch)
            return categorical_sample(
                params, 
                whitelist=event.support.marginal_pitch(),
                top_p=cons.pitch_temp,
                truncate_quantile=tqp
                )
        elif modality=='t':
            tqt = None if no_steer else cons.truncate_quantile_time 
            return self.time_dist.sample(
                params,
                truncate=event.support.marginal_time(),
                truncate_quantile=tqt,
                weight_top_p=cons.rhythm_temp,
                component_temp=cons.timing_temp,
            )
        elif modality=='v':
            # steer_density
            w_on = (
                1 if cons.steer_density is None 
                else 2**(cons.steer_density*2-1))
            w_off = 1/w_on
            vs = event.support.marginal_vel()
            for r in vs.ranges:
                if r.hi <= 0.5:
                    r.weight = w_off
                elif r.lo >= 0.5:
                    r.weight = w_on
            tqv = None if no_steer else cons.truncate_quantile_vel 
            return self.vel_dist.sample(
                params,
                truncate=vs,
                truncate_quantile=tqv
            ).round()
        else:
            raise ValueError

    def sample(self, 
            cons:EventConstraints=None, 
            order:str='vtip',
            predict_end=True
        ):
        """
        """
        if cons is None: cons = EventConstraints()

        if order=='random':
            order_ = list('iptv')
            random.shuffle(order_)
            order = ''.join(order_)

        # t = time.time_ns()
        event = self.init_event(cons)
        event.autoset()
        if event.is_complete(): 
            return event
        # print('after init: ', event.support)
        
        self.constrain_dur(event, cons)
        event.autoset()
        if event.is_complete(): 
            return event
        # print('after dur: ', event.support)
        
        self.constrain_poly(event, cons)
        event.autoset()

        if event.support.empty():
            raise NoPossibleEvents

        # enforce constraints, but let them break according to penalty
        # scores if all can't be satisfied
        # print(f'{self.held_notes=}')
        event.support.stratify()
        event.autoset()


        # cons_t = round((time.time_ns() - t)*1e-6)
        # print(f'{cons_t=}ms')

        # TODO: do we need to re-stratify after each sample?
        # sampling can't add any penalty
        # it also can't change the stratum
        # whatever atom is sample from *must* remain valid, right?

        processed = ''
        if order=='auto':
            # hmmm....
            # it should be possible to check the prob.mass of support
            # for each modality and pick the smallest one??
            # not free, it requires more projections
            raise NotImplementedError
        else:
            def next_modality():
                fixed, unfixed = '', ''
                for m in order:
                    if m not in processed:
                        if event.modality(m) is None:
                            unfixed += m
                        else:
                            fixed += m
                return (fixed + unfixed)[0]

        with torch.inference_mode():
            if self.h_query is None:
                self.h_query = self.h_proj(self.h)
            ctx = self.h_query

            while not event.is_complete():
                m = next_modality()
                processed += m
                # print(f'{m=} {order=}')
                value = event.modality(m)

                if value is None:
                    # sample next modality
                    value = self._sample(m, ctx, event, cons)
                    # print(f'set {m=} {value=}')
                    event.set(m, value.item())
                    event.autoset()
                
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor([[value]], device=ctx.device)
                    
                # embed
                ctx = ctx + self.embedding(m)(value)

        # query_t = round((time.time_ns() - t)*1e-6)
        # print(f'{query_t=}ms')

        # print(f'{event=}')
            if predict_end:
                end_params = self.end_proj(self.h)
                event.end = end_params.softmax(-1)[...,1].item()

        return event

    # TODO: should be possible to constrain duration per (i,p) pair,
    # not just per instrument?
    def query_vtip(self,
        note_on_map:dict[int,set[int]]|None=None, 
        note_off_map:dict[int,set[int]]|None=None,
        min_time:float|None=None, max_time:float|None=None,
        min_vel:float|None=None, max_vel:float|None=None,
        min_polyphony:dict[int,int]|int|None=None, 
        max_polyphony:dict[int,int]|int|None=None,
        min_duration:dict[int,float]|float|None=None, 
        max_duration:dict[int,float]|float|None=None, 
        pitch_temp:float|None=None,
        rhythm_temp:float|None=None, timing_temp:float|None=None,
        truncate_quantile_time:Tuple[float,float]|None=None,
        truncate_quantile_pitch:Tuple[float,float]|None=None,
        truncate_quantile_vel:Tuple[float,float]|None=None,
        steer_density:float|None=None,
        inst_weights:dict[int,float]|None=None,
        no_steer:set[int]|set=None,
        ):
        """
        Query in a fixed velocity->time->instrument->pitch order, sampling all
        modalities. Because velocity is sampled first, this query method can 
        automatically prevent double NoteOn or NoteOff. It's also possible to
        make some more detailed constraints per-instrument compared to `query`,
        including note duration constraints which can eliminate stuck notes.

        query_vipt is similar, but makes different compromises in applying 
        constraints. VTIP is likely to be better when setting min_time > 0 
        or otherwise heavily constraining time delta, while VIPT may be better
        in other cases.

        Args:
            note_on_map: possible note-ons as {instrument: [pitch]} 
                defaults to allowing any note. Notes already playing on a given
                instrument are always excluded.
            note_off_map: possible note-offs as {instrument: [pitch]}
                defaults to using only the instruments in note_on_map. 
                Notes not already playing on a given instrument are 
                automatically excluded.
            min_time: global minimum interevent time (default 0)
            max_time: global maximum interevent time (default no limit)
            min_vel: global minimum velocity for NoteOn events (default 1)
            max_vel: global maximum velocity for NoteOn events (default 127)
            min_polyphony: minimum number of concurrent notes per instrument.
                (default 0). Can be a dict mapping instrument to value,
                or a single value for all instruments.
                When an instrument has <= min polyphony, exclude NoteOffs
            max_polyphony: minimum number of concurrent notes per instrument.
                (default no limit). Can be a dict mapping instrument to value,
                or a single value for all instruments.
                When an instrument has >= max polyphony, exclude NoteOns.
            min_duration: minimum note length per instrument (default 0). Can   
                be a dict mapping instrument to value, or a single value for 
                all instruments.
            max_duration: maximum note length per instrument (default 0). Can   
                be a dict mapping instrument to value, or a single value for 
                all instruments.
            pitch_temp: if not None, apply top_p sampling to the pitch.
            rhythm_temp: if not None, apply top_p sampling to the weighting
                of mixture components. this affects coarse rhythmic patterns;
                0 is deterministic, 1 is 'natural' according to the model.
            timing_temp: if not None, apply temperature sampling to the time
                component. this affects fine timing; 0 is deterministic and 
                precise, 1 is 'natural' according to the model.
            truncate_quantile_time: applied after min_time, max_time
                truncate the remaining delta time distribution by quantile.
                e.g. truncate_quantile_time=(0.25, 0.75)
                excludes the shortest and longest 25% of probability mass.
            truncate_quantile_pitch: truncate the pitch distribution by 
                quantile. e.g. truncate_quantile_pitch=(0.5, 1) always samples
                above the median predicted pitch. Ignored for drums.
            truncate_quantile_vel: truncate the velocity distribution by 
                quantile. e.g. truncate_quantile_vel=(0, 0.5) always
                samples below the median predicted velocity. Affects only NoteOn.
            steer_density: adjust relative weight of NoteOn and NoteOff.
                values above 0.5 favor NoteOn, values below 0.5 favor NoteOff.
            inst_weights: multiplicatively adjust instrument probabilities. 
                Any instrument not included has a weight of 1. 0 would exclude
                an instrument completely (but better to do so via note_on_map)
            no_steer: collection of instruments to exclude from effect of 
                truncate_quantile_pitch.
        """
        cons_kw = dict(
            note_on_map=note_on_map,
            note_off_map=note_off_map,
            min_time=min_time,
            max_time=max_time,
            min_vel=min_vel,
            max_vel=max_vel,
            min_polyphony=min_polyphony,
            max_polyphony=max_polyphony,
            min_duration=min_duration,
            max_duration=max_duration,
            pitch_temp=pitch_temp,
            rhythm_temp=rhythm_temp,
            timing_temp=timing_temp,
            truncate_quantile_time=truncate_quantile_time,
            truncate_quantile_pitch=truncate_quantile_pitch,
            truncate_quantile_vel=truncate_quantile_vel,
            steer_density=steer_density,
            inst_weights=inst_weights,
            external=no_steer or set()
            )
        return self.sample(
            EventConstraints(**{
                k:v for k,v in cons_kw.items() if v is not None
                }), # type: ignore
            'vtip'
        ).as_dict()
    
    def query_vipt(self,
        note_on_map:Dict[int,List[int]]|None=None, 
        note_off_map:Dict[int,List[int]]|None=None,
        min_time:float|None=None, max_time:float|None=None,
        min_vel:float|None=None, max_vel:float|None=None,
        min_polyphony:Dict[int,int]|int|None=None, 
        max_polyphony:Dict[int,int]|int|None=None,
        min_duration:Dict[int,float]|float|None=None, 
        max_duration:Dict[int,float]|float|None=None, 
        pitch_temp:float|None=None,
        rhythm_temp:float=None, timing_temp:float=None,
        truncate_quantile_time:Tuple[float,float]|None=None,
        truncate_quantile_pitch:Tuple[float,float]|None=None,
        truncate_quantile_vel:Tuple[float,float]|None=None,
        steer_density:float=None,
        inst_weights:Dict[int,float]=None,
        no_steer:List[int]|set[int]=None,
        ):
        """
        Query in a fixed velocity->instrument->pitch->time order, sampling all
        modalities. Because velocity is sampled first, this query method can 
        automatically prevent double noteOn or NoteOff. It's also possible to
        make some more detailed constraints per-instrument compared to `query`,
        including note duration constraints which can eliminate stuck notes.

        query_vtip is similar, but makes different compromises in applying 
        constraints. VTIP is likely to be better when setting min_time > 0 
        or otherwise heavily constraing time delta, while VIPT may be better
        in other cases.

        Args:
            note_on_map: possible note-ons as {instrument: [pitch]} 
                defaults to allowing any note. Notes already playing on a given
                instrument are always excluded.
            note_off_map: possible note-offs as {instrument: [pitch]}
                defaults to using only the instruments in note_on_map. 
                Notes not already playing on a given instrument are 
                automatically excluded.
            min_time: global minimum interevent time (default 0)
            max_time: global maximum interevent time (default no limit)
            min_vel: global minimum velocity for NoteOn events (default 1)
            max_vel: global maximum velocity for NoteOn events (default 127)
            min_polyphony: minimum number of concurrent notes per instrument.
                (default 0). Can be a dict mapping instrument to value,
                or a single value for all instruments.
                When an instrument has <= min polyphony, exclude NoteOffs
            max_polyphony: minimum number of concurrent notes per instrument.
                (default no limit). Can be a dict mapping instrument to value,
                or a single value for all instruments.
                When an instrument has >= max polyphony, exclude NoteOns.
            min_duration: minimum note length per instrument (default 0). Can   
                be a dict mapping instrument to value, or a single value for 
                all instruments.
            max_duration: maximum note length per instrument (default 0). Can   
                be a dict mapping instrument to value, or a single value for 
                all instruments.
            pitch_temp: if not None, apply top_p sampling to the pitch.
            rhythm_temp: if not None, apply top_p sampling to the weighting
                of mixture components. this affects coarse rhythmic patterns;
                0 is deterministic, 1 is 'natural' according to the model.
            timing_temp: if not None, apply temperature sampling to the time
                component. this affects fine timing; 0 is deterministic and 
                precise, 1 is 'natural' according to the model.
            truncate_quantile_time: applied after min_time, max_time
                truncate the remaining delta time distribution by quantile.
                e.g. truncate_quantile_time=(0.25, 0.75)
                excludes the shortest 25% and longest 25% of interevent times.
            truncate_quantile_pitch: truncate the pitch distribution by 
                quantile. e.g. truncate_quantile_pitch=(0.5, 1) always samples
                above the median predicted pitch. Ignored for drums.
            truncate_quantile_vel: truncate the velocity distribution by 
                quantile. e.g. truncate_quantile_vel=(0, 0.5) always
                samples below the median predicted velocity. Affects only NoteOn.
            steer_density: adjust relative weight of NoteOn and NoteOff.
                values above 0.5 favor NoteOn, values below 0.5 favor NoteOff.
            inst_weights: multiplicatively adjust instrument probabilities. 
                Any instrument not included has a weight of 1. 0 would exclude
                an instrument completely (but better to do so via note_on_map)
            no_steer: collection of instruments to exclude from effect of 
                truncate_quantile_pitch and truncate_quantile_time.
        """
        cons_kw = dict(
            note_on_map=note_on_map,
            note_off_map=note_off_map,
            min_time=min_time,
            max_time=max_time,
            min_vel=min_vel,
            max_vel=max_vel,
            min_polyphony=min_polyphony,
            max_polyphony=max_polyphony,
            min_duration=min_duration,
            max_duration=max_duration,
            pitch_temp=pitch_temp,
            rhythm_temp=rhythm_temp,
            timing_temp=timing_temp,
            truncate_quantile_time=truncate_quantile_time,
            truncate_quantile_pitch=truncate_quantile_pitch,
            truncate_quantile_vel=truncate_quantile_vel,
            steer_density=steer_density,
            inst_weights=inst_weights,
            external=no_steer)
        return self.sample(
            EventConstraints(**{
                k:v for k,v in cons_kw.items() if v is not None
                }), # type: ignore
            'vipt'
        ).as_dict()

    # TODO: remove pitch_topk and sweep_time?
    # TODO: rewrite this to call sample?
    def query(self,
            next_inst:int=None, next_pitch:int=None, 
            next_time:float=None, next_vel:int=None,

            allow_end:bool=False,

            include_inst:set[int]=None, exclude_inst:set[int]=None,
            allow_anon:bool=True, 
            instrument_temp:float=None, 

            include_pitch:set[int]=None, exclude_pitch:set[int]=None,
            include_drum:set[int]=None,
            truncate_quantile_pitch:Tuple[float,float]=None,
            pitch_temp:float=None, 
            index_pitch:int=None,

            min_time:float=None, max_time:float=None,
            truncate_quantile_time:tuple[float, float]=None,
            rhythm_temp:float=None, timing_temp:float=None,

            min_vel:int=None, max_vel:int=None,
            velocity_temp:float=None,

            pitch_topk:int=None, sweep_time:bool=False, 

            handle:str=None, return_params:bool=False
            ) -> dict:
        """
        Sample a prediction for the next MIDI event.

        various constraints on the the next event can be requested.

        Args:
            # hard constraints

            next_inst: fix a particular instrument for the predicted event.
                sampled values will always condition on fixed values, so passing
                `next_inst=1`, for example, will make the event appropriate
                for the Grand Piano (instrument 1) to play.
            next_pitch: fix a particular MIDI number for the predicted event.
                sampled values will always condition on fixed values, so passing
                `next_pitch=60`, for example, will make the event a middle C
                (for melodic instruments) or High Bongo (for drums)
            next_time: fix a particular delta time for the predicted event.
                sampled values will always condition on fixed values, so passing
                `next_time=0`, for example, will make the event concurrent with
                the previous event.
            next_vel: fix a particular velocity for the predicted event.
                sampled values will always condition on fixed values, so passing
                `next_inst=0`, for example, will ensure the event is a noteOff.
                
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
                excludes the lowest and highest 25% of pitches
            index_pitch: if not None, deterministically take the
                nth most likely pitch instead of sampling.

            timing_temp: if not None, apply temperature sampling to the time
                component. this affects fine timing; 0 is deterministic and 
                precise, 1 is 'natural' according to the model.
            rhythm_temp: if not None, apply top_p sampling to the weighting
                of mixture components. this affects coarse rhythmic patterns;
                0 is deterministic, 1 is 'natural' according to the model.
            truncate_quantile_time: applied after min_time, max_time
                truncate the remaining delta time distribution by quantile.
                e.g. truncate_quantile_time=(0.25, 0.75)
                excludes the shortest 25% and longest 25% of interevent times.

            velocity_temp: if not None, apply temperature sampling to the 
                velocity component.

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
        ) - exclude_inst) #type: ignore
        if len(constrain_inst)==0:
            raise ValueError(f"""
            every instrument has been excluded. 
            {include_inst=} 
            {exclude_inst=}
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
            iperm = argsort(perm) # inverse permutation back to canonical order

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
            
            running_ctx = sum(context, start=torch.tensor(0.))
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
                end = torch.zeros(self.h.shape[:-1]) #type:ignore

            if sweep_time or pitch_topk:
                # return lists of predictions
                pred_inst = [x.item() for x in pred_inst] #type:ignore
                pred_pitch = [x.item() for x in pred_pitch] #type:ignore
                pred_time = [x.item() for x in pred_time] #type:ignore
                pred_vel = [x.item() for x in pred_vel] #type:ignore
                end = [x.item() for x in end]
                # print(pred_time, pred_pitch, pred_vel)
            else:
                # return single predictions
                pred_inst = pred_inst.item() #type:ignore
                pred_pitch = pred_pitch.item() #type:ignore
                pred_time = pred_time.item() #type:ignore
                pred_vel = pred_vel.item() #type:ignore
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

    def feed_query(self, inst:int, pitch:int, time:float, vel:float, 
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
            inst:int, pitch:int, time:float, vel:float, 
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
        self.h_query = None
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
        checkpoint = torch.load(
            path, map_location=torch.device('cpu'), weights_only=False)
        model = cls(**checkpoint['kw']['model'])
        model.load_state_dict(checkpoint['model_state'], strict=False)
        model.checkpoint_path = path
        model.reset()
        model.eval()
        return model
    
    def prompt(self, midi_file, merge=False):
        """Read a MIDI file and feed events to this Notochord model.

        When possible, the hidden states will be cached so re-using the same prompt will be fast.

        Args:
            midi_file: path of a midi file to read
        Returns:
            state: hidden state dict of the Notochord encoding the MIDI prompt
            channel_inst: dict mapping MIDI channel (0-index) to Notochord instrument (1-256)
        """
        return prompt(
            self, Path(midi_file), 
            merge=merge, state_hash=hash_states(self.get_state()))
    
def hash_states(s):
    if isinstance(s, dict):
        return {k:hash_states(v) for k,v in s.items()}
    elif isinstance(s, torch.Tensor):
        return hash_tensor(s)
    return s

def hash_tensor(t):
    return hashlib.md5(json.dumps(t.tolist()).encode('utf-8')).digest()

# TODO:
# edge case: anon insts may run out if channels change program enough
class InstrumentData:
    """
    channels: set of MIDI channels this instrument appeared on (0-indexed)
    pitches: set of MIDI pitches this instrument played
    velocities: set of MIDI velocities this instrument played
    orig_inst: the Notochord instrument corresponding to the MIDI program this
        instrument was derived from -- i.e. if this is an anonymous instrument
        because it collided with the same instrument on another channel,
        this annotates that original instrument
    notes: count of noteOn events after processing
    shortened: number of notes shorted to avoid pitch collisions
    dropped: number of notes dropped to avoid pitch collisions
    """
    def __init__(self):
        self.channels = set()
        self.pitches = set()
        self.velocities = set()
        self.orig_inst:int|None = None
        self.notes = 0
        self.shortened = 0
        self.dropped = 0
    def __repr__(self):
        pr = f' ({min(self.pitches)}-{max(self.pitches)})' if self.notes else ''
        return f'(notes={self.notes} shortened={self.shortened} dropped={self.dropped} orig_inst={self.orig_inst} channels={self.channels} pitches={len(self.pitches)}{pr})'
@mem.cache(ignore=('noto',))
def prompt(
    noto:Notochord, midi_file:str|Path, 
    merge:bool=False, insert_threshold:float=0.05, 
    jitter:float=1e-4,
    state_hash:int|None=None): 
    # state_hash is used for disk cache only
    """Read a MIDI file and feed events to a Notochord model.

    Args:
        noto: a Notochord
        midi_file: path of a midi file to read
        merge: whether to merge parts on the same instrument or use anonymous
            instrument IDs
        insert_threshold: minimum note duration, in seconds, to insert a
            noteOff rather than drop a noteOn when handling overlap
        state_hash: representation of model hidden state to use for caching results
    Returns:
        state: hidden state dict of the Notochord encoding the MIDI prompt
        inst_data: dict mapping Notochord instrument to metadata (see `InstrumentData` class)
    """
    # TODO: deduplicate this code?
    class AnonTracks:
        def __init__(self):
            self.n = 0
            self.n_drum = 0
        def __call__(self, drum=False):
            if drum:
                self.n_drum += 1
                return noto.first_anon_like(129)+self.n_drum
            else:
                self.n += 1
                return noto.first_anon_like(1)+self.n
    next_anon = AnonTracks()
    # track current instrument on each channel
    noto_channel_inst = defaultdict(next_anon)
    # tracks the original instrument when anon is used to disambiguate parts
    # note the 'original' instrument is still anon when there is no PC on channel
    orig_channel_inst = {}#defaultdict(lambda c: noto_channel_inst[c])
    # metadata for each instrument
    inst_data = defaultdict(InstrumentData)

    mid = mido.MidiFile(midi_file)
    ticks_per_beat = mid.ticks_per_beat
    us_per_beat = 500_000
    time_seconds = 0
    prev_time_seconds = 0
    # event_count = defaultdict(int)
    print(f'MIDI file: {ticks_per_beat} ticks, {us_per_beat} μs per beat')

    dropped_notes = set()

    def set_inst(chan, inst):
        orig_inst = inst
        # get anonymous instrument if already in use and not merging
        inst_reused = any(
            inst==i and chan!=c for c,i in noto_channel_inst.items())
        if inst_reused and not merge:
            inst = next_anon(drum=noto.is_drum(inst))

        noto_channel_inst[chan] = inst
        orig_channel_inst[chan] = orig_inst

    def feed_event(inst, pitch, dt, vel):
        # move noteOffs earlier, noteOns later,
        # to resemble training data augmentation better
        if vel==0:
            dt = max(jitter, dt-jitter)
        else:
            dt = dt+jitter
        noto.feed(inst, pitch, dt, vel)

    for msg in tqdm(mid, desc='ingesting MIDI prompt'):
        chan = msg.channel if hasattr(msg, 'channel') else None
        # when iterating over a track this is ticks,
        # when iterating the whole file it's seconds
        time_seconds += msg.time

        if msg.type=='program_change':
            set_inst(chan, msg.program + 1 + 128*int(msg.channel==9))

            # tqdm.write(str(msg))
            tqdm.write(f'MIDI file: set program {msg.program} (channel {chan}) at {time_seconds} seconds')
            
        elif msg.type=='set_tempo':
            us_per_beat = msg.tempo
            tqdm.write(f'MIDI file: set tempo {us_per_beat} μs/beat at {time_seconds} seconds')
            
        elif msg.type in ('note_on', 'note_off'):
            # make channel 10 with no PC anonymous drumkit
            if chan not in noto_channel_inst and chan==9:
                set_inst(chan, noto.first_anon_like(129))

            inst = noto_channel_inst[chan]
            orig_inst = (
                orig_channel_inst[chan] if chan in orig_channel_inst 
                else noto_channel_inst[chan])
            pitch = msg.note
            dt = time_seconds - prev_time_seconds
            vel = msg.velocity if msg.type=='note_on' else 0
            # event_count[(chan, mid_channel_inst[chan])] += 1

            d = inst_data[inst]
            d.orig_inst = orig_inst
            if vel > 0:
                # handle collision:
                # if the ongoing note is past a given length,
                # insert a noteOff
                # otherwise drop the second one
                if (inst, pitch) in noto.held_notes:
                    # if inst==50: tqdm.write(f'collision: {(inst, pitch)=} {time_seconds=}')
                    dur = noto.held_notes[(inst,pitch)] + dt
                    dropped_notes.add((chan, pitch))
                    if dur > insert_threshold:
                        # if inst==50: tqdm.write(f'insert {dur=}')
                        feed_event(inst, pitch, dt, 0)
                        d.shortened += 1
                        dt = 0
                    else:
                        # if inst==50: tqdm.write(f'drop {dur=}')
                        d.dropped += 1
                        continue
                d.notes += 1
                d.pitches.add(pitch)
                d.velocities.add(vel)
            else:
                # use first noteOff (likely from a different channel)
                # if (inst, pitch) not in noto.held_notes:
                    # continue
                # use corresponding noteOff
                if (chan, pitch) in dropped_notes:
                    # if inst==50: tqdm.write(f'dropped: {(chan, pitch)=} {time_seconds=}')
                    dropped_notes.remove((chan, pitch))
                    continue

            # if inst==50: tqdm.write(f'event: {(chan, inst, pitch, vel)=} {time_seconds=}')

            d.channels.add(chan)

            # event_count[mid_channel_inst[msg.channel]] += 1
            feed_event(inst, pitch, dt, vel)
            prev_time_seconds = time_seconds

        else: continue

    inst_data = {k:v for k,v in inst_data.items() if v.notes>0}
    print('MIDI file:', inst_data)
 
    return noto.get_state(), inst_data
