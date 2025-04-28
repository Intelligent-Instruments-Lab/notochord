"""
Notochord MIDI co-improviser server.
Notochord plays different instruments along with the player.
"""

# Authors:
#   Victor Shepardson
#   Intelligent Instruments Lab 2025

tui_doc = """Welcome to notochord homunculus.
You have 16 channels of MIDI, laid out in two rows of eight.
Each can be in one of three modes:
    input (-->02)
    follow (01->02)
    auto (02)
Input channels take MIDI input from the corresponding channel,
and send it to notochord. When using --thru, input is also copied
to the output.
    Follow channels act like a harmonizer; they query notochord for
a NoteOn whenever the followed channel has a NoteOn, and have the
corresponding NoteOff when the followed channel does.
    Auto channels are played autonomously by notochord.
Each channel can be muted independently of the others.
    Below the channel strips, there is a row of presets.
These can be edited in homunculus.toml; run `notochord files` to find 
it in your file explorer."""

# TODO: move soundfont / general MIDI stuff out of script

# TODO: make key bindings visibly click corresponding buttons
# TODO: make Mute a toggle but Reset a momentary

# TODO: color note by pitch class + register
# TODO: color instrument 1-128, MEL, DRUM, ANON, ANONDRUM
# TODO: color time
# TODO: id prediction as player / noto
# TODO: unify note log / prediction format
# TODO: grey out predictions when player or muted notochord

# TODO: controls display panel
# TODO: held notes display panel
# TODO: counts / weights display panel
# TODO: MIDI learn

import time
import random
import traceback
import shutil
from typing import Optional, Dict, Any
from numbers import Number
import math
import functools as ft
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import toml_file
import mido
from tqdm import tqdm

import torch
torch.set_num_threads(1)

import iipyper, notochord
from notochord import Notochord, NotoPerformance, NoPossibleEvents
from notochord.util import deep_update
from iipyper import OSC, MIDI, run, Stopwatch, repeat, cleanup, TUI, profile, lock

from rich.panel import Panel
from rich.pretty import Pretty
from textual.reactive import reactive
from textual.widgets import Header, Footer, Static, Button, Log, RichLog, Label
from textual.screen import Screen, ModalScreen
from textual.containers import Grid

def now():
    return time.perf_counter()

def main(
    checkpoint="notochord-latest.ckpt", # Notochord checkpoint
    config:Dict[int,Dict[str,Any]]=None, # map MIDI channel : GM instrument
    preset:str=None,
    preset_file:Path=None,

    midi_prompt:Path=None,
    prompt_config:bool=True, # set channel config based on prompt
    prompt_channel_order:str=None,
    prompt_merge:bool=False,

    seed:int=0,

    initial_mute=False, # start with Notochord muted
    initial_query=None, # DEPRECATED, now inverse of `initial_stop`
    initial_stop=False, # if False, auto voices play immediately

    midi_in:Optional[str]=None, # MIDI port for player input
    midi_out:Optional[str]=None, # MIDI port for Notochord output
    midi_control:Optional[str]=None, # MIDI port for note messages as controls
    thru=False, # copy player input to output
    send_pc=False, # send program change messages
    dump_midi=False, # print all incoming MIDI
    suppress_midi_feedback=True,
    input_channel_map=None,

    stop_on_end=False, # auto channels stop when end is sampled
    reset_on_end=False, # reset notochord when end is sampled
    end_exponent=1, # < 1 makes end more likely, >1 less likely
    min_end_time=8, # prevent sampling end before this many seconds since reset

    balance_sample=False, # choose instruments which have played less recently
    n_recent=32, # number of recent note-on events to consider for above
    n_margin=8, # amount of 'slack' in the balance_sample calculation
    
    max_note_len=11, # in seconds, to auto-release stuck Notochord notes
    max_time=None, # max time between events
    nominal_time=True, #DEPRECATED

    min_vel=None, #mininum velocity (besides noteOffs)
    max_vel=None, # maximum velocity

    osc_port=None, # if supplied, listen for OSC to set controls on this port
    osc_host='', # leave this as empty string to get all traffic on the port

    punch_in=False, # EXPERIMENTAL. this causes all channels to switch between auto and input mode when input is received or not
    punch_out_after=1.0, # time in seconds from last input to revert to auto

    use_tui=True, # run textual UI
    predict_input=True, # forecasted next events can be for input (preserves model distribution, but can lead to Notochord deciding not to play)
    predict_follow=False,
    debug_query=False, # don't query notochord when there is no pending event.
    testing=False,
    estimated_query_latency=10e-3,
    estimated_feed_latency=10e-3,
    lateness_margin=100e-3, # when events are playing later than this, slow down
    soft_lateness_margin=50e-3, # when events are playing later than this, slow down
    soundfont=None,
    limit_input=None,
    # thru_vel_offset=None,
    profiler=0,
    wipe_presets=False,
    ):
    """
    This a terminal app for using Notochord interactively with MIDI controllers and synthesizers. It allows combining both 'harmonizer' and 'improviser' features.
     
    Arguments to main can be given on the command line as flags, for example:

    `notochord homunculus --config "{1:{mode:auto, inst:1, mute:False}}" --initial-query --max-time 1`

    This says: play the grand piano autonomously on channel 1, start automatically, allow no more than 1 second between MIDI events. A different example:

    `notochord homunculus --config "{1:{mode:input, inst:1, mute:False}, 2:{mode:follow, inst:12, mute:False}}" --thru --send-pc`

    This says, take grand piano input on channel 1, and harmonize it with vibraphone on channel 2. Pass input through to the output, and also send program change messages to set the instruments on the synthesizer.

    You may also need to use the --midi-in or --midi-out flags to get MIDI to the right place.
    
    # MIDI Channels

    In homunculus, you have 16 MIDI channels. Each channel can be in one of three modes:

        * input (appearing like "-->01"), channel 1 comes from MIDI input channel 1.
        * follow (appearing like "01->02"), channel 2 plays whenever channel 1 plays.
        * auto (appearing like just "03"), channel 3 plays autonomously.

    Click the top section of each channel strip to cycle the mode.

    Each channel is also assigned a [General MIDI instrument](https://en.wikipedia.org/wiki/General_MIDI#Program_change_events). Each 'input' and 'auto' channel should have a unique General MIDI instrument, but 'follow' channels can be duplicates of others. If you try to assign duplicate instruments, they will automatically change to "anonymous" melodic or drum instruments, but still send the right program change messages when using --send-pc.

    Click the middle section of each channel strip to choose a new instrument.

    The bottom section of each channel strip allows muting individual voices.

    # Global controls

    Along the bottom, there are global query, sustain, mute and reset buttons. Query manually replaces the next pending note. Sustain stops all auto voices from playing without ending any open notes. Mute ends all open notes and stops auto voices. Reset ends all open notes, forgets all context and sets the Notochord model to its initial state.

    Args:
        checkpoint: path to notochord model checkpoint.

        config: 
            mapping from MIDI channels to voice specs.
            MIDI channels and General MIDI instruments are indexed from 1.
            see wikipedia.org/wiki/General_MIDI for values.
            There are 3 modes of voice, 'auto', 'follow' and 'input'. For example,
            ```
            {
                1:{
                    'mode':'input', 'inst':1
                }, # input grand piano on MIDI channel 1
                2:{
                    'mode':'follow', 'source':1, 'inst':1, 
                    'transpose':(-12,12)
                }, # harmonize the channel within 1 octave 1 with more piano
                3:{
                    'mode':'auto', 'inst':12, 'range':(36,72)
                }, # autonomous vibraphone voice in the MIDI pitch 36-72 range
                10:{
                    'mode':'auto', 'inst':129,
                }, # autonomous drums voice
                4:{
                    'mode':'follow', 'source':3, 'inst':10, 'range':(72,96)
                }, # harmonize channel 3 within upper registers of the glockenspiel
            }
            ```
            Notes:
            when two channels use the same instrument, one will be converted
            to an 'anonymous' instrument number (but it will still respect 
            the chosen instrument when using --send-pc)

        preset: 
            preset name (in preset file) to load config from
        preset_file: 
            path to a TOML file containing channel presets
            the default config file is `homunculus.toml`; 
            running `notochord files` will show its location.

        midi_prompt:
            path to a MIDI file to read in as a prompt.
            note that prompts can alternatively be associated with presets 
            in `homunculus.toml`
        prompt_config:
            if True, set unmuted instruments to those in the prompt file.
            prompt config overrides presets,
            but config from the `--condfig flag` will override the prompt
        prompt_channel_order:
            method to re-order the channels in a MIDI prompt
            'channel' (default, leave as they are in file)
            'instrument': sort by instrument ID 
                (keeping associated anonymous IDs together)
            'instrument_drums': sort by instrument ID,
                but with drums starting on channel 10
            'notes': sort by most notes
        prompt_merge:
            if True, merge multiple channels with the same GM program into one
            Notochord voice (with overlapping notes dropped or shortened)
            if False, use anonymous instruments.

        seed:
            global random seed (added to preset random seed)

        initial_mute: 
            start 'auto' voices muted so it won't play with input.
        initial_stop: 
            if True, begin in a stopped state, 
            so 'auto' voices don't begin playing without input.

        midi_in: 
            MIDI ports for input. 
            default is to use all input ports.
            can be comma-separated list of ports.
        midi_out: 
            MIDI ports for output. 
            default is to use only virtual 'From iipyper' port.
            can be comma-separated list of ports.
        thru: 
            if True, copy input MIDI to output ports.
            only makes sense if input and output ports are different.
        send_pc: 
            if True, send MIDI program change messages to set the General MIDI
            instrument on each channel according to player_config and noto_config.
            useful when using a General MIDI synthesizer like fluidsynth.
        dump_midi: 
            if True, print all incoming MIDI for debugging purposes
        suppress_midi_feedback:
            attempt to allow use of the one loopback port for both input and
            output, by ignoring any MIDI input which is identical to an output
            within a few milliseconds.
            
        balance_sample:
            choose 'auto' voices which have played less recently,
            ensures that all configured instruments will play.
        n_recent: 
            number of recent note-on events to consider for above
        n_margin: 
            controls the amount of slack in the balance_sample calculation

        max_note_len: 
            time in seconds after which to force-release sustained 'auto' notes.
        max_time: 
            maximum seconds between predicted events for 'auto' voices.
            default is the Notochord model's maximum (usually 10 seconds).
        lateness_margin:
            when events are playing later than this (in seconds), slow down

        min_vel: mininum velocity (except for noteOffs where vel=0)
        max_vel: maximum velocity

        osc_port: 
            optional. if supplied, listen for OSC to set controls
        osc_host: 
            hostname or IP of OSC sender.
            leave this as empty string to get all traffic on the port

        punch_in: EXPERIMENTAL. this causes all channels to switch between
            auto and input mode when input is received or not
        punch_out_after: time in seconds from last input NoteOn to revert to 
            auto mode

        use_tui: 
            run textual UI.
        predict_input: 
            forecasted next events can be for 'input' voices.
            generally should be True for manual input;
            use balance_sample to force 'auto' voices to play. 
            you might want it False if you have a very busy input.
        predict_follow:
            ditto for 'follow' voices. less obvious what the correct setting
            is, but False will be more efficient

        wipe_presets:
            if True, replaces your homunulus.toml with the defaults
            (may be useful after updating notochord)

        soundfont: path to a soundfont file from which default instrument 
            ranges will be loaded

        wipe_presets: CAUTION. replaces your homunculus.toml file with the 
            default. may fix errors after upgrading. backup your config file
            if you've been changing it! (`notochord files` to find it)
    """
    if osc_port is not None:
        osc = OSC(osc_host, osc_port)
    if midi_in is None and midi_control is not None:
        midi_in = set(mido.get_input_names()) - set(midi_control.split(','))
    midi = MIDI(midi_in, midi_out, suppress_feedback=suppress_midi_feedback)
    if midi_control is not None:
        midi_control = MIDI(
            midi_control, midi_control, 
            virtual_in_ports=0, virtual_out_ports=0)

    # backwards compat
    if initial_query is not None:
        initial_stop = not initial_query

    estimated_latency = estimated_feed_latency + estimated_query_latency

    if input_channel_map is None: input_channel_map = {}

    ### Textual UI
    tui = NotoTUI()
    print = notochord.print = iipyper.print = tui.print
    ###

    if soundfont is None:
        sf_inst_ranges = {}
    else:
        # attempt to get instrument ranges from the soundfont
        # assumes first bank is used
        # not sure if entirely correct
        from sf2utils.sf2parse import Sf2File
        from sf2utils.generator import Sf2Gen

        with open(soundfont, 'rb') as file:
            soundfont = Sf2File(file)
        sf_presets = {
            p.preset:p
            for p in soundfont.presets 
            if hasattr(p,'bank') and p.bank==0}
        sf_drum_presets = {
            p.preset:p
            for p in soundfont.presets 
            if hasattr(p,'bank') and p.bank==128}

        def _get_range(i):
            if i>128:
                if i-128 in sf_drum_presets:
                    p = sf_drum_presets[i-128]
                else:
                    p = sf_drum_presets[0]
            else:
                if i-1 in sf_presets:
                    p = sf_presets[i-1]
                else:
                    p = sf_presets[0]
            # union of bags in preset
            # bag: intersection of range with union of bags in instrument
            preset_range = set()
            for b in p.bags:
                bag_range = set()
                if Sf2Gen.OPER_INSTRUMENT in b.gens: 
                    inst = soundfont.instruments[b[Sf2Gen.OPER_INSTRUMENT].amount]
                    for bi in inst.bags:
                        if Sf2Gen.OPER_SAMPLE_ID in bi.gens:
                            if bi.key_range is not None:
                                l,h = bi.key_range
                                bag_range |= set(range(l, h+1))
                if b.key_range is not None: 
                    # print(b.key_range)
                    l,h = b.key_range
                    bag_range &= set(range(l, h+1))
                preset_range |= bag_range
            # print(f'{preset_range=}')
            if len(preset_range):
                return min(preset_range), max(preset_range)
            return 0, 127
        sf_inst_ranges = {i:_get_range(i) for i in range(1,257)}
    def get_range(i):
        return sf_inst_ranges.get(i, (0,127))

    # print(f'{sf_inst_ranges=}')

    if not nominal_time:
        print('nominal_time is deprecated; nominal_time=False now sets lateness_margin to 0')
        lateness_margin = 0
    
    # make preset file if it doesn't exist
    cfg_dir = Notochord.user_data_dir()
    default_preset_file = cfg_dir / 'homunculus.toml'
    src_preset_file = Path(__file__).parent / 'homunculus.toml'
    if wipe_presets or not default_preset_file.exists():
        shutil.copy(src_preset_file, default_preset_file)

    ### presets and config
    try:    
        if preset_file is None:
            global_config = toml_file.Config(str(default_preset_file))
        else:
            global_config = toml_file.Config(str(preset_file))
        presets = global_config.get('preset', {})
        control_meta = global_config.get('control', {})
        action_meta = global_config.get('action', {})

    except Exception:
        print('WARNING: failed to load presets from file')
        print(traceback.print_exc(file=tui))
        global_config = {}
        presets = {}
        control_meta = []
        action_meta = []

    # print(f'{global_config=}')
    # print(f'{presets=}')

    # store control values
    controls = {ctrl['name']:ctrl.get('value', None) for ctrl in control_meta}

    # load notochord model
    try:
        noto = Notochord.from_checkpoint(checkpoint)
        noto.eval()
        noto.feed(0,0,0,0) # smoke test
        noto.query()
        noto.reset()
    except Exception:
        print("""error loading notochord model""")
        raise

    # defaults
    def default_config_channel(i):
        """default values for presets
        all fields should appear here, even if default is None
        """
        d = global_config.get('default', {})
        # toml_file's get method also sets the value for some reason??
        # and it refuses to set a None value...
        def get(k, default):
            return d[k] if k in d else default
        return {
            'mode': get('mode', 'auto'), 
            'inst': noto.first_anon_like(1),
            'mute': get('mute', True), 
            'source': max(1,i-1), 
            'note_shift': get('note_shift', 0),
            'duration': get('duration', None),
            'poly': get('poly', None),
            'range': get('range', None),
            'transpose': get('transpose', None),
            'cc': get('cc', toml_file.Config())
            }
    
    # convert MIDI channels to int
    # print(presets)
    for p in presets:
        if 'channel' not in p:
            p['channel'] = {}
        else:
            p['channel'] = {int(k):v for k,v in p['channel'].items()}   

    ### this feeds all events from the prompt file to notochord
    def do_prompt(prompt_file, channel_order=None, merge_channels=False, seed=None):
        prompt_random = random.Random(seed)
        prompt_file = Path(prompt_file).expanduser()
        if prompt_file.is_dir():
            prompt_file = prompt_random.choice([
                p for p in 
                prompt_file.glob('**/*')
                if p.is_file()
                ])
        print(f'{prompt_file=}')
        noto.reset()
        initial_state, inst_data = noto.prompt(
            prompt_file, merge=merge_channels)
        # first get the 16 most active parts
        insts = sorted(inst_data, key=lambda x: -inst_data[x].notes)[:16]
        print(f'{inst_data=}')
        # then order them onto channels
        def make_cfg(i):
            return {
                'inst':i, 'mute':False, 
                'program_inst':inst_data[i].orig_inst}
        if channel_order=='instrument':
            # sort by (original_inst, noto_inst)
            config = {
                c+1:make_cfg(i)
                for c,i in enumerate(sorted(
                    insts, key=lambda k: (inst_data[k].orig_inst, k)))}
        elif channel_order=='instrument_drums':
            insts = sorted(insts, key=lambda k: (inst_data[k].orig_inst, k))
            drum_insts = [i for i in insts if noto.is_drum(i)]
            mel_insts = [i for i in insts if i not in drum_insts]
            config = {c+10:make_cfg(i) for c,i in enumerate(drum_insts)}
            cs = sorted(set(range(1,17)) - set(config))
            config.update({c:make_cfg(i) for c,i in zip(cs, mel_insts)})
        elif channel_order=='main_keyboard_first':
            drum_insts = [i for i in insts if noto.is_drum(i)]
            kb_insts = []
            nonkb_insts = []
            for i in insts:
                if i in drum_insts: continue
                l,h = get_range(inst_data[i].orig_inst)
                if l<=21 and h>=108:
                    kb_insts.append(i)
                else:
                    nonkb_insts.append(i)
            mel_insts = kb_insts + nonkb_insts
            config = {c+10:make_cfg(i) for c,i in enumerate(drum_insts)}
            cs = sorted(set(range(1,17)) - set(config))
            config.update({c:make_cfg(i) for c,i in zip(cs, mel_insts)})
        elif channel_order=='notes':
            config = {c+1:make_cfg(i) for c,i in enumerate(insts)}
        elif channel_order=='channel' or channel_order is None:
            # NOTE this takes the minimum channel if an instrument
            # appears on multiple channels
            # the instrument with most notes will take precedence
            # if multiple instruments appear
            # but could be better to avoid collisions in some cases...
            config = {
                min(inst_data[i].channels)+1:make_cfg(i) 
                for i in reversed(insts)}
        else:
            raise ValueError(f"""
                unknown {channel_order=} 
                (options are 'instrument', 'notes', or 'channel')""")
        return initial_state, config
            
    global_initial_state = None
    config_ingest = None
    if midi_prompt is not None:
        global_initial_state, config_ingest = do_prompt(
            midi_prompt, 
            merge_channels=prompt_merge, channel_order=prompt_channel_order)
        if not prompt_config:
            config_ingest = None
    initial_state = global_initial_state

    # process prompts in each preset
    for p in presets:
        if 'prompt' in p:
            print(f'prompting "{p["prompt"]}" for preset "{p["name"]}"')
            prompt_seed = p.get('seed')
            if seed is None and p.get('seed') is None:
                prompt_seed = None
            else:
                prompt_seed = (seed or 0) + p.get('seed', 0)
            p['initial_state'], prompt_cfg = do_prompt(
                p['prompt'], 
                merge_channels=p.get('prompt_merge_channels'),
                channel_order=p.get('prompt_channel_order'),
                seed=prompt_seed
                )
            # print(f'{prompt_cfg=}')
            preset_cfg = p['channel']
            for k,chan_cfg in prompt_cfg.items():
                chan_cfg.update(preset_cfg.get(k, {}))
                preset_cfg[k] = chan_cfg
            # print(f"preset channel config: {preset_cfg}")

    # def validate_config():
    #     assert all(
    #         v['source'] in config for v in config.values() if v['mode']=='follow'
    #         ), 'ERROR: no source given for follow voice'
    #     # TODO: check for follow cycles
    # validate_config()

    def mode_insts(t, allow_muted=True):
        if isinstance(t, str):
            t = t,
        # set of instruments with given mode(s)
        return {
            v['inst'] for v in config.values() 
            if v['mode'] in t and (allow_muted or not v['mute'])
            }
    def mode_chans(t):
        if isinstance(t, str):
            t = t,
        # list of channels with given mode
        return [k for k,v in config.items() if v['mode'] in t]
    def channel_inst(c):
        return config[c]['inst']
    # def channel_insts():
        # list of channel,instrument pairs
        # return [(c,channel_inst(c)) for c in config]
    def inst_ranges(insts):
        # instruments to sets of allowed MIDI numbers
        r = {}
        for v in config.values():
            i = v['inst']
            if i in insts:
                s = set(range(*(v.get('range') or get_range(i))))
                if i in r:
                    r[i] |= s
                else:
                    r[i] = s
        return r
    def auto_inst_channel(i):
        for k,v in config.items():
            if v['mode']=='auto' and v['inst']==i:
                return k
        return None
    def channel_followers(chan):
        # print(f'channel_followers {chan=} {config=}')
        # return channel of all 'follow' voices with given source
        return [
            k for k,v in config.items() 
            if v['mode']=='follow' 
            and v.get('source', None)==chan]

    def get_free_anon_like(i):
        s = set(noto.anon_like(i)) - {d['inst'] for d in config.values()}
        return next(iter(s))

    def dedup_inst(c, i):
        if any(
            c_other!=c
            and d['inst']==i 
            and d['mode']!='follow'
            for c_other, d in config.items()
            ):
            i = get_free_anon_like(i)
        return i

    def do_send_pc(c, i):
        # warn_inst(i)
        # assuming fluidsynth -o synth.midi-bank-select=mma
        if noto.is_drum(i):
            midi.control_change(channel=c-1, control=0, value=1)
        else:
            midi.control_change(channel=c-1, control=32, value=0)
            midi.control_change(channel=c-1, control=0, value=0)
        if noto.is_anon(i):
            program = 0
        else:
            # convert to 0-index
            program = (i-1) % 128
        midi.program_change(channel=c-1, program=program)

    # simple class to hold pending event prediction
    class Prediction:
        def __init__(self):
            self.gate = not initial_mute
            self.reset()
        def reset(self):
            self.stopped = False
            self.last_event_time = None
            self.lateness = 0
            self.cum_end_prob = 0
            self.clear()
        def clear(self):
            self.event = None
            self.next_event_time = None
            # tui.defer(prediction=None)
        def occurred(self):
            self.last_event_time = self.next_event_time
            self.clear()
        def set(self, event):
            if event.get('vel',0) > 0:
                self.stopped = False
            if event.get('time') is None:
                event['time'] = self.time_since()
            self.event = event
            self.next_event_time = event['time'] + (
                self.last_event_time or now())
        def time_since(self):
            """current actual time since nominal time of last played event"""
            if self.last_event_time is None:
                return 0
            else:
                return now() - self.last_event_time
        def time_until(self):
            if self.next_event_time is None: 
                return float('inf')
            else:
                return self.next_event_time - now()
        def is_auto(self):
            if self.event is None:
                return False
            return self.event['inst'] in mode_insts('auto')
        def sample_end(self):
            if self.event is None:
                return False
            end_prob = self.event.get('end', 0) ** end_exponent
            # cumulative end probability
            self.cum_end_prob = 1 - (1-self.cum_end_prob)*(1-end_prob)
            # print(f'{self.cum_end_prob=}')
            return random.random() < end_prob
    pending = Prediction()    

    # tracks held notes, recently played instruments, etc
    # NOTE: Notochord now tracks held notes,
    # but NotoPerformance also tracks channels,
    # lets you attach user data to notes,
    # remembers past events, and contains other utilities
    history = NotoPerformance()

    status = {'reset_time':now()}

    action_queue = []

    def display_event(tag, memo, inst, pitch, vel, channel, **kw):
        """print an event to the terminal"""
        if tag is None:
            return
        now = str(datetime.now())[:-2]
        # tstr = "O" if kw["time"]<=1e-3 else ""
        s = f'{now}   inst {inst:3d}   pitch {pitch:3d}   vel {vel:3d}   ch {channel:2d} {tag}'
        # s = f'{tag}:\t{inst=:4d}   {pitch=:4d}   {vel=:4d}   {channel=:3d}'
        if memo is not None:
            s += f' ({memo})'
        tui.defer(note=s)

    # @profile(print=print, enable=profiler)
    def send_midi(note, velocity, channel):
        kind = 'note_on' if velocity > 0 else 'note_off'
        cfg = config[channel]
        # get channel note map
        note_map = cfg.get('note_map', {})
        if note in note_map:
            note = note_map[note]
        elif 'note_shift' in cfg:
            note = note - cfg['note_shift']

        if note < 0:
            print(f'WARNING: dropped note < 0 ({note}, {channel=})')
            return
        if note >= 128:
            print(f'WARNING: dropped note >= 128 ({note}, {channel=})')
            return

        port = cfg.get('port', None)

        midi.send(kind, note=note, velocity=velocity, channel=channel-1, port=port)

    @profile(print=print, enable=profiler)
    def play_event(
            channel, 
            parent=None, # parent note as (channel, inst, pitch)
            feed=True, 
            send=True, 
            tag=None, memo=None):
        """realize an event as MIDI, terminal display, and Notochord update"""
        with profile('rest', print=print, enable=profiler):
            event = pending.event
            if event is None:
                print("WARNING: play_event on null event")
                return
            time_until = pending.time_until()
            pending.lateness = -time_until # set at the time of playing
            # if time_until < 10e-3:
                # print(f'late {-time_until} at {now()=}')
            pending.occurred()
            # normalize values
            vel = event['vel'] = math.ceil(event['vel'])

            # send out as MIDI
            if send:
                send_midi(event['pitch'], vel, channel)

            # print
            display_event(tag, memo=memo, channel=channel, **event)

        if feed:
            # feed to NotoPerformance
            # put a stopwatch in the held_note_data field for tracking note length
            with profile('history.feed', print=print, enable=profiler>1):
                history.feed(held_note_data={
                    'duration':Stopwatch(),
                    'parent':parent
                    }, channel=channel, **event)
            # feed to model
            with profile('noto.feed', print=print, enable=profiler):
                noto.feed(**event)

        follow_event(event, channel, feed=feed)

    @profile(print=print, enable=profiler)
    def follow_event(source_event, source_channel, feed=True):
        source_vel = source_event['vel']
        source_pitch = source_event['pitch']
        source_inst = source_event['inst']
        source_k = (source_channel, source_inst, source_pitch)

        dt = 0

        if source_vel > 0:

            # NoteOn
            for noto_channel in channel_followers(source_channel):
                cfg = config[noto_channel]
                
                if cfg.get('mute', False): continue

                noto_inst = cfg['inst']
                min_x, max_x = cfg.get('transpose') or (-128,128)
                lo, hi = cfg.get('range') or (0,127)

                already_playing = {
                    p for i,p in noto.held_notes if noto_inst==i}
                # already_playing = {
                #     note.pitch for note in history.notes if noto_inst==note.inst}
                # print(f'{already_playing=}')

                pitch_range = range(
                    max(lo,source_pitch+min_x), min(hi, source_pitch+max_x+1))
                pitches = (
                    set(pitch_range) - {source_pitch} - already_playing
                )

                if len(pitches)==0:
                    # edge case: no possible pitch
                    print(f'skipping follow {noto_channel=}, no pitches available')
                    print(f'{pitch_range} minus {source_pitch} minus {already_playing}')
                    print(f'{lo=} {min_x=} {hi=} {max_x=} {source_pitch=}')
                    continue
                elif len(pitches)==1:
                    # edge case: there is exactly one possible pitch
                    pending.set(dict(
                        inst=noto_inst, pitch=list(pitches)[0], 
                        time=dt, vel=source_vel))
                else:
                    # notochord chooses pitch
                    with profile('noto.query', print=print, enable=profiler):
                        pending.set(noto.query(
                            next_inst=noto_inst, next_time=dt, next_vel=source_vel,
                            include_pitch=pitches))
                    
                play_event(
                    noto_channel, feed=feed,
                    parent=source_k, tag='NOTO', memo='follow')
        # NoteOff
        else:
            # print(f'{history.note_data=}')
            dependents = [
                noto_k # chan, inst, pitch
                for noto_k, note_data
                in history.note_data.items()
                if note_data['parent']==source_k
            ]

            for noto_channel, noto_inst, noto_pitch in dependents:
                pending.set(dict(
                    inst=noto_inst, pitch=noto_pitch, time=dt, vel=0))
                play_event(noto_channel, feed=feed, tag='NOTO', memo='follow')

    # @lock
    @profile(print=print, enable=profiler)
    def noto_reset():
        """reset Notochord and end all of its held notes"""
        print('RESET')

        status['reset_time'] = now()

        # cancel pending predictions
        pending.reset()
        
        # end Notochord held notes
        # skip feeding for speed, since notochord gets reset anyway
        end_held(feed=False, memo='reset')
                
        # reset notochord state
        noto.reset(state=initial_state)
        # reset history
        history.push()

        # TODO: feed note-ons from any held input/follower notes?

    # @lock
    @profile(print=print, enable=profiler)
    def noto_mute(sustain=False):
        tui.query_one('#mute').label = 'UNMUTE' if pending.gate else 'MUTE'
        # if sustain:
        tui.query_one('#sustain').label = 'END SUSTAIN' if pending.gate else 'SUSTAIN'

        pending.gate = not pending.gate

        if sustain:
            print('END SUSTAIN' if pending.gate else 'SUSTAIN')
        else:
            print('UNMUTE' if pending.gate else 'MUTE')

        # if unmuting, we're done
        if not pending.gate:
            # cancel pending predictions
            if not sustain:
                end_held(memo='mute')
            pending.clear()

    def noto_stop():
        print('STOP')
        end_held(memo='stop on end')
        pending.stopped = True
        
    @profile(print=print, enable=profiler)    
    def end_held(feed=True, channel=None, memo=None):
        # end+feed all held notes
        channels = mode_chans('auto') if channel is None else [channel]
        ended_any = False
        for note in history.notes:
            if note.chan in channels:
                pending.set(dict(inst=note.inst, pitch=note.pitch, vel=0))
                play_event(
                    channel=note.chan, feed=feed, tag='NOTO', memo=memo)
                ended_any = True
        return ended_any

    # query Notochord for a new next event
    # @lock
    @profile(print=print, enable=profiler)
    def auto_query(
            predict_input=predict_input, 
            predict_follow=predict_follow,
            immediate=False):
        
        # NOTE: replaced this with duration constraints;
        # should test more before deleting
        # check for stuck notes
        # and prioritize ending those
        # for (_, inst, pitch), note_data in history.note_data.items():
        #     dur = note_data['duration'].read()
        #     if (
        #         inst in mode_insts('auto') 
        #         and dur > max_note_len*(.1+controls.get('steer_duration', 1))
        #         ):
        #         # query for the end of a note with flexible timing
        #         # with profile('query', print=print, enable=profiler):
        #         t = pending.time_since()
        #         mt = max(t, min(max_time or np.inf, t+0.2))
        #         pending.set(noto.query(
        #             next_inst=inst, next_pitch=pitch,
        #             next_vel=0, min_time=t, max_time=mt))
        #         print(f'END STUCK NOTE {inst=},{pitch=},{dur=}')
        #         return

        if immediate:
            # sampling immediately after realizing an event
            min_time = 0
        else:
            if pending.event is not None and not pending.is_auto():
                # re-sampling ahead after sampling an input or follow voice
                min_time = pending.event['time']
            else:
                # otherwise sampling after some delay
                min_time = pending.time_since()
            min_time = min_time-estimated_feed_latency
        # print(f'{immediate=} {min_time=}')

        # print(f'{pending.lateness=} {pending.time_since()=}')

        lateness = pending.lateness if immediate else -pending.time_until()
        if lateness > lateness_margin:
            backoff = lateness + estimated_latency
            if backoff > min_time:
                min_time = backoff
                print(f'set {min_time=} due to {lateness=}')

        # if max_time < min_time, exclude input instruments
        input_late = max_time is not None and max_time < min_time

        inst_modes = ['auto']
        if predict_follow:
            inst_modes.append('follow')
        if predict_input and not input_late:
            inst_modes.append('input')
        allowed_insts = mode_insts(inst_modes, allow_muted=False)

        # held_notes = history.held_inst_pitch_map()
        # print(f'{held_notes=}')
        steer_time = 1-controls.get('steer_rate', 0.5)
        steer_pitch = controls.get('steer_pitch', 0.5)
        steer_density = controls.get('steer_density', 0.5)
        steer_velocity = controls.get('steer_velocity', 0.5)

        if lateness > soft_lateness_margin and lateness <= lateness_margin:
            steer_time = 1
            print(f'set {steer_time=} due to {lateness=}')

        rhythm_temp = controls.get('rhythm_temp', 1)
        timing_temp = controls.get('timing_temp', 1)
        
        tqt = (max(0,steer_time-0.5), min(1, steer_time+0.5))
        tqp = (max(0,steer_pitch-0.5), min(1, steer_pitch+0.5))
        tqv = (max(0,steer_velocity-0.5), min(1, steer_velocity+0.5))

        # idea: maintain an 'instrument presence' quantity
        # incorporating time since / number of notes playing
        # ideally this would distinguish sustained from percussive instruments too
        
        # balance_sample: note-ons only from instruments which have played less
        inst_weights = None
        if balance_sample:
            counts = defaultdict(int)
            for i,c in history.inst_counts(n=n_recent).items():
                counts[i] = c
            # print(f'{counts=}')
            inst_weights = {}
            mc = max(counts.values()) if len(counts) else 0
            for i in allowed_insts:
                # don't upweight player controlled instruments though
                if i in mode_insts('auto'):
                    inst_weights[i] = math.exp(max(0, mc - counts[i] - n_margin))
                else:
                    inst_weights[i] = 1.
        # print(f'{inst_weights=}')

        # VTIP is better for time interventions,
        # VIPT is better for instrument interventions
        if min_time > estimated_latency or abs(steer_time-0.5) > abs(steer_pitch-0.5):
            query_method = noto.query_vtip
            # print('VTIP')
        else:
            query_method = noto.query_vipt
            # print('VIPT')
        # query_method = noto.query_vipt ### DEBUG
        # query_method = noto.query_vtip ### DEBUG
        query_method = profile(print=print, enable=profiler)(query_method)

        # print(f'considering {insts} for note_on')
        # use only currently selected instruments
        inst_pitch_map = inst_ranges(allowed_insts)
        note_on_map = {
            i: set(inst_pitch_map[i])#-set(held_notes[i]) # exclude held notes
            for i in allowed_insts#allowed_insts
            if i in inst_pitch_map
        }

        min_polyphony = {}
        max_polyphony = {}
        min_duration = {}
        max_duration = {}
        for i in note_on_map:
            c = auto_inst_channel(i)
            if c is None: continue
            cfg = config[c]
            if cfg is not None and cfg.get('poly') is not None:
                min_polyphony[i], max_polyphony[i] = cfg['poly']
            if cfg is not None and cfg.get('duration') is not None:
                min_duration[i], max_duration[i] = cfg['duration']
            elif max_note_len is not None:
                max_duration[i] = max_note_len

        # print(note_on_map, note_off_map)

        max_t = None if max_time is None else max(max_time, min_time+0.2)

        # print(f'{note_on_map=}')
        try:
            pending.set(query_method(
                note_on_map, #note_off_map,
                min_polyphony=min_polyphony, max_polyphony=max_polyphony,
                min_duration=min_duration, max_duration=max_duration,
                min_time=min_time, max_time=max_t,
                min_vel=min_vel, max_vel=max_vel,
                truncate_quantile_time=tqt,
                truncate_quantile_pitch=tqp,
                truncate_quantile_vel=tqv,
                rhythm_temp=rhythm_temp,
                timing_temp=timing_temp,
                steer_density=steer_density,
                inst_weights=inst_weights,
                no_steer=mode_insts(('input','follow'), allow_muted=False),
            ))
        except NoPossibleEvents:
            pass
            # print(f'stopping; no possible events')
            # pending.stopped = True
            # pending.clear()
        except Exception:
            print(f'WARNING: query failed. {allowed_insts=} {note_on_map=}')
            print(f'{noto.held_notes=}')
            print(f'{config=}')
            traceback.print_exc(file=tui)
            # pending.clear()

    #### MIDI handling

    # print all incoming MIDI for debugging
    if dump_midi:
        @midi.handle
        def _(msg):
            print(msg)

    @midi.handle(type='program_change')
    def _(msg):
        """
        TODO:Program change events set GM instruments on the corresponding channel
        """
        # c = msg.channel+1
        # c = input_channel_map.get(c, c)
        # i = msg.program+1
        # action_queue.append(ft.partial(set_inst(c,i)))
        # if thru:
        raise NotImplementedError

    @midi.handle(type='pitchwheel')
    def _(msg):
        """
        pitchwheel affects steer_pitch
        """
        if thru:
            midi.send(msg)
        # print(msg)
        controls['steer_pitch'] = (msg.pitch+8192)/16384
        # print(controls)
        # print(f'{controls["steer_pitch"]=}')

    # very basic CC handling for controls
    control_cc = {}
    control_osc = {}
    for ctrl in control_meta:
        name = ctrl['name']
        ccs = ctrl.get('control_change', [])
        if isinstance(ccs, Number) or isinstance(ccs, str):
            ccs = (ccs,)
        for cc in ccs:
            control_cc[cc] = ctrl
        control_osc[f'/notochord/homunculus/{name}'] = ctrl
    action_cc = {}
    action_note = {}
    action_osc = {}
    for act in action_meta:
        name = act['name']
        ccs = act.get('control_change', [])
        notes = act.get('note', [])
        if isinstance(ccs, Number) or isinstance(ccs, str):
            ccs = (ccs,)
        if isinstance(notes, Number) or isinstance(notes, str):
            notes = (notes,)
        for cc in ccs:
            action_cc[cc] = act
        for note in notes:
            action_note[note] = act
        action_osc[f'/notochord/homunculus/{name}'] = act

    def dispatch_action(k, v=None):   
        if k=='reset':
            action_queue.append(noto_reset)
        elif k=='query':
            action_queue.append(auto_query)
        elif k=='mute':
            action_queue.append(noto_mute)
        elif k=='preset':
            action_queue.append(ft.partial(set_preset, v or 0))
        elif k=='preset_reset':
            action_queue.append(ft.partial(set_preset, v or 0))
            action_queue.append(noto_reset)
        elif k=='stop':
            action_queue.append(noto_stop)
        else:
            print(f'WARNING: action "{k}" not recognized')

    @midi.handle(type='control_change')
    def _(msg):
        """
        these are global controls listening on all channels: 
        CC 01: steer pitch (>64 higher pitches, <64 lower)
        CC 02: steer density (>64 more simultaneous notes, <64 fewer)
        CC 03: steer rate (>64 more events, <64 fewer)
        """
        if msg.control in control_cc:
            ctrl = control_cc[msg.control]
            name = ctrl['name']
            lo, hi = ctrl.get('range', (0,1))
            controls[name] = msg.value/127 * (hi-lo) + lo
            print(f"{name}={controls[name]}")

        if msg.control in action_cc:
            k = action_cc[msg.control]['name']
            v = action_cc[msg.control].get('value')
            dispatch_action(k, v)

    if midi_control is not None:
        preset_keys = range(112,120)
        momentary_keys = [120]
        for note in preset_keys:
            midi_control.note_on(
                channel=0, note=note, velocity=0)
            
        @midi_control.handle(type=('note_on'))
        def _(msg, port):
            print('control note event', msg)

            if msg.velocity and msg.note in action_note:
                k = action_note[msg.note]['name']
                v = action_note[msg.note].get('value')
                dispatch_action(k, v)
            
            # MIDI feedback (lights)
            if msg.note in preset_keys:
                # if msg.velocity==0: return
                if msg.velocity>0:
                    for note in preset_keys:
                        midi_control.note_on(
                            channel=0, note=note, velocity=0, port=port)
                midi_control.note_on(
                    channel=0, note=msg.note, velocity=70 if msg.velocity else 127, port=port)
            if msg.note in momentary_keys:
                midi_control.note_on(
                    channel=0, note=msg.note, velocity=msg.velocity, port=port)

    if osc_port is not None:
        @osc.handle('/notochord/homunculus/*')
        def _(route, *a):
            # print('OSC:', route, *a)
            if route in control_osc:
                ctrl = control_osc[route]
                name = ctrl['name']
                assert len(a)==1
                arg = a[0]
                assert isinstance(arg, Number)
                lo, hi = ctrl.get('range', (0,1))   
                controls[name] = min(hi, max(lo, arg))
                print(f"{name}={controls[name]}")

            if route in action_osc:
                action = action_osc[route]
                k = action['name']
                dispatch_action(k, v=a[0] if len(a) else action.get('value'))

    input_sw = Stopwatch()
    dropped = set()# (channel, pitch)
    input_dts = []
    @midi.handle(type=('note_on', 'note_off'))
    def _(msg):
        """
        MIDI NoteOn and NoteOff events affect input channels
        e.g. a channel displaying -->01 will listen to note events on channel 1
        a channel displaying 02->03 will follow note events on channel 2
        """
        channel = msg.channel + 1
        # convert from 0-index
        channel = input_channel_map.get(channel, channel)

        if punch_in:
            # set_mode(channel, 'input')
            action_queue.append(ft.partial(set_mode, channel, 'input'))

        cfg = config[channel]

        if not (punch_in or channel in mode_chans('input')):
            print(f"{channel=} {mode_chans('input')=}")
            print(f'WARNING: ignoring MIDI {msg} on non-input channel {channel}')
            return
        
        if cfg['mute']:
            print(f'WARNING: ignoring MIDI {msg} on muted channel {channel}')
            return

        inst = channel_inst(channel)
        pitch = msg.note + cfg.get('note_shift', 0)
        vel = msg.velocity if msg.type=='note_on' else 0
        
        dt = input_sw.punch()
        # print(f'EVENT {dt=} {msg}')
        if len(input_dts) >= 10:
            input_dts.pop(0)
        input_dts.append(dt)
        input_dens = len(input_dts) / sum(input_dts)
        # TODO: 
        # want to drop input when event density is high,
        # not just dt is short
        k = (channel,pitch)
        if vel==0 and k in dropped:
            dropped.remove(k)
            print(f'WARNING: ignoring rate-limited input')
            return
        if vel>0 and limit_input and input_dens>limit_input:
            print(f'WARNING: ignoring rate-limited input {input_dens=}')
            dropped.add(k)
            return 
        
        action_queue.append(ft.partial(
            do_note_input, channel, inst, pitch, vel))

    def do_note_input(channel, inst, pitch, vel):
        # feed event to Notochord
        pending.set({'inst':inst, 'pitch':pitch, 'vel':vel})
        play_event(channel=channel, send=thru, tag='PLAYER')

    @profile(print=print, enable=profiler)
    def auto_event():
        # print('auto_event')
        # 'auto' event happens:
        event = pending.event
        inst, pitch, vel = event['inst'], event['pitch'], math.ceil(event['vel'])
        chan = auto_inst_channel(inst)
        if chan is None:
            raise ValueError(f"channel not found for instrument {inst}")

        # shouldn't happen, but prevent
        # note on which is already playing or note off which is not
        if (vel>0) == ((inst, pitch) in noto.held_notes): 
            print(f'WARNING: re-query for invalid {vel=}, {inst=}, {pitch=}')
            auto_query()
            return
                
        do_stop = False
        do_reset = False
        if (
            (stop_on_end or reset_on_end) 
            and (now()-status['reset_time'] > min_end_time) 
            and pending.sample_end()
            ):
            print('END')
            do_stop = stop_on_end
            do_reset = reset_on_end # needs to happen after final event plays

        play_event(channel=chan, tag='NOTO')

        if do_reset:
            noto_reset()
        elif do_stop:
            noto_stop()

    # @profile(print=print, enable=profiler)
    def maybe_punch_out():
        none_held = set(mode_chans('input'))
        for c,_,_ in history.notes:
            if c in none_held:
                none_held.remove(c)

        evt = history.events
        recent_events = evt[
            (evt.vel > 0) &
            (evt.wall_time_ns > time.time_ns() - punch_out_after*1e9)
            ]
        for c in none_held:
            # print(recent_events.channel)
            # print(f'{c=} {(c not in recent_events.channel)=}')
            if c not in recent_events.channel.values:
                set_mode(c, 'auto')

    @cleanup
    def _():
        """end any remaining notes"""
        print(f'cleanup: {noto.held_notes=}')
        # TODO: this should run in repeat thread..?
        end_held(feed=False, memo='cleanup')

    ### update_* keeps the UI in sync with the state

    def update_config():
        # pass
        # print(config)
        for c,v in config.items():
            # tui.set_channel(c, v)
            tui.call_from_anywhere(tui.set_channel, c, v)

    def update_presets():
        for k,p in enumerate(presets):
            # tui.set_preset(k, p.get('name'))
            tui.call_from_anywhere(tui.set_preset, k, p.get('name'))

    ### set_* does whatever necessary to change channel properties
    ### calls update_config() to keep the UI in sync

    def set_mode(c, m, update=True):
        if c in config:
            prev_m = config[c]['mode']
        else:
            prev_m = None
        if m==prev_m:
            return
        
        if m=='follow':
            if 'source' not in config[c]:
                print('WARNING: follower without a source, setting to 1')
                config[c]['source'] = 1
        
        config[c]['mode'] = m
        print(f'set channel {c} from {prev_m} to {m} mode')

        ### NOTE changing a channel with followers to input causes stuck notes

        if prev_m=='follow':
            # emancipate held notes
            for (dep_c,_,_), note_data in history.note_data.items():
                if dep_c==c:
                    note_data['parent'] = None

        if prev_m=='auto':
            # release held notes
            end_held(channel=c, memo='mode change')

        if m=='auto':
            # immediately refresh prediction
            pending.clear()

        if update:
            update_config()

    def set_inst(c, i, program=None, update=True, allow_pc=True):
        # print(f'set channel {c} instrument {i}')
        req_i = i
        if c in config:
            prev_i = config[c]['inst']
        else:
            prev_i = None

        # don't steal from lower channels
        lower_insts = {d['inst'] for c2, d in config.items() if c2 < c}
        if i in lower_insts:
            # first anon not used by a lower channel
            i = min(set(noto.anon_like(i)) - lower_insts)

        # end held notes on old instrument
        if prev_i!=i and config[c]['mode']!='input':
            end_held(channel=c, memo='instrument change')
            pending.clear()

        # then set config
        config[c]['inst'] = i

        # steal from higher channels
        for c2 in range(c+1,17):
            i2 = config[c2]['inst']
            if i2==i or i2 in lower_insts:
                set_inst(
                    c2, get_free_anon_like(i), 
                    update=False, allow_pc=False)

        # send pc if appropriate
        if send_pc and allow_pc:
            do_send_pc(c, req_i if program is None else program)

        print(f'set channel {c} to instrument {i} (was {prev_i}, requested {req_i})')
        
        if update:
            update_config()

    # @lock
    def set_mute(c, b, update=True):
        config[c]['mute'] = b
        if b:
            print(f'mute channel {c}')
            # release held notes
            if config[c]['mode']!='input':
                end_held(channel=c, memo='mute channel')
                pending.clear()
        else:
            print(f'unmute channel {c}')

        if update:
            update_config()

    def set_cc(c, ccs, update=True):
        # print(f'{c=}, {ccs=}')
        for k,v in ccs.items():
            k = int(k)
            midi.send('control_change', channel=c-1, control=k, value=v)
            # print(f'{(k,v)=}')

    # @lock
    def set_preset(p, update=True):
        nonlocal initial_state
        if p >= len(presets): return
        print(f'load preset: {p+1}')
        preset = presets[p]

        overlay = preset.get('overlay', False)

        state = preset.get('initial_state')
        if state is None:
            if overlay:
                print(f'leaving current initial state')
            else:
                print(f'using global initial state')
                initial_state = global_initial_state
        else:
            print(f'using initial state from preset prompt')
            initial_state = state

        if update:
            for i in range(NotoPresets.n_presets):
                tui.query_one('#'+preset_id(i)).variant = (
                    'warning' if i==p else 'default')

        set_config(preset['channel'], overlay=overlay, update=update)

    def set_config(cfg, overlay=False, update=True):
        # NOTE: config should *not* be updated before calling set_* 
        if overlay:
            # just apply settings which are in the preset
            for c in sorted(cfg):
                v = {**cfg.get(c, {})}
                if 'mode' in v:
                    set_mode(c, v.pop('mode'), update=False)
                if 'inst' in v:
                    set_inst(
                        c, v.pop('inst'), 
                        program=v.pop('program_inst', None), 
                        update=False)
                if 'mute' in v:
                    set_mute(c, v.pop('mute'), update=False)
                if 'cc' in v:
                    set_cc(c, v.pop('cc'), update=False)
                config[c].update(v)
        else:
            # deterministically apply preset on top of defaults
            # for c in range(16,0,-1):    
            for c in range(1,17):    
                v = default_config_channel(c)
                # v.update(cfg.get(c, {}))
                deep_update(v, cfg.get(c, {}))
                print(f'{v=}')
                set_mode(c, v.pop('mode'), update=False)
                set_inst(
                    c, v.pop('inst'), 
                    program=v.pop('program_inst', None), 
                    update=False)
                set_mute(c, v.pop('mute'), update=False)
                set_cc(c, v.pop('cc'), update=False)
                config[c].update(v)

        if update:
            update_config()

    ### action_* runs on key/button press;
    ### invokes cycler / picker logic and schedules set_*

    # this is pretty awful
    # need a better way to reconcile iipyper and textual here
    def action_mode(c):
        if c not in config: return
        # TODO: mode picker
        if config[c]['mode'] == 'auto':
            mode = 'input'
        elif config[c]['mode'] == 'input' and config[c]['source']!=c:
            # TODO: source picker for follow
            mode = 'follow'
        else:
            mode = 'auto'
        action_queue.append(ft.partial(set_mode, c, mode))

    def action_inst(c):
        print(f'inst channel {c}')
        tui.push_screen(InstrumentGroupSelect(c))

    def action_mute(c):
        if i not in config: return
        # set_mute(c, not config[c].get('mute', False))
        action_queue.append(ft.partial(
            set_mute, c, not config[c].get('mute', False)))

    def action_preset(p):
        if p >= len(presets): 
            return
        action_queue.append(ft.partial(set_preset, p))

    ### set actions which have an index argument
    ### TODO move this logic into @tui.set_action

    for i in range(1,17):
        setattr(tui, f'action_mode_{i}', ft.partial(action_mode, i))
        setattr(tui, f'action_inst_{i}', ft.partial(action_inst, i))
        setattr(tui, f'action_mute_{i}', ft.partial(action_mute, i))

    for i in range(NotoPresets.n_presets):
        setattr(tui, f'action_preset_{i}', ft.partial(action_preset, i))

    ### additional key/button actions

    @tui.set_action
    def mute():
        action_queue.append(noto_mute)

    @tui.set_action
    def sustain():
        action_queue.append(ft.partial(noto_mute, sustain=True))
    
    @tui.set_action
    def reset():
        action_queue.append(noto_reset)
    
    @tui.set_action
    def query():
        action_queue.append(auto_query)

    @tui.set_action
    def stop():
        action_queue.append(noto_stop)

    @tui.set_action
    def preset1():
        action_queue.append(ft.partial(set_preset, 0))
    @tui.set_action
    def preset2():
        action_queue.append(ft.partial(set_preset, 1))
    @tui.set_action
    def preset3():
        action_queue.append(ft.partial(set_preset, 2))
    @tui.set_action
    def preset4():
        action_queue.append(ft.partial(set_preset, 3))
    @tui.set_action
    def preset5():
        action_queue.append(ft.partial(set_preset, 4))
    @tui.set_action
    def preset6():
        action_queue.append(ft.partial(set_preset, 5))
    @tui.set_action
    def preset7():
        action_queue.append(ft.partial(set_preset, 6))
    @tui.set_action
    def preset8():
        action_queue.append(ft.partial(set_preset, 7))
    @tui.set_action
    def preset9():
        action_queue.append(ft.partial(set_preset, 8))
    @tui.set_action
    def preset10():
        action_queue.append(ft.partial(set_preset, 9))

    ### TUI classes which close over variables defined in main

    class Instrument(Button):
        """button which picks an instrument"""
        def __init__(self, c, i):
            super().__init__(inst_label(i))
            self.channel = c
            self.inst = i
        def on_button_pressed(self, event: Button.Pressed):
            self.app.pop_screen()
            self.app.pop_screen()
            # set_inst(self.channel, self.inst)
            action_queue.append(ft.partial(
                set_inst, self.channel, self.inst))

    class InstrumentGroup(Button):
        """button which picks an instrument group"""
        def __init__(self, text, c, g):
            super().__init__(text)
            self.channel = c
            self.group = g
        def on_button_pressed(self, event: Button.Pressed):
            # show inst buttons
            tui.push_screen(InstrumentSelect(self.channel, self.group))

    class InstrumentSelect(ModalScreen):
        """Screen with instruments"""
        def __init__(self, c, g):
            super().__init__()
            self.channel = c
            self.group = g

        def compose(self):
            yield Grid(
                *(
                    Instrument(self.channel, i)
                    for i in gm_groups[self.group][1]
                ), id="dialog",
            )

    class InstrumentGroupSelect(ModalScreen):
        """Screen with instrument groups"""
        # TODO: add other features to this screen -- transpose, range, etc?
        def __init__(self, c):
            super().__init__()
            self.channel = c

        def compose(self):
            yield Grid(
                *(
                    InstrumentGroup(s, self.channel, g)
                    for g,(s,_) in enumerate(gm_groups)
                ), id="dialog",
            )

    initial_preset = 0
    if isinstance(preset, str):
        for n,p in enumerate(presets):
            if preset == p['name']:
                initial_preset = n
                break

    config_cli = config or {}
    config = {i:default_config_channel(i) for i in range(1,17)}
    set_preset(initial_preset, update=False)
    if config_ingest is not None:
        set_config(config_ingest, overlay=False, update=False)
    set_config(config_cli, overlay=True, update=False)

    prediction_displayed = [None]
    @repeat(lock=True, err_file=tui)
    # @profile(print=print, enable=profiler)
    def _():
        """Loop, process enqueued actions and check if predicted next event happens"""
        # print(f'repeat {time.time()}')
        for _ in range(8): # process multiple actions per tick
            if len(action_queue):
                # print(action_queue)
                action_queue.pop(0)()
        # TODO: immediate query if an input has just happened

        # if there is no predicted event,
        # or it's not for an auto voice,
        # sample one:
        if pending.gate and not pending.stopped and not pending.is_auto():
            # with profile('auto_query', print=print, enable=profiler):
            auto_query()

        # if unmuted, predicted event is auto, and its time has passed,
        # realize it
        if (
            not testing and
            pending.gate and
            pending.is_auto() and
            pending.time_until() <= 0
            # pending.time_until() <= estimated_feed_latency
            ):
            # with profile('auto_event', print=print, enable=profiler):
            auto_event()


            # query for new prediction
            if not (pending.stopped or debug_query):
                # with profile('auto_query', print=print, enable=profiler):
                auto_query(immediate=True)
        else:
            # otherwise there is a pause, update the UI with next prediction
            if pending.event != prediction_displayed[0]:
                tui.defer(prediction=pending.event)
                prediction_displayed[0] = pending.event
            if punch_in:
                maybe_punch_out()

        # TODO note sure if this is needed anymore
        # adaptive time resolution here -- yield to other threads when 
        # next event is not expected, but time precisely when next event
        # is imminent
        wait = 10e-3
        if len(action_queue):
            wait = 0
        elif pending.is_auto():
            wait = pending.time_until()
        # print(f'{wait=}')
        r = max(0, min(10e-3, wait))
        return r

    @tui.on
    def mount(): 
        update_config()
        update_presets()
        print('MIDI handling:')
        print(midi.get_docs())
        if osc_port is not None:
            print('OSC handling:')
            print(osc.get_docs())
        print(tui_doc)
        print('For more detailed documentation, see:')
        print('https://intelligent-instruments-lab.github.io/notochord/reference/notochord/app/homunculus/')
        print('or run `notochord homunculus --help`')
        print('to exit, use CTRL+C')

        action_queue.append(noto_reset)

        if initial_stop:
            action_queue.append(noto_stop)
        else:
            action_queue.append(ft.partial(
                auto_query, predict_input=False, predict_follow=False))

    if use_tui:
        tui.run()
    else:
        if initial_stop:
            noto_stop()
        else:
            auto_query(predict_input=False, predict_follow=False)
        

### def TUI components ###
class NotoLog(Log):
    value = reactive('')
    def watch_value(self, time: float) -> None:
        self.write_line(self.value)

class NotoPrediction(Static):
    value = reactive(None)
    def watch_value(self, time: float) -> None:
        evt = self.value
        if evt is None:
            s = ''
        else:
            s = f"\tinstrument: {evt['inst']:3d}    pitch: {evt['pitch']:3d}    time: {int(evt['time']*1000):4d} ms    velocity:{int(evt['vel']):3d}     end: {evt.get('end', 0):.5f}"
        self.update(Panel(s, title='prediction'))

class Mixer(Static):
    def compose(self):
        for i in range(1,17):
            yield MixerButtons(i, id=f"mixer_{i}")

def preset_id(i):
    return f"preset_{i}"
def mode_id(i):
    return f"mode_{i}"
def inst_id(i):
    return f"inst_{i}"
def mute_id(i):
    return f"mute_{i}"
class MixerButtons(Static):
    def __init__(self, idx, **kw):
        self.idx = idx
        super().__init__(**kw)

    def compose(self):
        yield Button(
            f"{self.idx:02d}", 
            id=mode_id(self.idx),
            classes="cmode"
            )
        yield Button(
            f"--- \n-----\n-----", 
            id=inst_id(self.idx),
            classes="cinst"
            )
        yield Button(
            f"MUTE", 
            id=mute_id(self.idx),
            classes="cmute"
            )
        
    def on_mount(self) -> None:
        lines = [
            "cycle channel mode:",
            f"-->{self.idx:02d}   input from MIDI channel {self.idx}"
        ]
        if self.idx > 0: lines.append(
            f"{self.idx-1:02d}->{self.idx:02d}  follow channel {self.idx-1}")
        lines.append(
            f"{self.idx:02d}      notochord plays autonomously"
        )
        self.query_one("#"+mode_id(self.idx)).tooltip = '\n'.join(lines)
        self.query_one("#"+inst_id(self.idx)).tooltip = "open instrument picker"
        self.query_one("#"+mute_id(self.idx)).tooltip = f"mute channel {self.idx}"
        
class NotoPresets(Static):
    n_presets = 10
    def compose(self):
        for i in range(NotoPresets.n_presets):
            yield Button('---', id=preset_id(i))

    def on_mount(self) -> None:
        for i in range(NotoPresets.n_presets):
            self.query_one("#"+preset_id(i)).tooltip = f"load preset {i+1}"

class NotoControl(Static):
    def compose(self):
        # yield NotoToggle()
        # yield Checkbox("Mute", id="mute")
        yield Button("Mute", id="mute", variant="error")
        yield Button("Sustain", id="sustain", variant="primary")
        yield Button("Query", id="query")
        yield Button("Stop", id="stop")
        yield Button("Reset", id="reset", variant="warning")

    def on_mount(self) -> None:
        self.query_one("#mute").tooltip = "master mute notochord"
        self.query_one("#sustain").tooltip = "master sustain -- prevent any NoteOffs from notochord"
        self.query_one("#query").tooltip = "manually query a new event"
        self.query_one("#reset").tooltip = "reset notochord"

class NotoTUI(TUI):
    CSS_PATH = 'homunculus.css'

    BINDINGS = [
        ("m", "mute", "Mute Notochord"),
        ("s", "sustain", "Sustain"),
        ("q", "query", "Re-query Notochord"),
        ("x", "stop", "Stop Notochord"),
        ("r", "reset", "Reset Notochord"),
        ("1", "preset1", ""),
        ("2", "preset2", ""),
        ("3", "preset3", ""),
        ("4", "preset4", ""),
        ("5", "preset5", ""),
        ("6", "preset6", ""),
        ("7", "preset7", ""),
        ("8", "preset8", ""),
        ("9", "preset9", ""),
        ("0", "preset10", ""),
        ]
    
    def compose(self):
        """Create child widgets for the app."""
        yield Header()
        yield self.std_log
        yield NotoLog(id='note')
        yield NotoPrediction(id='prediction')
        yield Mixer()
        yield NotoPresets()
        yield NotoControl()
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(NotoPrediction).tooltip = "displays the next predicted event"
        # print(self.screen)

    def set_preset(self, idx, name):
        if idx < NotoPresets.n_presets:
            node = self.query_one('#'+preset_id(idx))
            node.label = str(idx) if name is None else name
        else:
            self.write(f'warning: more than {NotoPresets.n_presets} presets\n')
        # node.label = name

    def set_channel(self, chan, cfg):
        # print(f'set_channel {cfg}')
        inst_node = self.query_one('#'+inst_id(chan))
        mode_node = self.query_one('#'+mode_id(chan))
        mute_node = self.query_one('#'+mute_id(chan))

        if cfg is None:
            inst_node.variant = 'default'
            mute_node.variant = 'default'
            mode_node.variant = 'default'
            return
        
        mode = cfg['mode']
        mute = cfg['mute']
        inst = cfg['inst']

        inst_node.label = inst_label(inst)

        if mode=='auto':
            mode_node.label = f"{chan:02d}"
        elif mode=='input':
            mode_node.label = f"-->{chan:02d}"
        elif mode=='follow':
            mode_node.label = f"{cfg['source']:02d}->{chan:02d}"

        if mute:
            mode_node.variant = 'default'
            inst_node.variant = 'default'
            mute_node.label = 'UNMUTE'
            mute_node.variant = 'error'
        else:
            mode_node.variant = 'primary'
            inst_node.variant = 'warning'
            mute_node.label = 'MUTE'
            mute_node.variant = 'default'

gm_names = [
    '_SEQ_\nSTART',
    'GRAND\nPIANO', 'BRGHT\nPIANO', 'EGRND\nPIANO', 'HONKY\n-TONK', 
    'RHODE\nPIANO', 'FM   \nPIANO', 'HRPSI\nCHORD', 'CLAV \n INET',
    'CEL  \n ESTA', 'GLOCN\nSPIEL', 'MUSIC\n BOX ', 'VIBRA\nPHONE', 
    'MAR  \n IMBA', 'XYLO \nPHONE', 'TUBLR\n BELL', 'DULCI\n  MER',
    'DRAWB\nORGAN', 'PERC \nORGAN', 'ROCK \nORGAN', 'CHRCH\nORGAN', 
    'REED \nORGAN', 'ACCOR\n DION', 'HARMO\n NICA', 'BANDO\n NEON',  
    'A-GTR\nNYLON', 'A-GTR\nSTEEL', 'E-GTR\nJAZZ ', 'E-GTR\nCLEAN', 
    'E-GTR\nMUTED', 'E-GTR\nDRIVE', 'E-GTR\nDIST ', 'E-GTR\nHRMNC',  
    'A-GTR\n BASS', 'EBASS\nFINGR', 'EBASS\n PICK', 'EBASS\nFRTLS', 
    'SLAP \nBASS1', 'SLAP \nBASS2', 'SYNTH\nBASS1', 'SYNTH\nBASS2',  
    'STRNG\nVIOLN', 'STRNG\nVIOLA', 'STRNG\nCELLO', 'STRNG\nCBASS', 
    'STRNG\nTREMO', 'STRNG\nPIZZC', 'ORCH \n HARP', 'TIMP \n  ANI',  
    'STRNG\nENSB1', 'STRNG\nENSB2', 'SYNTH\nSTRG1', 'SYNTH\nSTRG2', 
    'CHOIR\n AAH ', 'CHOIR\n OOH ', 'SYNTH\nVOICE', 'ORCH \n  HIT',  
    'TRUM \n  PET', 'TROM \n BONE', 'TUBA \n     ', 'MUTED\nTRMPT', 
    'FRNCH\nHORN ', 'BRASS\nSECTN', 'SYNTH\nBRSS1', 'SYNTH\nBRSS2',  
    'SPRNO\n SAX ', 'ALTO \n  SAX', 'TENOR\n  SAX', 'BARI \n  SAX', 
    'OBOE \n     ', 'ENGLS\nHORN ', 'BASS \n  OON', 'CLARI\n  NET',  
    'PICC \n  OLO', 'FLUTE\n     ', 'RECO \n RDER', ' PAN \nFLUTE',
    'BLOWN\nBOTTL', 'SHAKU\nHACHI', 'WHIS \n  TLE', 'OCA  \n RINA',  
    'LEAD1\nSQUAR', 'LEAD2\n SAW ', 'LEAD3\n TRI ', 'LEAD4\nCHIFF',
    'LEAD5\nCHRNG', 'LEAD6\nVOICE', 'LEAD7\nFIFTH', 'LEAD8\nSYNTH',  
    'PAD 1\nNWAGE', 'PAD 2\n WARM', 'PAD 3\n POLY', 'PAD 4\nCHOIR',
    'PAD 5\nGLASS', 'PAD 6\nMETAL', 'PAD 7\n HALO', 'PAD 8\nSWEEP',
    'FX  1\n RAIN', 'FX  2\nSDTRK', 'FX  3\nCRYST', 'FX  4\nATMOS',
    'FX  5\nBRGHT', 'FX  6\nGOBLN', 'FX  7\nECHOS', 'FX  8\nSCIFI',
    'SITAR\n     ', 'BANJO\n     ', 'SHAM \n ISEN', 'KOTO \n     ',
    'KAL  \n IMBA', 'BAG  \n PIPE', 'FID  \n  DLE', 'SHA  \n  NAI',  
    'TINKL\nBELL ', 'AGO  \n   GÔ', 'STEEL\nDRUM ', 'WOOD \nBLOCK', 
    'TAIKO\nDRUM ', 'MELO \n  TOM', 'SYNTH\nDRUM ', ' REV \nCYMBL',  
    'GTR  \n FRET', 'BRE  \n  ATH', ' SEA \nSHORE', 'BIRD \nTWEET',
    'TELE \nPHONE', 'HELI \nCOPTR', 'APP  \nLAUSE', 'GUN  \n SHOT',  
] + (
    ['STD  \n  KIT']*8 + 
    ['ROOM \n  KIT']*8 + 
    ['ROCK \n  KIT']*8 + 
    ['ELCTR\n  KIT']*8 + 
    ['JAZZ \n  KIT']*8 + 
    ['BRUSH\n  KIT']*8 + 
    ['ORCHS\n  KIT']*8 + 
    ['SFX  \n  KIT']*8 + 
    ['DRUM \n KIT?']*64 
) + ['ANON \n MEL ']*32 + ['ANON \nDRUMS']*32
gm_groups = [
    ('PIANO\n     ', range(1,9)),
    ('CHROM\nPERC ', range(9,17)),
    ('ORGAN\n     ', range(17,25)),
    ('GUI  \n TARS', range(25,33)),
    ('BASS \n GTRS', range(33,41)),
    ('STRIN\n   GS', range(41,49)),
    ('ENSEM\n BLES', range(49,57)),
    ('BRASS\n     ', range(57,65)),
    ('REEDS\n     ', range(65,73)),
    ('PIPES\n     ', range(73,81)),
    ('SYNTH\nLEADS', range(81,89)),
    ('SYNTH\nPADS ', range(89,97)),
    ('SYNTH\nFX   ', range(97,105)),
    ('MISC \n  MEL', range(105,113)),
    ('MISC \n PERC', range(113,121)),
    ('SOUND\nFX   ', range(121,129)),
    ('DRUM \n KITS', [128+i for i in (1,9,17,25,33,41,49,57)]),
    # (' STD \nDRUMS', range(129,137)),
    # ('ROOM \nDRUMS', range(137,145)),
    # ('ROCK \nDRUMS', range(145,153)),
    # ('ELCTR\nDRUMS', range(153,161)),
    # ('JAZZ \nDRUMS', range(161,169)),
    # ('BRUSH\nDRUMS', range(169,177)),
    # ('ORCH \nDRUMS', range(177,185)),
    # (' SFX \nDRUMS', range(185,193)),
    ('ANON\n  MEL', range(257,273)),
    ('ANON\n DRUM', range(289,305)),
]
def inst_label(i):
    if i is None:
        return f"--- \n-----\n-----"
    return f'{i:03d} \n{gm_names[i]}'
### end def TUI components###

if __name__=='__main__':
    run(main)
