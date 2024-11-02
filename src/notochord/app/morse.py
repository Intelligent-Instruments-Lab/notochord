"""
Notochord + Language Model + Morse Code.

Authors:
  Victor Shepardson
  Intelligent Instruments Lab 2024
"""

import random
from copy import deepcopy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from notochord import Notochord
from iipyper import MIDI, run, Stopwatch, cleanup, repeat, _lock
import time

torch.set_num_threads(1)

def now():
    return time.time()

def main(
        noto_channel=1,
        noto_inst=20, # General MIDI numbered from 1 (see Notochord.feed docstring)
        midi_in=None, # MIDI port for player input
        midi_out=None, # MIDI port for Notochord output
        notochord="notochord-latest.ckpt", # Notochord checkpoint
        lm="state-spaces/mamba-790m-hf",
        verbose=0,
        prompt='morse code is ',
        dit_dur=0.05,
        send_pc=True,
        lm_temp=1,
        noto_temp=1.5,
        buffer_events=3,
        morse_chars = '·–|'
        ):
    midi = MIDI(midi_in, midi_out)

    print(f'loading notochord model {notochord}')
    noto = Notochord.from_checkpoint(notochord)
    noto.eval()

    c_dot, c_dash, c_space = morse_chars

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

    if send_pc:
        do_send_pc(noto_channel, noto_inst)

    print(f'loading language model {lm}')
    tokenizer = AutoTokenizer.from_pretrained(lm)
    lm = AutoModelForCausalLM.from_pretrained(lm)
    lm.eval()

    tokens = tokenizer.batch_decode(range(len(tokenizer)))
    alphabet = 'abcdefghijklmnopqrstuvwxyzðþöéæ"\'+-=/.,:;!?() '
    alphabet = set(alphabet)
    # indices of within-vocab tokens
    token_idx = [
        i for i,t in enumerate(tokens) if all(c in alphabet for c in t)]
    tokens = [tokens[i] for i in token_idx]
    # token_node_index maps tokens directly to tree nodes for speed
    token_root, token_node_index = TokenNode.make_tree(tokens)

    def set_token_probs(probs):
        for t,p in zip(tokens, probs):
            token_node_index[t].prob = p

    def lm_prompt(prompt):
        """feed a text to the language model and return probabilities for the next token and model state"""
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        with torch.inference_mode():
            state = lm(input_ids, use_cache=True)
            probs = state.logits[0,-1,token_idx].softmax(0).tolist()
        return probs, state

    def lm_feed_query(token, state):
        """feed a token to the language model and return probabilities for the next token and model state"""
        with torch.inference_mode():
            input_ids = tokenizer(token, return_tensors='pt')['input_ids']
            # print(input_ids, tokens[input_ids.item()])
            cache = None if state is None else state.cache_params
            # t = now()
            state = lm(input_ids, use_cache=True, cache_params=cache)
            probs = state.logits[0,-1,token_idx].softmax(0).tolist()
            # print('LM query ms:', int(1000*(now()-t)))

        return probs, state
    
    def set_morse_probs_from_token(morse_tree, token_tree):
        """set the probability of every character in morse to be the sum of probabilities of all currently possible tokens which begin with that character"""
        for node in morse_tree.nodes():
            node.prob = token_tree.children.get(
                node.char, MorseNode()).subtree_prob()
            if verbose > 2:
                print('----setting', node)

    def combine_probs(noto_dt_probs, lm_dt_probs):
        if len(noto_dt_probs)==3:
            noto_dt_probs = [noto_dt_probs[0], sum(noto_dt_probs[1:])]
        return [
            p1**(1/noto_temp) * p2**(1/lm_temp) 
            for p1, p2 in zip(noto_dt_probs, lm_dt_probs)]

    # TODO cache to disk so no delay when prompt is repeated
    if prompt.endswith(' '):
        prompt = prompt[:-1]
        initial_space = True
    else:
        initial_space = False
    initial_token_probs, initial_lm_state = lm_prompt(prompt)

    class State:
        def __init__(self):
            self.reset()
        def reset(self):
            # self.next_event = None
            # self.next_event_time = None
            self.next_events = []
            self.next_event_times = []
            self.last_event_time = None
            self.noto_held_notes = set()
            self.synth_held_notes = set()

            self.token_probs = initial_token_probs
            self.lm_state = deepcopy(initial_lm_state)
            self.morse_tree = MorseNode.make_tree()

            code = self.morse_tree.make_code()
            self.morse = ''.join(
                c_space+' ' if c==' ' else ''.join(c_dot if e else c_dash for e in code[c])
                for c in prompt
                )
            self.text = prompt
            self.token_seq = [] # TODO add tokenizer output
            self.midi_seq = [] # TODO construct MIDI prompt
            self.is_gap = True
            self.first_event = True
            
            self.token_tree = token_root
            set_token_probs(self.token_probs)
            # self.token_tree.set_probs(token_idx, self.token_probs)
            # self.token_tree = TokenNode.make_tree(
                # tokens, self.token_probs)
            # self.token_tree = TokenNode.make_tree(
            #     tokens, self.token_probs, alphabet)
            if initial_space:
                self.token_tree = self.token_tree.traverse(' ')
                self.text += ' '
            set_morse_probs_from_token(self.morse_tree, self.token_tree)

    S = State()

    def release_all():
        """end any remaining notes"""
        for chan,pitch in S.synth_held_notes:
            midi.note_on(note=pitch, velocity=0, channel=chan-1)

    def reset():
        # TODO expose this
        noto.reset()
        S.reset()
        release_all()

    reset()

    def noto_query(dt=None):
        """query notochord + LM as needed for the next MIDI event"""
        query = {}
        held_pitches = {p for _,p in S.noto_held_notes}
        if S.is_gap:
            # if determining the length of a gap,
            # that means we are predicting the next noteOff
            query['min_vel'] = 1
            query['include_pitch'] = set(range(20,100)) - held_pitches
            # query['next_pitch'] = 40
            # query['include_pitch'] = set(range(20,100)) - held_pitches

        else:
            # if determining the length of the tone,
            # that means we are predicting the next noteOff
            query['next_vel'] = 0
            # print(f'{held_pitches=}')
            if len(held_pitches)>1:
                query['include_pitch'] = held_pitches
            else:
                query['next_pitch'] = list(held_pitches)[0]
        # query['include_inst'] = [1, 8] 
        query['next_inst'] = noto_inst

        if S.is_gap:
            dts = [1,3,7]
        else:
            dts = [1,3]
        dts = torch.tensor([dts])*dit_dur

        with torch.inference_mode():
            event = noto.query(return_params=True, **query)
            time_sample = noto.time_dist(event['time_params'], dts)
            noto_dt_probs = time_sample['log_prob'].exp().squeeze()
            noto_dt_probs = [x.item() for x in noto_dt_probs]

        # print(f'{event["vel"]=}, {event["pitch"]=}')

        # get LM dt probs from tree
        if S.is_gap:
            if verbose > 1:
                print(f'------noto prob [1u, 3u, 7u] {noto_dt_probs}')
            # 1: char continues; 3,7: end of char
            # here, the 3,7 probs only need to add up correctly (hence prob/2)
            # we'll check the LM again only if the token ends
            lm_dt_probs = [S.morse_tree.children_prob(), S.morse_tree.prob]
            if verbose > 1:
                print(f'------LM prob [{S.morse_tree.children_chars()}, {S.morse_tree.char}] {lm_dt_probs}')
        else:
            if verbose > 1:
                print(f'------noto prob [1u, 3u] {noto_dt_probs}')
            # 1: dit, 3: dah
            dit = S.morse_tree.traverse(True)
            dah = S.morse_tree.traverse(False)
            lm_dt_probs = [dit.subtree_prob(), dah.subtree_prob()]
            if verbose > 1:
                print(f'------LM prob [dit, dah] [{dit.chars()}, {dah.chars()}] {lm_dt_probs}')

        # print(f'{"gap" if is_gap else "tone"} {lm_dt_probs=}')

        probs = combine_probs(noto_dt_probs, lm_dt_probs)
        if verbose > 1:
            print(f'------combined {probs=}')

        # here we sample 1 vs 3 or 1 vs 3,7
        is_1u = random.random() < probs[0] / (sum(probs) + 1e-10)

        is_7u = False # can be changed only when new token starts, below
        if S.is_gap:
            if is_1u:
                if verbose > 0:
                    print('----character continues')
            else:
                c = S.morse_tree.char
                if verbose > 0:
                    print('--character complete:', c)
                S.text += c
                S.morse += ' '

                # put the new character into the current token
                S.token_tree = S.token_tree.traverse(c)
                # print(f'{token_tree.token} {"".join(token_tree.children)}')
                # possible end of token + LM query
                # t = now()
                if S.token_tree.sample_terminal(verbose=verbose):
                    if verbose > 0:
                        print('token complete:', S.token_tree.token)
                    S.token_seq.append(S.token_tree.token)
                    S.token_probs, S.lm_state = lm_feed_query(
                        S.token_tree.token, S.lm_state)
                    S.token_tree = token_root
                    set_token_probs(S.token_probs)

                    # now sample nonspace vs space
                    noto_dt_probs = noto_dt_probs[1:]
                    space_prob = S.token_tree.children[' '].subtree_prob()
                    # don't needlessly traverse entire tree to sum probs
                    lm_dt_probs = [
                        # S.token_tree.subtree_prob()-space_prob, space_prob]
                        sum(S.token_probs)-space_prob, space_prob]

                    # NOTE: prob of space characters coming only from LM
                    # notochord is usually reluctant 
                    # probs = combine_probs(noto_dt_probs, lm_dt_probs)
                    probs = lm_dt_probs
                    is_7u = random.random() < probs[1] / sum(probs)
                    if verbose > 1:
                        print(f'------LM prob [nonspace, space] {lm_dt_probs}')

                    if is_7u:
                        if verbose > 0:
                            print('space')
                        S.token_tree = S.token_tree.traverse(' ')
                        S.text += ' '
                        S.morse += c_space+' '
                    else:
                        if verbose > 0:
                            print('word continues')

                else:
                    if verbose > 0:
                        print('--token continues')

                # print('check token end ms:', int(1000*(now()-t)))

                # get next character probs from the token tree
                S.morse_tree = MorseNode.make_tree()
                set_morse_probs_from_token(S.morse_tree, S.token_tree)
                        
        else:
            S.morse_tree = S.morse_tree.traverse(is_1u)
            S.morse += c_dot if is_1u else c_dash
            if verbose > 0:
                print('----dit' if is_1u else '----dah')

        event['time'] = dts.squeeze()[0 if is_1u else (2 if is_7u else 1)]

        if S.first_event:
            # set the initial dt to 0 to avoid biasing notochord toward 1u
            event['time'] = 0
            S.first_event = False

        with torch.inference_mode():
            noto.feed(**event)
        S.midi_seq.append({k:event[k] for k in (
            'inst','pitch','time','vel')})
        
        k = (noto_channel, event['pitch'])
        if event['vel']>0:
            S.noto_held_notes.add(k)
        else:
            try:
                S.noto_held_notes.remove(k)
            except Exception:
                print(f'{k} not in noto_held_notes')

        S.is_gap = not S.is_gap

        print(S.morse)
        print(S.text)

        return event

    @midi.handle(type='program_change')
    def _(msg):
        """
        Program change events set instruments
        """
        nonlocal noto_inst
        if msg.channel == noto_channel:
            noto_inst = msg.program

    @midi.handle(type='control_change', control=0, channel=noto_channel-1)
    def _(msg):
        """
        any CC0 message resets Notochord
        """
        noto.reset()

    @midi.handle(type=('note_on', 'note_off'), channel=noto_channel-1)
    def _(msg):
        """
        MIDI NoteOn events
        """
        if noto is None:
            print('Notochord model not loaded')
            return
    
    # @repeat(1e-3, lock=True)
    @repeat(lock=True)
    # @repeat()
    def _():
        """Loop, consuming scheduled events"""
        # print(S.next_event is None, now(), S.next_event_time)

        # print(f'{now()=}')

        late = now() - S.next_event_times[0] if len(S.next_events) else -1
        # late = -1 if S.next_event is None else now() - S.next_event_time
        if late >= 0:
            if late > 10e-3:
                print(f'{late=}')
            # send MIDI
            event = S.next_events.pop(0)
            pitch = int(event['pitch'])
            vel = int(event['vel'])
            midi.note_on(note=pitch, velocity=vel, channel=noto_channel-1)

            k = (noto_channel, pitch)
            if vel:
                S.synth_held_notes.add(k)
            else:
                try:
                    S.synth_held_notes.remove(k)
                except Exception:
                    print(f'{k} not in synth_held_notes')

            # S.last_event_time = S.next_event_time
            # S.next_event = None
            S.last_event_time = S.next_event_times.pop(0)

        dt = S.next_event_times[0] - now() if len(S.next_event_times) else 10e-3
        return max(0, min(10e-3, dt))

    @repeat(10e-3, lock=False)
    def _():
        """Loop, predicting next events"""
        if len(S.next_events) < buffer_events:
            final_time = (
                S.next_event_times[-1] if len(S.next_event_times) 
                else S.last_event_time)
            
            # t = now()
            # if final_time is not None:
                # print('buffered time ms:', int(1000*(final_time-t)))
            event = noto_query()
            # print('query time ms:', int(1000*(now()-t)))

            with _lock:
                S.next_events.append(event)

                if final_time is None:
                    S.next_event_times.append(now())
                else:
                    S.next_event_times.append(final_time + float(event['time']))

        # print('buffered events:', len(S.next_events))

        # return max(0, min(10e-3, S.next_event_times[0] - now()))
        # return max(0, min(10e-3, S.next_event_time - now()))


    @cleanup
    def _():
        release_all()

class MorseNode:
    def __init__(self, 
            char:str=None, 
            prob:float=0, 
            dit:'MorseNode'=None, 
            dah:'MorseNode'=None
        ):
        self.char = char
        self.prob = prob
        self.dit = dit
        self.dah = dah

    def subtree_prob(self):
        prob = self.prob
        for d in (self.dit, self.dah):
            if d is not None: 
                prob += d.subtree_prob()
        return prob
    
    def children_prob(self):
        return self.subtree_prob() - self.prob
    
    def is_leaf(self):
        return self.dit is None and self.dah is None
    
    def nodes(self):
        nodes = [self]
        for d in (self.dit, self.dah):
            if d is not None:
                nodes.extend(d.nodes())
        return nodes
    
    def chars(self):
        return ''.join(
            MorseNode.char for MorseNode in self.nodes() if MorseNode.char is not None)
    
    def children_chars(self):
        chars = ''
        for d in (self.dit, self.dah):
            if d is not None:
                chars += d.chars()
        return chars
    
    def traverse(self, *seq):
        if not len(seq):
            return self
        b, *seq = seq
        d = self.dit if b else self.dah
        # print(b, seq, d)
        return MorseNode() if d is None else d.traverse(*seq)
        
    def __repr__(self):
        s = f'{self.char} {self.prob}'
        for d in (self.dit, self.dah):
            if d is not None: 
                s += f' {d.char}'
        return s
    
    def make_code(self, path=None):
        """return a mapping from characters to morse"""
        code = {}
        if path is None: path = tuple()
        if self.char is not None:
            code[self.char] = path
        for d,b in ((self.dit, True), (self.dah, False)):
            if d is not None:
                code.update(d.make_code(path+(b,)))
        return code

    @classmethod
    def make_tree(cls):
        return MorseNode(
            dit=MorseNode('e', 12.7,
                dit=MorseNode('i', 7.0,
                    dit=MorseNode('s', 6.3,
                        dit=MorseNode('h', 6.1,
                            dit=MorseNode('5'),
                            dah=MorseNode('4'),
                        ),
                        dah=MorseNode('v', 1.0,
                            # dit=MorseNode('ŝ'),
                            dah=MorseNode('3'),
                        ),
                    ),
                    dah=MorseNode('u', 2.8,
                        dit=MorseNode('f', 2.2,
                            dit=MorseNode('é'),
                            dah=MorseNode('4'),
                        ),
                        dah=MorseNode(#'ü',
                            dit=MorseNode('ð',
                                dit=MorseNode('?')
                            ),
                            dah=MorseNode('2'),
                        ),
                    ),
                ),
                dah=MorseNode('a', 8.2,
                    dit=MorseNode('r', 6.0,
                        dit=MorseNode('l', 4.0,
                            # dit=MorseNode(),
                            dah=MorseNode(#'è',
                                dit=MorseNode('"')
                            ),
                        ),
                        dah=MorseNode('æ',
                            dit=MorseNode('+',
                                dah=MorseNode('.')
                            )
                        ),
                    ),
                    dah=MorseNode('w', 2.4,
                        dit=MorseNode('p', 1.9,
                            dit=MorseNode('þ'),
                            dah=MorseNode(#'à',
                                dit=MorseNode('@'),
                            ),
                        ),
                        dah=MorseNode('j', 0.2,
                            # dit=MorseNode('ĵ'),
                            dah=MorseNode('1',
                                dit=MorseNode("'")
                            ),
                        ),
                    ),
                ),
            ),
            dah=MorseNode('t', 9.1,
                dit=MorseNode('n', 6.7,
                    dit=MorseNode('d', 4.3,
                        dit=MorseNode('b', 1.5,
                            dit=MorseNode('6',
                                dah=MorseNode('-')
                            ),
                            dah=MorseNode('='),
                        ),
                        dah=MorseNode('x', 0.2,
                            dit=MorseNode('/')
                        ),
                    ),
                    dah=MorseNode('k', 0.8,
                        dit=MorseNode('c', 2.8,
                            # dit=MorseNode('ç'),
                            dah=MorseNode(#empty MorseNode
                                dit=MorseNode(';'),
                                dah=MorseNode('!')
                            ),
                        ),
                        dah=MorseNode('y', 2.0,
                            dit=MorseNode('(',
                                dah=MorseNode(')')
                            )
                        ),
                    ),
                ),
                dah=MorseNode('m', 2.4,
                    dit=MorseNode('g', 2.0,
                        dit=MorseNode('z', 0.1,
                            dit=MorseNode('7'),
                            dah=MorseNode(
                                dah=MorseNode(',')
                            ),
                        ),
                        dah=MorseNode('q', 0.1),
                    ),
                    dah=MorseNode('o', 7.5,
                        dit=MorseNode('ö',
                            dit=MorseNode('8',
                                dit=MorseNode(':')
                            ),         
                        ),
                        dah=MorseNode(
                            dit=MorseNode('9'),
                            dah=MorseNode('0')
                        ),
                    ),
                ),
            )
        )

class TokenNode:
    def __init__(self, 
            token:str=None, 
            prob:float=0, 
            children=None,
        ):
        self.token = token
        self.prob = prob
        if children is None:
            children = {}
        self.children = children
            
    def subtree_prob(self):
        prob = self.prob
        for node in self.children.values():
            prob += node.subtree_prob()
        return prob
    
    def children_prob(self):
        return self.subtree_prob() - self.prob
    
    def is_leaf(self):
        return not len(self.children)
    
    def traverse(self, s):
        if not len(s):
            return self
        c, *s = s
        node = self.children.get(c)
        return TokenNode() if node is None else node.traverse(s)
    
    def nodes(self):
        yield self
        for child in self.children.values():
            for node in child.nodes():
                yield node
    
    def sample_terminal(self, verbose=0):
        if verbose>2:
            print(f'----terminal prob: {self.prob / self.subtree_prob()}')
        if self.prob > 0:
            if random.random() < self.prob / self.subtree_prob():
                return True
        return False
        
    def __repr__(self):
        s = f'{self.token} {self.prob} {"".join(self.children)}'
        return s

    @classmethod
    def make_tree(cls, tokens):
        root = TokenNode()
        index = {}
        for token in tokens:
            tree = root
            for c in token:
                if c not in tree.children:
                    tree.children[c] = TokenNode()
                tree = tree.children[c]
            tree.token = token
            index[token] = tree
        return root, index

if __name__=='__main__':
    run(main)
