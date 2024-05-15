"""
Authors:
  Victor Shepardson
  Jack Armitage
  Intelligent Instruments Lab 2022
"""

from notochord import Notochord
from iipyper import OSC, run
from iipyper.types import *
import numpy as np
from time import time

def main(host="127.0.0.1", receive_port=9999, send_port=None, 
        checkpoint="notochord-latest.ckpt"):
    osc = OSC(host, receive_port)

    if checkpoint is not None:
        predictor = Notochord.from_checkpoint(checkpoint)
        predictor.eval()
        predictor.reset()
        print('notochord created')
    else:
        predictor = None
 
    @osc.handle('/notochord/feed', return_port=send_port)
    def _(address, a:Splat[4], **kw):
        """
        Feed an event with given instrument, pitch, time and velocity
        For example: 
        /notochord/feed 1 60 0.2 99
            feed the model an event with instrument 1 (grand piano),
            MIDI number 60 (middle C), 200ms elapsed since last event,
            MIDI velocity 99
        """
        print(f"{address} {a} {kw}")
        predictor.feed(*a, **kw) 

    @osc.handle('/notochord/query', return_port=send_port)
    def _(address, **kw):
        """
        Arguments are given in key value pairs, mapping onto the Python API.
        For example:
        /notochord/query
            sample the next event with no constraints
        /notochord/query next_inst 1
            constrain next instrument to MIDI program 1 (grand piano)
        /notochord/query next_pitch 60
            constrain next pitch to MIDI number 60 (middle C)
        /notochord/query next_time 0.3
            constrain delta time from previous event to exactly 300ms
        /notochord/query next_vel 0
            constrain next velocity to 0 (noteOff)
        /notochord/query next_inst 129 next_vel 127
            constrain next instrument to 129 (standard drums),
            and next velocity to 127 (max)
        /notochord/query include_inst [ 1 2 3 ]
            constrain next instrument to be from within the given list 
            (client must support OSC arrays)
        /notochord/query pitch_temp 0.5 min_time 0.1
            query with sampling temperature for pitch reduced to 0.5,
            and constrain delta time since last event to at least 100ms

        for all options, see https://intelligent-instruments-lab.github.io/notochord/reference/notochord/model/#notochord.model.Notochord.query
        """
        print(f"{address} {kw}")
        r = predictor.query(**kw) 
        return (
            '/return/notochord/query', 
            *[x for pair in r.items() for x in pair])
    
    @osc.handle('/notochord/query_feed', return_port=send_port)
    def _(address, **kw):
        """
        /notochord/query_feed {feed arguments} {query arguments}
            query for an event and immediately feed it in one call
        """
        print(f"{address} {kw}")
        r = predictor.query_feed(**kw) 
        return (
            '/return/notochord/query', 
            *[x for pair in r.items() for x in pair])
    
    @osc.handle('/notochord/feed_query', return_port=send_port)
    def _(address, a:Splat[4], **kw):
        """
        /notochord/feed_query {feed arguments} {query arguments}
            feed and event, then query for the next in one call
        """
        print(f"{address} {a} {kw}")
        r = predictor.feed_query(*a, **kw) 
        return (
            '/return/notochord/query', 
            *[x for pair in r.items() for x in pair])

    @osc.handle('/notochord/reset', 
                return_port=send_port)
    def _(address, **kw):
        """
        /notochord/reset
            reset the model and prime it with start-of-sequence token
        /notochord/reset "start" false
            reset the model without sending a start token
        """
        print(f"{address} {kw}")
        predictor.reset(**kw) 

    @osc.handle('/notochord/load', 
                return_port=send_port)
    def _(address, path:str):
        """
        load your own pretrained notochord model
        /notochord/load "/path/to/my/notochord/model"
        """
        # `nonlocal` is needed to assign to closed-over name
        nonlocal predictor
        predictor = Notochord.from_checkpoint(path)
        predictor.eval()
        predictor.reset()
        print('notochord created')

    print('usage:')
    print(osc.get_docs())


if __name__=='__main__':
    run(main)
