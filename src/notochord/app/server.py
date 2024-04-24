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
        print(f"{address} {a} {kw}")
        predictor.feed(*a, **kw) 

    @osc.handle('/notochord/query', return_port=send_port)
    def _(address, **kw):
        print(f"{address} {kw}")
        r = predictor.query(**kw) 
        return (
            '/return/notochord/query', 
            *[x for pair in r.items() for x in pair])
    
    @osc.handle('/notochord/query_feed', return_port=send_port)
    def _(address, **kw):
        print(f"{address} {kw}")
        r = predictor.query_feed(**kw) 
        return (
            '/return/notochord/query', 
            *[x for pair in r.items() for x in pair])
    
    @osc.handle('/notochord/feed_query', return_port=send_port)
    def _(address, a:Splat[4], **kw):
        print(f"{address} {a} {kw}")
        r = predictor.feed_query(*a, **kw) 
        return (
            '/return/notochord/query', 
            *[x for pair in r.items() for x in pair])

    @osc.handle('/notochord/reset', return_port=send_port)
    def _(address, **kw):
        print(f"{address} {kw}")
        predictor.reset(**kw) 

    @osc.handle('/notochord/load', return_port=send_port)
    def _(address, path:str):
        # `nonlocal` is needed to assign to closed-over name
        nonlocal predictor
        predictor = Notochord.from_checkpoint(path)
        predictor.eval()
        predictor.reset()
        print('notochord created')


if __name__=='__main__':
    run(main)
