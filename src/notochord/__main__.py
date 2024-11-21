import sys, subprocess

import fire
from iipyper import run

from notochord import Notochord
from notochord.app import *

def help():
    print("""
    available subcommands:
        server: run the Notochord API OSC server
        homunculus: run the Notochord homunculus TUI
        harmonizer: run the Notochord harmonizer TUI
        improviser: run the Notochord improviser TUI
        txalaparta: run the txalaparta app
        morse: run the morse code app
        prompt: run a MIDI file through a notochord model and cache hidden states
        train: train a Notochord model (GPU recommended)
        files: show the location of Notochord models and config files on disk
    """)

def _main():
    # print(sys.argv)
    try:
        if sys.argv[1] == 'server':
            sys.argv = sys.argv[1:]
            run(server)
        if sys.argv[1] == 'homunculus':
            sys.argv = sys.argv[1:]
            run(homunculus) 
        if sys.argv[1] == 'harmonizer':
            sys.argv = sys.argv[1:]
            run(harmonizer)
        if sys.argv[1] == 'improviser':
            sys.argv = sys.argv[1:]
            run(improviser)
        if sys.argv[1] == 'txalaparta':
            sys.argv = sys.argv[1:]
            run(txalaparta)
        if sys.argv[1] == 'morse':
            sys.argv = sys.argv[1:]
            run(morse)
        if sys.argv[1] == 'prompt':
            fire.Fire(prompt, sys.argv[2:])
            return
        if sys.argv[1] == 'train':
            from notochord.train import Resumable
            fire.Fire(Resumable, sys.argv[1:])
        if sys.argv[1] == 'files':
            d = Notochord.user_data_dir()
            print(d)
            if sys.platform=='darwin':
                subprocess.run(('open', d))
            elif sys.platform=='win32':
                os.startfile(d)
        else:
            help()
    except IndexError:
        help()

if __name__=='__main__':
    _main()
