"""
Authors:
  Victor Shepardson
  Intelligent Instruments Lab 2024
"""

from notochord import Notochord

def main(
        midi_file, 
        checkpoint="notochord-latest.ckpt",
        start_tokens=True
        ):
    """
    Run the contents of a MIDI file through a Notochord model and cache the results.

    Args:
        midi_file: path to a MIDI file to prompt with.
        checkpoint: path to Notochord model.
        start_tokens: whether to include send start-of-sequence tokens in prompt
    """
    print(f'{midi_file=} {checkpoint=}')
    noto = Notochord.from_checkpoint(checkpoint)
    noto.eval()
    noto.reset(start=start_tokens)
    _, inst_map = noto.prompt(midi_file)
    print('MIDI channel to instrument:', inst_map)