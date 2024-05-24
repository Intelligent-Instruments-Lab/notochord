# Notochord ([Documentation](https://intelligent-instruments-lab.github.io/notochord/) | [Paper](https://zenodo.org/record/7088404 "Notochord AIMC 2022 paper") | [Video](https://www.youtube.com/watch?v=mkBKAyudL0A "Notochord AIMC 2022 video"))
 
<div align="middle">
<img alt="Max Ernst, Stratified Rocks, Nature's Gift of Gneiss Lava Iceland Moss 2 kinds of lungwort 2 kinds of ruptures of the perinaeum growths of the heart b) the same thing in a well-polished little box somewhat more expensive, 1920" src="https://user-images.githubusercontent.com/4522484/223191876-251d461a-5bfc-439a-8df0-3841e7c76c4a.jpeg" width="60%" />
</div>

Notochord is a neural network model for MIDI performances. This package contains the training and inference model implemented in pytorch, as well as interactive MIDI processing apps using iipyper. 
<!-- Some further examples involving SuperCollider and TidalCycles can be found in the parent repo under `examples`. -->

## Getting Started

Using your python environment manager of choice (e.g. virtualenv, [conda](https://github.com/conda-forge/miniforge)), make a new environment with a Python version at least 3.10. Then `pip install notochord`.

For developing `notochord`, see our [dev repo](https://github.com/Intelligent-Instruments-Lab/iil-dev.git)

### Install fluidsynth (optional)
[fluidsynth](https://github.com/FluidSynth/fluidsynth) is a General MIDI synthesizer which you can install from the package manager. On macOS:
```
brew install fluidsynth
```
fluidsynth needs a soundfont to run, like this one: https://drive.google.com/file/d/1-cwBWZIYYTxFwzcWFaoGA7Kjx5SEjVAa/view

run fluidsynth in a terminal. For example, `fluidsynth -v -o midi.portname="fluidsynth" -o synth.midi-bank-select=mma ~/'Downloads/soundfonts/Timbres of Heaven (XGM) 4.00(G).sf2'`

## Notochord MIDI Apps

Notochord includes several [iipyper](https://github.com/Intelligent-Instruments-Lab/iipyper.git) apps which can be run in a terminal. They have a clickable text-mode user interface and connect directly to MIDI ports, so you can wire them up to your controllers, DAW, etc.

The `homunculus` provides a text-based graphical interface to manage multiple input, harmonizing or autonomous notochord channels:
```
notochord homunculus
```
You can set the MIDI in and out ports with `--midi-in` and `--midi-out`. If you use a General MIDI synthesizer like fluidsynth, you can add `--send-pc` to also send program change messages.

If you are using fluidsynth as above, try:
```
notochord homunculus --send-pc --midi-out fluidsynth --thru
```

Note: on windows, there are no virtual MIDI ports and no system MIDI loopback, so you may need to attach some MIDI devices or run a loopback driver like [loopMIDI](https://www.tobias-erichsen.de/software/loopmidi.html) before starting the app.

There are also two simpler notochord apps: `improviser` and `harmonizer`. The harmonizer adds extra concurrent notes for each MIDI note you play in. In a terminal, make sure your notochord Python environment is active and run:
```
notochord harmonizer
```
try `notochord harmonizer --help`
to see more options.

Development is now focused on `homunculus`, which is intended to subsume all features of `improviser` and `harmonizer`.


## Python API

See the docs for `Notochord.feed` and `Notochord.query` for the low-level Notochord inference API which can be used from Python code. `notochord/app/simple_harmonizer.py` provides a minimal example of how to build an interactive app.

## OSC server

You can also expose the inference API over Open Sound Control:
```
notochord server
```
this will run notochord and listen continously for OSC messages.

<!-- `examples/notochord/generate-demo.scd` and `examples/notochord/harmonize-demo.scd` are example scripts for interacting with the notochord server from SuperCollider. -->

## Tidal interface

see `notochord/tidalcycles` in [iil-examples](https://github.com/Intelligent-Instruments-Lab/iil-examples.git) repo (updated examples coming soon):

add `Notochord.hs` to your tidal boot file. Probably replace the `tidal <- startTidal` line with something like:
```haskell
:script ~/iil-examples/notochord/tidalcycles/Notochord.hs

let sdOscMap = (superdirtTarget, [superdirtShape])
let oscMap = [sdOscMap,ncOscMap]

tidal <- startStream defaultConfig {cFrameTimespan = 1/240} oscMap
```

In a terminal, start the python server as described above.

In Supercollider, step through `examples/notochord/tidalcycles/tidal-notochord-demo.scd` which will receive from Tidal, talk to the python server, and send MIDI on to a synthesizer. There are two options, either send to fluidsynth to synthesize General MIDI, or specify your own mapping of instruments to channels and send on to your own DAW or synth.

## Train your own Notochord model (GPU recommended)

preprocess the data:
```
python notochord/scripts/lakh_prep.py --data_path /path/to/midi/files --dest_path /path/to/data/storage
```
launch a training job:
```
python notochord/train.py --data_dir /path/to/data/storage --log_dir /path/for/tensorboard/logs --model_dir /path/for/checkpoints --results_dir /path/for/other/logs train
```
progress can be monitored via tensorboard.
