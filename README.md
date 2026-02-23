[![DOI](https://zenodo.org/badge/722231412.svg)](https://doi.org/10.5281/zenodo.16116595)

# Notochord ([Documentation](https://intelligent-instruments-lab.github.io/notochord/) | [Paper](https://zenodo.org/record/7088404 "Notochord AIMC 2022 paper") | [Video](https://www.youtube.com/watch?v=mkBKAyudL0A "Notochord AIMC 2022 video"))
 
<div align="middle">
<img alt="Max Ernst, Stratified Rocks, Nature's Gift of Gneiss Lava Iceland Moss 2 kinds of lungwort 2 kinds of ruptures of the perinaeum growths of the heart b) the same thing in a well-polished little box somewhat more expensive, 1920" src="https://user-images.githubusercontent.com/4522484/223191876-251d461a-5bfc-439a-8df0-3841e7c76c4a.jpeg" width="60%" />
</div>

Notochord is a neural network model for MIDI performances. This package contains the training and inference model implemented in pytorch, as well as interactive MIDI processing apps using iipyper. 

[API Reference](https://intelligent-instruments-lab.github.io/notochord/reference/notochord/model/#notochord.model.Notochord)

<!-- Some further examples involving SuperCollider and TidalCycles can be found in the parent repo under `examples`. -->

## Getting Started

We recommend using [uv](https://docs.astral.sh/uv/) to run notochord, install it as a tool, or create a project incorporating it.

### try out the homunculus MIDI app right now
```bash
uvx notochord homunculus
```

### persistently install notochord
```bash
uv tool install notochord
notochord --help
```

### create a Python project using notochord
```bash
uv init my-project
cd my-project
uv add notochord
.venv/bin/python -m notochord --help
```

### Install fluidsynth (optional)
[fluidsynth](https://github.com/FluidSynth/fluidsynth) is a General MIDI synthesizer which you can install from the package manager. On macOS:
```
brew install fluidsynth
```
fluidsynth needs a SoundFont to run, like [this one (Google Drive link)](https://drive.google.com/file/d/1-cwBWZIYYTxFwzcWFaoGA7Kjx5SEjVAa/view).

You can run fluidsynth in a terminal, supplying the path to your SoundFont. For example, `fluidsynth -v -o midi.portname="fluidsynth" -o synth.midi-bank-select=mma ~/'Downloads/soundfonts/Timbres\ of\ Heaven\ (XGM)\ 4.00(G).sf2'`

### without uv

Using your python environment manager of choice (e.g. virtualenv, conda), make a new environment with a Python version between 3.10-3.13. Then `pip install notochord`.

### developing

For developing `notochord`, see our [dev repo](https://github.com/Intelligent-Instruments-Lab/iil-dev.git)

## Notochord Homunculus

Notochord includes several [iipyper](https://github.com/Intelligent-Instruments-Lab/iipyper.git) apps which can be run in a terminal. They have a clickable text-mode user interface and connect directly to MIDI ports, so you can wire them up to your controllers, DAW, etc.

The `homunculus` provides a text-based graphical interface to manage multiple input, harmonizing or autonomous notochord voices:
```
notochord homunculus
```
You can set the MIDI in and out ports with `--midi-in` and `--midi-out`. If you use a General MIDI synthesizer like fluidsynth, you can add `--send-pc` to also send program change messages. More information in the [Homunculus docs](https://intelligent-instruments-lab.github.io/notochord/reference/notochord/app/homunculus/#notochord.app.homunculus.main), or run `notochord homunculus --help`

If you are using fluidsynth as above, try:
```
notochord homunculus --send-pc --midi-out fluidsynth --thru
```

Note: on windows, there are no virtual MIDI ports and no system MIDI loopback, so you may need to attach some MIDI devices or run a loopback driver like [loopMIDI](https://www.tobias-erichsen.de/software/loopmidi.html) before starting the app.

If you pass homunculus a MIDI file using the `--midi-prompt` flag, it will play as if continuing after the end of that file.

Adding the `--punch-in` flag will automatically switch voices to input mode when MIDI is received and back to auto after some time passes.

<!-- There are also two simpler notochord apps: `improviser` and `harmonizer`. The harmonizer adds extra concurrent notes for each MIDI note you play in. In a terminal, make sure your notochord Python environment is active and run:
```
notochord harmonizer
```
try `notochord harmonizer --help`
to see more options.

Development is now focused on `homunculus`, which is intended to subsume all features of `improviser` and `harmonizer`. -->


## Python API

See the docs for [`Notochord.reset`](https://intelligent-instruments-lab.github.io/notochord/reference/notochord/model/#notochord.model.Notochord.reset), [`Notochord.feed`](https://intelligent-instruments-lab.github.io/notochord/reference/notochord/model/#notochord.model.Notochord.feed) and [`Notochord.sample`](https://intelligent-instruments-lab.github.io/notochord/reference/notochord/model/#notochord.model.Notochord.sample) (or older [`Notochord.query`](https://intelligent-instruments-lab.github.io/notochord/reference/notochord/model/#notochord.model.Notochord.sample)) for the low-level Notochord inference API which can be used from Python code. `notochord/app/simple_harmonizer.py` provides a minimal example of how to build an interactive app.

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

Select the `train` option when installing, e.g. `uv tool install notochord[train]`. 

### preprocess data
```bash
notochord prep --data_path /path/to/midi/files \
--dest_path /path/to/data/storage
```

This will process all MIDI files under a directory, converting them to torch tensors which notochord can train on. The built-in preprocessing assumes a dataset of General MIDI files, like the [Lakh MIDI dataset](https://colinraffel.com/projects/lmd/). If you use your own MIDI files, you likely want to label parts using MIDI program change messages (different instruments for each channel). Only ProgramChange, NoteOn and NoteOff events are processed by notochord.

### launch a training job
```bash
notochord train my-model --data_dir /path/to/data/storage \
--log_dir /path/for/tensorboard/logs \
--model_dir /path/for/checkpoints \
--results_dir /path/for/other/logs
```

The above will train a new notochord model from scratch. By adding the `--model` argument, you can set model hyperparameters (as documented in [`Notochord.__init__`](https://intelligent-instruments-lab.github.io/notochord/reference/notochord/model/#notochord.model.Notochord.__init__)). For example, `--model '{rnn_hidden:512, rnn_layers:1, mlp_layers:1}'` would train a smaller model than the default.

If using your own dataset, you may want to turn off data augmentation options with `--aug-speed 0`, `--aug-transpose 0` or `--aug-remap False`.

Training progress can be monitored via tensorboard:
```bash
tensorboard --logdir /path/for/tensorboard/logs
```

The most important value is `valid/loss`. As long as it decreases, the model should continue to improve. Training will continue until the job is killed with ctrl+C.

### resuming and fine-tuning

You can resume training from an existing checkpoint by adding `--resume --checkpoint /path/to/model/checkpoint`. It is also possible to initialize training from a checkpoint without carrying over optimizer states etc from the original training, by leaving out the `--resume` flag. For fine-tuning on small datasets, consider adding `--freeze-rnn`. Training options are documented under [`Trainer.__init__`](https://intelligent-instruments-lab.github.io/notochord/reference/notochord/train/#notochord.train.Trainer.__init__)

To fine the latest base notochord model, you can run `notochord homunculus` at least once to download it, then use `notochord files` to find the downloaded model.


### use your custom model

You can load a custom model in the Python API using `Notchord.from_checkpoint` or use it with homunculus using e.g. `notochord homunculus --checkpoint /path/for/checkpoints/my-model/XXX.ckpt`
