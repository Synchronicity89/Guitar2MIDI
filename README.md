# Guitar2MIDI

Polyphonic guitar audio-to-MIDI conversion using a small DSP preprocessing stage plus selectable transcription backends.

## What It Does

- Reads a WAV file
- Downmixes stereo to mono
- Applies a guitar-focused high-pass plus a steep high-frequency rolloff
- Normalizes the signal to near full scale after the rest of preprocessing
- Runs Omnizart transcription by default
- Can fall back to Basic Pitch transcription
- Removes short or low-velocity ghost notes
- Can transpose the source audio by up to one octave before inference
- Can export the exact fully preprocessed WAV that is fed into transcription
- Prints per-stage timing so preprocessing and inference cost are visible
- Writes a `.mid` file

## Requirements

- Ubuntu in WSL2 is the tested environment
- Python 3.11 is required in the tested WSL setup
- `ffmpeg` and `libsndfile1` installed in Ubuntu

## Setup

From Ubuntu/WSL:

```bash
sudo apt update
sudo apt install -y ffmpeg libsndfile1 python3.11 python3.11-venv
cd /mnt/c/Users/baker/Guitar2MIDI
python3.11 -m venv audio_midi_env_ubuntu311
source audio_midi_env_ubuntu311/bin/activate
python -m pip install --upgrade pip wheel
pip install -r requirements.txt
```

## Usage

Convert one WAV file:

```bash
cd /mnt/c/Users/baker/Guitar2MIDI
source audio_midi_env_ubuntu311/bin/activate
python Convert.py Audio_Samples/andyguitar1.wav
```

The default transcription backend is `omnizart`, because it is currently faster in this setup and does a better job recovering the low notes from the sample guitar recordings.

The default DSP now rolls off high frequencies sharply above about `1200 Hz`, which is just above the `1108.73 Hz` fundamental of the 21st fret on the high E string.

Write to a custom MIDI path:

```bash
python Convert.py Audio_Samples/andyguitar1.wav -o output.mid
```

Adjust transcription sensitivity:

```bash
python Convert.py Audio_Samples/andyguitar1.wav --backend basic-pitch --onset 0.55 --frame 0.30 --min-len 70 --min-vel 25
```

Force the legacy Basic Pitch backend instead of Omnizart:

```bash
python Convert.py Audio_Samples/andyguitar1.wav --backend basic-pitch
```

Transpose the source up by one octave before transcription:

```bash
python Convert.py Audio_Samples/andyguitar1.wav --transpose 12
```

The default transpose backend is `ffmpeg` with the `rubberband` filter because it is much faster than the original in-process `librosa` path. You can still force the older backend for comparison:

```bash
python Convert.py Audio_Samples/andyguitar1.wav --transpose 12 --transpose-backend librosa
```

Save the fully preprocessed audio that will be sent into inference:

```bash
python Convert.py Audio_Samples/andyguitar1.wav --save-preprocessed Audio_Samples/andyguitar1_preprocessed.wav
```

The current defaults are intentionally permissive for low 7-string note recovery:

- `--backend omnizart`
- `--transpose 0`
- `--min-len 20`
- `--min-vel 1`
- `--highcut 1200`

This may admit extra junk notes, which can be tuned down later.

The default normalization target is `--normalize-peak 0.98`, applied after transpose and filtering.

## Sample Files

The repository expects sample WAV files under `Audio_Samples/`. Their contents are ignored by Git.

## Notes

- Python 3.12 is not used here because the `basic-pitch` dependency stack does not resolve cleanly in this setup.
- `setuptools` is pinned below 81 because `resampy` in the `basic-pitch` stack still relies on `pkg_resources`.
- The official MT3 source install is currently blocked on Python 3.11 because its live `flax` dependency now requires Python 3.12.
- `Omnizart` currently works best from its dedicated WSL Python 3.11 environment, and `--backend basic-pitch` remains available when needed.
- Generated MIDI files are ignored by Git by default.