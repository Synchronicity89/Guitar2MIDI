# Guitar2MIDI

Polyphonic guitar audio-to-MIDI conversion using a small DSP preprocessing stage plus Spotify Basic Pitch.

## What It Does

- Reads a WAV file
- Downmixes stereo to mono
- Applies a band-pass filter for guitar-friendly frequencies
- Normalizes the signal before inference
- Runs Basic Pitch transcription
- Removes short or low-velocity ghost notes
- Writes a `.mid` file

## Requirements

- Ubuntu in WSL2 is the tested environment
- Python 3.11 is required for `basic-pitch` in this project setup
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

Write to a custom MIDI path:

```bash
python Convert.py Audio_Samples/andyguitar1.wav -o output.mid
```

Adjust transcription sensitivity:

```bash
python Convert.py Audio_Samples/andyguitar1.wav --onset 0.55 --frame 0.30 --min-len 70 --min-vel 25
```

## Sample Files

The repository expects sample WAV files under `Audio_Samples/`. Their contents are ignored by Git.

## Notes

- Python 3.12 is not used here because the `basic-pitch` dependency stack does not resolve cleanly in this setup.
- `setuptools` is pinned below 81 because `resampy` in the `basic-pitch` stack still relies on `pkg_resources`.
- Generated MIDI files are ignored by Git by default.