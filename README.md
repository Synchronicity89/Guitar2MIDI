# Guitar2MIDI

Polyphonic guitar audio-to-MIDI conversion using a small DSP preprocessing stage plus Spotify Basic Pitch.

## What It Does

- Reads a WAV file
- Downmixes stereo to mono
- Applies a band-pass filter for guitar-friendly frequencies
- Normalizes the signal to near full scale after the rest of preprocessing
- Runs Basic Pitch transcription
- Removes short or low-velocity ghost notes
- Can transpose the source audio by up to one octave before inference
- Prefers `ffmpeg` rubberband pitch shifting for faster transposition and falls back to `librosa` when unavailable
- Can export the exact fully preprocessed WAV that is fed into transcription
- Can split a 7-string guitar into a low-strings pass and an upper-strings pass, then merge them into a two-track MIDI file
- Prints per-stage timing so preprocessing and inference cost are visible
- Writes a `.mid` file

## Requirements

- Ubuntu in WSL2 is the tested environment
- Python 3.11 is required for `basic-pitch` in this project setup
- `ffmpeg` and `libsndfile1` installed in Ubuntu
- Current checked environment: Basic Pitch loads through TensorFlow 2.15 and can use the WSL GPU when the matching NVIDIA Python runtime wheels are installed

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

Optional TensorFlow GPU runtime alignment for the tested WSL environment:

```bash
pip install --upgrade --force-reinstall \
	nvidia-cublas-cu12==12.2.5.6 \
	nvidia-cuda-cupti-cu12==12.2.142 \
	nvidia-cuda-nvcc-cu12==12.2.140 \
	nvidia-cuda-nvrtc-cu12==12.2.140 \
	nvidia-cuda-runtime-cu12==12.2.140 \
	nvidia-cudnn-cu12==8.9.4.25 \
	nvidia-cufft-cu12==11.0.8.103 \
	nvidia-curand-cu12==10.3.3.141 \
	nvidia-cusolver-cu12==11.5.2.141 \
	nvidia-cusparse-cu12==12.1.2.141 \
	nvidia-nccl-cu12==2.16.5 \
	nvidia-nvjitlink-cu12==12.2.140
```

`Convert.py` automatically adds the installed NVIDIA wheel `lib` directories to `LD_LIBRARY_PATH` before importing Basic Pitch, so no manual shell export is needed once those packages are aligned.

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

Transpose the source up by one octave before transcription:

```bash
python Convert.py Audio_Samples/andyguitar1.wav --transpose 12
```

Save the fully preprocessed audio that will be sent into inference:

```bash
python Convert.py Audio_Samples/andyguitar1.wav --save-preprocessed Audio_Samples/andyguitar1_preprocessed.wav
```

Split a 7-string guitar into low two strings plus upper five strings, and produce a combined two-track MIDI file:

```bash
python Convert.py Audio_Samples/andyguitar1.wav --split-seven-string -o Audio_Samples/andyguitar1_split.mid
```

Override the split filter edges when needed:

```bash
python Convert.py Audio_Samples/andyguitar1.wav \
	--split-seven-string \
	--low-strings-lowcut 55 \
	--low-strings-highcut 220 \
	--high-strings-lowcut 95 \
	--high-strings-highcut 5000 \
	--low-strings-max-freq 350 \
	--high-strings-min-freq 98
```

The current defaults are intentionally permissive for low 7-string note recovery:

- `--transpose 0`
- `--min-len 20`
- `--min-vel 1`

This may admit extra junk notes, which can be tuned down later.

The default normalization target is `--normalize-peak 0.98`, applied after transpose and filtering.

## 7-String Split Recommendations

Recommended starting point for a standard 7-string tuning `B1 E2 A2 D3 G3 B3 E4`:

- Low two strings audio pass: `55 Hz` to `220 Hz`
- Upper five strings audio pass: `95 Hz` to `5000 Hz`
- Low two strings MIDI ceiling: `350 Hz`
- Upper five strings MIDI floor: `98 Hz`

Why these numbers are a reasonable starting point:

- The open 7th string low B is `61.74 Hz`, so a `55 Hz` low cutoff still keeps that fundamental.
- The open 6th string E is `82.41 Hz`, so the low-strings pass comfortably includes both lowest open strings.
- `220 Hz` is `A3`, which is the 17th fret on the low E string and about the 22nd fret on the low B string. Below that point, notes are still strongly associated with the bass side of the instrument.
- The upper five strings start at open `A2 = 110 Hz`, so a `95 Hz` low cutoff keeps the A string while still reducing some low-B leakage.
- The overlap from roughly `95 Hz` to `220 Hz` is intentional because a practical Butterworth band-pass does not separate strings sharply, and the pitch ranges of fretted notes already overlap heavily in that zone.
- `350 Hz` as the low-strings MIDI ceiling keeps the bass pass focused near the low-string register while still allowing notes above `A3` to survive if the filter leaves enough fundamental energy.
- `98 Hz` as the upper-strings MIDI floor is `G2`, slightly below the open A string, which gives the upper pass a little safety margin instead of cutting right at `110 Hz`.

This split is about improving left-hand versus right-hand style separation in the MIDI, not perfectly identifying the original string. Once the low E string is fretted up to `A2` and above, those notes overlap the open and fretted range of the A string, so no frequency-only split can fully recover the played string identity.

## Sample Files

The repository expects sample WAV files under `Audio_Samples/`. Their contents are ignored by Git.

## Notes

- Python 3.12 is not used here because the `basic-pitch` dependency stack does not resolve cleanly in this setup.
- `setuptools` is pinned below 81 because `resampy` in the `basic-pitch` stack still relies on `pkg_resources`.
- Generated MIDI files are ignored by Git by default.
- In the current WSL environment, Basic Pitch is running on TensorFlow CPU. GPU use will require fixing the missing TensorFlow CUDA/cuDNN runtime libraries first.
- In the current tested WSL environment, Basic Pitch now detects `TensorFlow GPU (1 device)` after aligning the NVIDIA Python runtime wheels to TensorFlow 2.15 and letting `Convert.py` expose their `lib` directories automatically.
- This shared environment may now conflict with CUDA-pinned PyTorch packages. If you need Torch CUDA and Basic Pitch TensorFlow GPU at the same time, use separate virtual environments.