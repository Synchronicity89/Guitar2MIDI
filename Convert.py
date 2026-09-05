#!/usr/bin/env python3
"""
audio_to_midi.py
Hybrid Audio-to-MIDI converter for polyphonic guitar.
Applies pre-filtering DSP -> Basic Pitch NN Inference -> MIDI cleanup.
"""

import argparse
import os
import tempfile
import numpy as np
import soundfile as sf
from scipy import signal
from basic_pitch.inference import predict


LOW_B1_HZ = 61.74
LOW_B1_MIDI_FLOOR_HZ = 55.0
DEFAULT_LOWCUT_HZ = 40.0
DEFAULT_HIGHCUT_HZ = 5000.0
DEFAULT_MAX_FREQUENCY_HZ = 1500.0


def apply_bandpass_filter(audio: np.ndarray, sr: int, lowcut: float = DEFAULT_LOWCUT_HZ, highcut: float = DEFAULT_HIGHCUT_HZ) -> np.ndarray:
    """Applies a 4th-order Butterworth bandpass filter using Second-Order Sections."""
    # Nyquist frequency guard
    nyq = 0.5 * sr
    highcut = min(highcut, nyq - 100.0)
    
    # Keep the cutoff comfortably below a 7-string guitar low B1 fundamental.
    sos = signal.butter(4, [lowcut, highcut], btype='bandpass', fs=sr, output='sos')
    filtered = signal.sosfilt(sos, audio)
    return filtered.astype(np.float32)


def preprocess_audio(input_wav: str, lowcut: float, highcut: float) -> tuple[str, int]:
    """Reads audio, downmixes to mono, bandpasses, normalizes, and saves to a temp WAV."""
    data, sr = sf.read(input_wav, dtype='float32')

    # Downmix stereo to mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Filter out DC offsets, handling noise, and ultrasonic string noise
    filtered = apply_bandpass_filter(data, sr, lowcut, highcut)

    # Normalize peak to -1 dBFS (approx 0.89) to keep inference consistent
    peak = np.max(np.abs(filtered))
    if peak > 0:
        filtered = (filtered / peak) * 0.89

    # Write conditioned signal to a temporary WAV for the inference engine
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(temp_file.name, filtered, sr, subtype='PCM_16')
    return temp_file.name, sr


def cleanup_midi(midi_data, min_velocity: int = 20, min_duration_sec: float = 0.05):
    """Prunes short ghost artifacts and sub-threshold velocity hits from MIDI."""
    for instrument in midi_data.instruments:
        cleaned_notes = []
        for note in instrument.notes:
            duration = note.end - note.start
            if note.velocity >= min_velocity and duration >= min_duration_sec:
                cleaned_notes.append(note)
        instrument.notes = cleaned_notes
    return midi_data


def convert_wav_to_midi(
    input_wav: str,
    output_mid: str,
    onset_thresh: float = 0.60,
    frame_thresh: float = 0.35,
    min_note_len_ms: int = 58,
    min_velocity: int = 20,
    lowcut: float = DEFAULT_LOWCUT_HZ,
    highcut: float = DEFAULT_HIGHCUT_HZ,
    minimum_frequency: float = LOW_B1_MIDI_FLOOR_HZ,
    maximum_frequency: float = DEFAULT_MAX_FREQUENCY_HZ
):
    print(f"[*] Pre-processing audio: {input_wav}")
    temp_wav_path, _ = preprocess_audio(input_wav, lowcut, highcut)

    try:
        print("[*] Running polyphonic neural inference...")
        # Note limits cover 7-string low B1 (61.74 Hz) through the upper guitar range.
        _, midi_data, _ = predict(
            temp_wav_path,
            onset_threshold=onset_thresh,
            frame_threshold=frame_thresh,
            minimum_note_length=min_note_len_ms,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
            melodia_trick=False  # Disabled to preserve independent polyphonic chord decays
        )

        print(f"[*] Filtering ghost notes (vel < {min_velocity}, len < {min_note_len_ms}ms)...")
        cleaned_midi = cleanup_midi(
            midi_data, 
            min_velocity=min_velocity, 
            min_duration_sec=(min_note_len_ms / 1000.0)
        )

        cleaned_midi.write(output_mid)
        print(f"[+] Successfully generated: {output_mid}")

    finally:
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert guitar WAV audio to polyphonic MIDI.")
    parser.add_argument("input", help="Path to input .wav file")
    parser.add_argument("-o", "--output", help="Path to output .mid file", default=None)
    parser.add_argument("--onset", type=float, default=0.60, help="Onset sensitivity threshold [0.1 - 0.9] (default: 0.60)")
    parser.add_argument("--frame", type=float, default=0.35, help="Frame sustain threshold [0.1 - 0.9] (default: 0.35)")
    parser.add_argument("--min-len", type=int, default=58, help="Minimum note length in ms (default: 58)")
    parser.add_argument("--min-vel", type=int, default=20, help="Minimum MIDI velocity threshold (default: 20)")
    parser.add_argument("--lowcut", type=float, default=DEFAULT_LOWCUT_HZ, help="Band-pass low cutoff in Hz (default: 40.0 for 7-string low B support)")
    parser.add_argument("--highcut", type=float, default=DEFAULT_HIGHCUT_HZ, help="Band-pass high cutoff in Hz (default: 5000.0)")
    parser.add_argument("--min-freq", type=float, default=LOW_B1_MIDI_FLOOR_HZ, help="Inference minimum frequency in Hz (default: 55.0 to include 7-string low B1 at 61.74 Hz)")
    parser.add_argument("--max-freq", type=float, default=DEFAULT_MAX_FREQUENCY_HZ, help="Inference maximum frequency in Hz (default: 1500.0)")
    
    args = parser.parse_args()

    out_file = args.output if args.output else os.path.splitext(args.input)[0] + ".mid"
    convert_wav_to_midi(
        args.input,
        out_file,
        onset_thresh=args.onset,
        frame_thresh=args.frame,
        min_note_len_ms=args.min_len,
        min_velocity=args.min_vel,
        lowcut=args.lowcut,
        highcut=args.highcut,
        minimum_frequency=args.min_freq,
        maximum_frequency=args.max_freq
    )