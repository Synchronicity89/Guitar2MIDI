#!/usr/bin/env python3
"""
audio_to_midi.py
Hybrid Audio-to-MIDI converter for polyphonic guitar.
Applies pre-filtering DSP -> Basic Pitch NN Inference -> MIDI cleanup.
"""

import argparse
import math
import os
import shutil
import subprocess
import tempfile
import time
import librosa
import numpy as np
import soundfile as sf
from scipy import signal
from basic_pitch.inference import predict


LOW_B1_HZ = 61.74
LOW_B1_MIDI_FLOOR_HZ = 55.0
DEFAULT_LOWCUT_HZ = 40.0
DEFAULT_HIGHCUT_HZ = 5000.0
DEFAULT_MAX_FREQUENCY_HZ = 1500.0
DEFAULT_NORMALIZE_PEAK = 0.98
MIN_TRANSPOSE_SEMITONES = -12
MAX_TRANSPOSE_SEMITONES = 12
DEFAULT_MIN_NOTE_LEN_MS = 20
DEFAULT_MIN_VELOCITY = 1
DEFAULT_TRANSPOSE_BACKEND = "ffmpeg"
DEFAULT_LOW_B_BOOST_DB = 8.0
DEFAULT_LOW_B_BOOST_Q = 0.8
DEFAULT_HARMONIC_ATTENUATION_DB = 6.0
DEFAULT_HARMONIC_NOTCH_Q = 5.0
DEFAULT_HARMONIC_COUNT = 3


def apply_bandpass_filter(audio: np.ndarray, sr: int, lowcut: float = DEFAULT_LOWCUT_HZ, highcut: float = DEFAULT_HIGHCUT_HZ) -> np.ndarray:
    """Applies a 4th-order Butterworth bandpass filter using Second-Order Sections."""
    # Nyquist frequency guard
    nyq = 0.5 * sr
    highcut = min(highcut, nyq - 100.0)
    
    # Keep the cutoff comfortably below a 7-string guitar low B1 fundamental.
    sos = signal.butter(4, [lowcut, highcut], btype='bandpass', fs=sr, output='sos')
    filtered = signal.sosfilt(sos, audio)
    return filtered.astype(np.float32)


def apply_peaking_eq(audio: np.ndarray, sr: int, center_hz: float, gain_db: float, q: float) -> np.ndarray:
    """Applies a biquad peaking EQ around a target center frequency."""
    if center_hz <= 0 or center_hz >= sr * 0.5:
        return audio.astype(np.float32)

    amplitude = 10.0 ** (gain_db / 40.0)
    omega = 2.0 * math.pi * center_hz / sr
    alpha = math.sin(omega) / (2.0 * q)
    cos_omega = math.cos(omega)

    b0 = 1.0 + alpha * amplitude
    b1 = -2.0 * cos_omega
    b2 = 1.0 - alpha * amplitude
    a0 = 1.0 + alpha / amplitude
    a1 = -2.0 * cos_omega
    a2 = 1.0 - alpha / amplitude

    b = np.array([b0, b1, b2], dtype=np.float64) / a0
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    return signal.lfilter(b, a, audio).astype(np.float32)


def apply_harmonic_notches(
    audio: np.ndarray,
    sr: int,
    fundamental_hz: float,
    harmonic_count: int,
    attenuation_db: float,
    q: float,
) -> np.ndarray:
    """Attenuates a few harmonics above a target fundamental using narrow notch filters."""
    if attenuation_db <= 0 or harmonic_count <= 0:
        return audio.astype(np.float32)

    output = audio.astype(np.float32)
    # Apply the requested notch more than once to deepen attenuation without widening it too much.
    passes = max(1, int(round(attenuation_db / 3.0)))
    nyquist = sr * 0.5

    for harmonic in range(2, harmonic_count + 2):
        center_hz = fundamental_hz * harmonic
        if center_hz >= nyquist:
            break
        b, a = signal.iirnotch(center_hz, q, fs=sr)
        for _ in range(passes):
            output = signal.lfilter(b, a, output).astype(np.float32)

    return output


def enhance_low_b_fundamental(
    audio: np.ndarray,
    sr: int,
    low_b_boost_db: float,
    low_b_boost_q: float,
    harmonic_attenuation_db: float,
    harmonic_notch_q: float,
    harmonic_count: int,
) -> np.ndarray:
    """Boosts the low-B fundamental region and suppresses early harmonics that can dominate pitch estimation."""
    boosted = apply_peaking_eq(audio, sr, LOW_B1_HZ, low_b_boost_db, low_b_boost_q)
    return apply_harmonic_notches(
        boosted,
        sr,
        LOW_B1_HZ,
        harmonic_count,
        harmonic_attenuation_db,
        harmonic_notch_q,
    )


def shift_audio_key_with_librosa(audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Shifts pitch without changing duration."""
    if semitones == 0:
        return audio
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)


def shift_audio_key_with_ffmpeg(input_wav: str, semitones: float) -> tuple[np.ndarray, int]:
    """Uses ffmpeg rubberband to shift pitch without changing duration."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for transpose-backend=ffmpeg")

    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_file.close()
    try:
        ratio = 2.0 ** (semitones / 12.0)
        command = [
            "ffmpeg",
            "-y",
            "-i",
            input_wav,
            "-vn",
            "-af",
            f"rubberband=pitch={ratio}",
            temp_file.name,
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        shifted, sr = sf.read(temp_file.name, dtype="float32")
        if shifted.ndim > 1:
            shifted = np.mean(shifted, axis=1)
        return shifted, sr
    finally:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)


def shift_audio_key(input_wav: str, audio: np.ndarray, sr: int, semitones: float, backend: str) -> tuple[np.ndarray, int]:
    """Shifts pitch without changing duration using the requested backend."""
    if semitones == 0:
        return audio, sr
    if backend == "ffmpeg":
        return shift_audio_key_with_ffmpeg(input_wav, semitones)
    if backend == "librosa":
        return shift_audio_key_with_librosa(audio, sr, semitones), sr
    raise ValueError(f"Unsupported transpose backend: {backend}")


def normalize_audio(audio: np.ndarray, target_peak: float = DEFAULT_NORMALIZE_PEAK) -> np.ndarray:
    """Raises signal level close to full scale after all other processing."""
    peak = np.max(np.abs(audio))
    if peak <= 0:
        return audio
    return (audio / peak * target_peak).astype(np.float32)


def preprocess_audio(
    input_wav: str,
    lowcut: float,
    highcut: float,
    transpose_semitones: float,
    transpose_backend: str,
    normalize_peak: float,
    low_b_boost_db: float,
    low_b_boost_q: float,
    harmonic_attenuation_db: float,
    harmonic_notch_q: float,
    harmonic_count: int,
    save_preprocessed_path: str | None = None,
) -> tuple[str, int, dict[str, float]]:
    """Reads audio, downmixes, optionally transposes, bandpasses, normalizes, and saves to a temp WAV."""
    timings: dict[str, float] = {}

    stage_start = time.perf_counter()
    data, sr = sf.read(input_wav, dtype='float32')
    timings["read_audio_sec"] = time.perf_counter() - stage_start

    # Downmix stereo to mono
    stage_start = time.perf_counter()
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    timings["downmix_sec"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    shifted, sr = shift_audio_key(input_wav, data, sr, transpose_semitones, transpose_backend)
    timings["transpose_sec"] = time.perf_counter() - stage_start

    # Filter out DC offsets, handling noise, and ultrasonic string noise
    stage_start = time.perf_counter()
    filtered = apply_bandpass_filter(shifted, sr, lowcut, highcut)
    timings["filter_sec"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    enhanced = enhance_low_b_fundamental(
        filtered,
        sr,
        low_b_boost_db,
        low_b_boost_q,
        harmonic_attenuation_db,
        harmonic_notch_q,
        harmonic_count,
    )
    timings["low_b_enhance_sec"] = time.perf_counter() - stage_start

    # Normalize after the rest of preprocessing so the exported and inferred signal match.
    stage_start = time.perf_counter()
    normalized = normalize_audio(enhanced, target_peak=normalize_peak)
    timings["normalize_sec"] = time.perf_counter() - stage_start

    if save_preprocessed_path:
        stage_start = time.perf_counter()
        sf.write(save_preprocessed_path, normalized, sr, subtype='PCM_16')
        timings["save_preprocessed_sec"] = time.perf_counter() - stage_start

    # Write conditioned signal to a temporary WAV for the inference engine
    stage_start = time.perf_counter()
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(temp_file.name, normalized, sr, subtype='PCM_16')
    timings["write_temp_wav_sec"] = time.perf_counter() - stage_start
    return temp_file.name, sr, timings


def print_timings(timings: dict[str, float]) -> None:
    """Prints per-stage elapsed time in seconds."""
    print("[*] Timing summary:")
    for stage, elapsed in timings.items():
        print(f"    {stage}: {elapsed:.3f}s")


def cleanup_midi(midi_data, min_velocity: int = DEFAULT_MIN_VELOCITY, min_duration_sec: float = DEFAULT_MIN_NOTE_LEN_MS / 1000.0):
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
    min_note_len_ms: int = DEFAULT_MIN_NOTE_LEN_MS,
    min_velocity: int = DEFAULT_MIN_VELOCITY,
    lowcut: float = DEFAULT_LOWCUT_HZ,
    highcut: float = DEFAULT_HIGHCUT_HZ,
    minimum_frequency: float = LOW_B1_MIDI_FLOOR_HZ,
    maximum_frequency: float = DEFAULT_MAX_FREQUENCY_HZ,
    transpose_semitones: float = 0.0,
    transpose_backend: str = DEFAULT_TRANSPOSE_BACKEND,
    normalize_peak: float = DEFAULT_NORMALIZE_PEAK,
    low_b_boost_db: float = DEFAULT_LOW_B_BOOST_DB,
    low_b_boost_q: float = DEFAULT_LOW_B_BOOST_Q,
    harmonic_attenuation_db: float = DEFAULT_HARMONIC_ATTENUATION_DB,
    harmonic_notch_q: float = DEFAULT_HARMONIC_NOTCH_Q,
    harmonic_count: int = DEFAULT_HARMONIC_COUNT,
    save_preprocessed_path: str | None = None,
):
    total_start = time.perf_counter()
    print(f"[*] Pre-processing audio: {input_wav}")
    if transpose_semitones:
        print(f"[*] Transposing audio by {transpose_semitones:+.1f} semitones before inference with {transpose_backend}...")
    if save_preprocessed_path:
        print(f"[*] Saving fully preprocessed audio to: {save_preprocessed_path}")
    temp_wav_path, _, timings = preprocess_audio(
        input_wav,
        lowcut,
        highcut,
        transpose_semitones,
        transpose_backend,
        normalize_peak,
        low_b_boost_db,
        low_b_boost_q,
        harmonic_attenuation_db,
        harmonic_notch_q,
        harmonic_count,
        save_preprocessed_path,
    )

    try:
        print("[*] Running polyphonic neural inference...")
        # Note limits cover 7-string low B1 (61.74 Hz) through the upper guitar range.
        stage_start = time.perf_counter()
        _, midi_data, _ = predict(
            temp_wav_path,
            onset_threshold=onset_thresh,
            frame_threshold=frame_thresh,
            minimum_note_length=min_note_len_ms,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
            melodia_trick=False  # Disabled to preserve independent polyphonic chord decays
        )
        timings["inference_sec"] = time.perf_counter() - stage_start

        print(f"[*] Filtering ghost notes (vel < {min_velocity}, len < {min_note_len_ms}ms)...")
        stage_start = time.perf_counter()
        cleaned_midi = cleanup_midi(
            midi_data, 
            min_velocity=min_velocity, 
            min_duration_sec=(min_note_len_ms / 1000.0)
        )
        timings["cleanup_sec"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        cleaned_midi.write(output_mid)
        timings["write_midi_sec"] = time.perf_counter() - stage_start
        timings["total_sec"] = time.perf_counter() - total_start
        print_timings(timings)
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
    parser.add_argument("--min-len", type=int, default=DEFAULT_MIN_NOTE_LEN_MS, help="Minimum note length in ms (default: 20)")
    parser.add_argument("--min-vel", type=int, default=DEFAULT_MIN_VELOCITY, help="Minimum MIDI velocity threshold (default: 1)")
    parser.add_argument("--lowcut", type=float, default=DEFAULT_LOWCUT_HZ, help="Band-pass low cutoff in Hz (default: 40.0 for 7-string low B support)")
    parser.add_argument("--highcut", type=float, default=DEFAULT_HIGHCUT_HZ, help="Band-pass high cutoff in Hz (default: 5000.0)")
    parser.add_argument("--min-freq", type=float, default=LOW_B1_MIDI_FLOOR_HZ, help="Inference minimum frequency in Hz (default: 55.0 to include 7-string low B1 at 61.74 Hz)")
    parser.add_argument("--max-freq", type=float, default=DEFAULT_MAX_FREQUENCY_HZ, help="Inference maximum frequency in Hz (default: 1500.0)")
    parser.add_argument("--transpose", type=float, default=0.0, help="Pitch-shift the source audio before inference in semitones, from -12 to +12 (default: 0)")
    parser.add_argument("--transpose-backend", choices=["ffmpeg", "librosa"], default=DEFAULT_TRANSPOSE_BACKEND, help="Backend used for pitch shifting before inference (default: ffmpeg)")
    parser.add_argument("--normalize-peak", type=float, default=DEFAULT_NORMALIZE_PEAK, help="Peak normalization target applied after preprocessing, from 0 to 1 (default: 0.98)")
    parser.add_argument("--low-b-boost-db", type=float, default=DEFAULT_LOW_B_BOOST_DB, help="Peaking EQ boost in dB around low B1 after filtering (default: 8.0)")
    parser.add_argument("--low-b-boost-q", type=float, default=DEFAULT_LOW_B_BOOST_Q, help="Q for the low B1 peaking EQ boost (default: 0.8)")
    parser.add_argument("--harmonic-attenuation-db", type=float, default=DEFAULT_HARMONIC_ATTENUATION_DB, help="Approximate attenuation depth in dB for early low-B harmonics after filtering (default: 6.0)")
    parser.add_argument("--harmonic-notch-q", type=float, default=DEFAULT_HARMONIC_NOTCH_Q, help="Q for harmonic notch filters above low B1 (default: 5.0)")
    parser.add_argument("--harmonic-count", type=int, default=DEFAULT_HARMONIC_COUNT, help="How many low-B harmonics above the fundamental to attenuate (default: 3)")
    parser.add_argument("--save-preprocessed", help="Optional path to save the fully preprocessed WAV that will be fed into transcription", default=None)
    
    args = parser.parse_args()

    if not MIN_TRANSPOSE_SEMITONES <= args.transpose <= MAX_TRANSPOSE_SEMITONES:
        parser.error("--transpose must be between -12 and +12 semitones")
    if not 0 < args.normalize_peak <= 1:
        parser.error("--normalize-peak must be greater than 0 and at most 1")
    if args.low_b_boost_q <= 0:
        parser.error("--low-b-boost-q must be greater than 0")
    if args.harmonic_notch_q <= 0:
        parser.error("--harmonic-notch-q must be greater than 0")
    if args.harmonic_count < 0:
        parser.error("--harmonic-count must be 0 or greater")

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
        maximum_frequency=args.max_freq,
        transpose_semitones=args.transpose,
        transpose_backend=args.transpose_backend,
        normalize_peak=args.normalize_peak,
        low_b_boost_db=args.low_b_boost_db,
        low_b_boost_q=args.low_b_boost_q,
        harmonic_attenuation_db=args.harmonic_attenuation_db,
        harmonic_notch_q=args.harmonic_notch_q,
        harmonic_count=args.harmonic_count,
        save_preprocessed_path=args.save_preprocessed,
    )