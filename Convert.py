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
import site
import subprocess
import tempfile
import time

import librosa
import numpy as np
import pretty_midi
import soundfile as sf
from scipy import signal


def configure_linux_nvidia_runtime_library_path() -> None:
    """Makes TensorFlow-visible NVIDIA wheel libraries discoverable in Linux venvs before import."""
    if os.name != "posix":
        return

    lib_dirs: list[str] = []
    for site_packages_dir in site.getsitepackages():
        nvidia_root = os.path.join(site_packages_dir, "nvidia")
        if not os.path.isdir(nvidia_root):
            continue
        for dirpath, _, filenames in os.walk(nvidia_root):
            if os.path.basename(dirpath) == "lib" and any(filename.endswith(".so") for filename in filenames):
                lib_dirs.append(dirpath)

    if not lib_dirs:
        return

    current_path_entries = [entry for entry in os.environ.get("LD_LIBRARY_PATH", "").split(":") if entry]
    merged_entries: list[str] = []
    for entry in lib_dirs + current_path_entries:
        if entry not in merged_entries:
            merged_entries.append(entry)
    os.environ["LD_LIBRARY_PATH"] = ":".join(merged_entries)


configure_linux_nvidia_runtime_library_path()

import basic_pitch.inference as basic_pitch_inference
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
DEFAULT_LOW_B_BOOST_DB = 8.0
DEFAULT_LOW_B_BOOST_Q = 0.8
DEFAULT_HARMONIC_ATTENUATION_DB = 6.0
DEFAULT_HARMONIC_NOTCH_Q = 5.0
DEFAULT_HARMONIC_COUNT = 3
DEFAULT_SPLIT_LOW_STRINGS_LOWCUT_HZ = 55.0
DEFAULT_SPLIT_LOW_STRINGS_HIGHCUT_HZ = 220.0
DEFAULT_SPLIT_HIGH_STRINGS_LOWCUT_HZ = 95.0
DEFAULT_SPLIT_HIGH_STRINGS_HIGHCUT_HZ = 5000.0
DEFAULT_SPLIT_LOW_STRINGS_MAX_FREQUENCY_HZ = 350.0
DEFAULT_SPLIT_HIGH_STRINGS_MIN_FREQUENCY_HZ = 98.0


def describe_basic_pitch_runtime() -> str:
    """Returns the Basic Pitch runtime that will be used on this machine."""
    if getattr(basic_pitch_inference, "TF_PRESENT", False):
        try:
            import tensorflow as tf

            gpu_devices = tf.config.list_physical_devices("GPU")
        except Exception as exc:
            return f"TensorFlow (GPU check failed: {exc!r})"
        if gpu_devices:
            return f"TensorFlow GPU ({len(gpu_devices)} device(s))"
        return "TensorFlow CPU"

    if getattr(basic_pitch_inference, "ONNX_PRESENT", False):
        return "ONNX Runtime CPU"
    if getattr(basic_pitch_inference, "TFLITE_PRESENT", False):
        return "TensorFlow Lite CPU"
    return "Unknown runtime"


def run_ffmpeg_rubberband_pitch_shift(audio: np.ndarray, sr: int, semitones: float) -> np.ndarray | None:
    """Uses ffmpeg's rubberband filter when available because it is much faster than librosa on long files."""
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return None

    pitch_ratio = 2.0 ** (semitones / 12.0)
    input_temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    output_temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    input_temp.close()
    output_temp.close()

    try:
        sf.write(input_temp.name, audio.astype(np.float32), sr, subtype="PCM_16")
        command = [
            ffmpeg_path,
            "-y",
            "-loglevel",
            "error",
            "-i",
            input_temp.name,
            "-af",
            f"rubberband=pitch={pitch_ratio:.10f}",
            "-ar",
            str(sr),
            "-ac",
            "1",
            output_temp.name,
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        shifted_audio, _ = sf.read(output_temp.name, dtype="float32")
        if shifted_audio.ndim > 1:
            shifted_audio = np.mean(shifted_audio, axis=1)
        return shifted_audio.astype(np.float32)
    except (subprocess.CalledProcessError, OSError, RuntimeError, sf.LibsndfileError):
        return None
    finally:
        for temp_path in (input_temp.name, output_temp.name):
            if os.path.exists(temp_path):
                os.remove(temp_path)


def shift_audio_key(audio: np.ndarray, sr: int, semitones: float) -> tuple[np.ndarray, str]:
    """Shifts pitch without changing duration."""
    if semitones == 0:
        return audio.astype(np.float32), "none"

    ffmpeg_shifted = run_ffmpeg_rubberband_pitch_shift(audio, sr, semitones)
    if ffmpeg_shifted is not None:
        return ffmpeg_shifted, "ffmpeg-rubberband"

    shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)
    return shifted.astype(np.float32), "librosa"


def load_source_audio(input_wav: str, transpose_semitones: float) -> tuple[np.ndarray, int, dict[str, float], str]:
    """Reads the source WAV once so split passes can reuse the same mono, transposed signal."""
    timings: dict[str, float] = {}

    stage_start = time.perf_counter()
    data, sr = sf.read(input_wav, dtype="float32")
    timings["read_audio_sec"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data = data.astype(np.float32, copy=False)
    timings["downmix_sec"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    shifted, transpose_backend = shift_audio_key(data, sr, transpose_semitones)
    timings["transpose_sec"] = time.perf_counter() - stage_start
    return shifted.astype(np.float32), sr, timings, transpose_backend


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


def normalize_audio(audio: np.ndarray, target_peak: float = DEFAULT_NORMALIZE_PEAK) -> np.ndarray:
    """Raises signal level close to full scale after all other processing."""
    peak = np.max(np.abs(audio))
    if peak <= 0:
        return audio
    return (audio / peak * target_peak).astype(np.float32)


def preprocess_audio_array(
    audio: np.ndarray,
    sr: int,
    lowcut: float,
    highcut: float,
    normalize_peak: float,
    low_b_boost_db: float,
    low_b_boost_q: float,
    harmonic_attenuation_db: float,
    harmonic_notch_q: float,
    harmonic_count: int,
    enhance_low_b: bool = True,
    save_preprocessed_path: str | None = None,
) -> tuple[str, dict[str, float]]:
    """Bandpasses, optionally enhances the low-B region, normalizes, and saves to a temp WAV."""
    timings: dict[str, float] = {}

    # Filter out DC offsets, handling noise, and ultrasonic string noise
    stage_start = time.perf_counter()
    filtered = apply_bandpass_filter(audio, sr, lowcut, highcut)
    timings["filter_sec"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    if enhance_low_b:
        enhanced = enhance_low_b_fundamental(
            filtered,
            sr,
            low_b_boost_db,
            low_b_boost_q,
            harmonic_attenuation_db,
            harmonic_notch_q,
            harmonic_count,
        )
    else:
        enhanced = filtered.astype(np.float32)
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
    return temp_file.name, timings


def print_timings(timings: dict[str, float]) -> None:
    """Prints per-stage elapsed time in seconds."""
    print("[*] Timing summary:")
    for stage, elapsed in timings.items():
        print(f"    {stage}: {elapsed:.3f}s")


def prefix_timings(prefix: str, timings: dict[str, float]) -> dict[str, float]:
    """Prefixes timing keys so split passes can be reported together."""
    return {f"{prefix}_{stage}": elapsed for stage, elapsed in timings.items()}


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


def combine_midi_tracks(low_midi_data: pretty_midi.PrettyMIDI, high_midi_data: pretty_midi.PrettyMIDI, output_mid: str) -> None:
    """Flattens each transcription into its own track inside a combined two-track MIDI file."""
    combined = pretty_midi.PrettyMIDI(initial_tempo=120)
    combined.instruments.append(flatten_midi_to_track(low_midi_data, "Low strings"))
    combined.instruments.append(flatten_midi_to_track(high_midi_data, "Upper strings"))
    combined.write(output_mid)


def flatten_midi_to_track(midi_data: pretty_midi.PrettyMIDI, track_name: str) -> pretty_midi.Instrument:
    """Copies all notes and controller data into a single named track."""
    program = midi_data.instruments[0].program if midi_data.instruments else 0
    track = pretty_midi.Instrument(program=program, name=track_name)

    for instrument in midi_data.instruments:
        for note in instrument.notes:
            track.notes.append(
                pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note.start,
                    end=note.end,
                )
            )
        for bend in instrument.pitch_bends:
            track.pitch_bends.append(pretty_midi.PitchBend(pitch=bend.pitch, time=bend.time))
        for control_change in instrument.control_changes:
            track.control_changes.append(
                pretty_midi.ControlChange(
                    number=control_change.number,
                    value=control_change.value,
                    time=control_change.time,
                )
            )

    track.notes.sort(key=lambda note: (note.start, note.pitch, note.end))
    return track


def derive_split_output_paths(
    input_wav: str,
    combined_output: str | None,
    low_output: str | None,
    high_output: str | None,
) -> tuple[str, str, str]:
    """Builds deterministic output paths for split low/high and combined MIDI files."""
    if combined_output:
        combined_mid = combined_output
        combined_stem, _ = os.path.splitext(combined_output)
    else:
        input_stem, _ = os.path.splitext(input_wav)
        combined_stem = f"{input_stem}_split"
        combined_mid = f"{combined_stem}.mid"

    low_mid = low_output or f"{combined_stem}_low_strings.mid"
    high_mid = high_output or f"{combined_stem}_upper_strings.mid"
    return low_mid, high_mid, combined_mid


def derive_split_preprocessed_paths(save_preprocessed_path: str | None) -> tuple[str | None, str | None]:
    """Derives optional preprocessed WAV export paths for the low/high split branches."""
    if not save_preprocessed_path:
        return None, None

    stem, extension = os.path.splitext(save_preprocessed_path)
    extension = extension or ".wav"
    return f"{stem}_low_strings{extension}", f"{stem}_upper_strings{extension}"


def transcribe_preprocessed_wav(
    temp_wav_path: str,
    output_mid: str,
    onset_thresh: float,
    frame_thresh: float,
    min_note_len_ms: int,
    min_velocity: int,
    minimum_frequency: float,
    maximum_frequency: float,
) -> tuple[pretty_midi.PrettyMIDI, dict[str, float]]:
    """Runs Basic Pitch on an already prepared temporary WAV and writes the cleaned MIDI."""
    timings: dict[str, float] = {}

    print(f"[*] Basic Pitch backend: {describe_basic_pitch_runtime()}")
    print("[*] Running polyphonic neural inference...")
    stage_start = time.perf_counter()
    _, midi_data, _ = predict(
        temp_wav_path,
        onset_threshold=onset_thresh,
        frame_threshold=frame_thresh,
        minimum_note_length=min_note_len_ms,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
        melodia_trick=False,
    )
    timings["inference_sec"] = time.perf_counter() - stage_start

    print(f"[*] Filtering ghost notes (vel < {min_velocity}, len < {min_note_len_ms}ms)...")
    stage_start = time.perf_counter()
    cleaned_midi = cleanup_midi(
        midi_data,
        min_velocity=min_velocity,
        min_duration_sec=(min_note_len_ms / 1000.0),
    )
    timings["cleanup_sec"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    cleaned_midi.write(output_mid)
    timings["write_midi_sec"] = time.perf_counter() - stage_start
    return cleaned_midi, timings


def convert_audio_segment_to_midi(
    source_audio: np.ndarray,
    sr: int,
    output_mid: str,
    onset_thresh: float,
    frame_thresh: float,
    min_note_len_ms: int,
    min_velocity: int,
    lowcut: float,
    highcut: float,
    minimum_frequency: float,
    maximum_frequency: float,
    normalize_peak: float,
    low_b_boost_db: float,
    low_b_boost_q: float,
    harmonic_attenuation_db: float,
    harmonic_notch_q: float,
    harmonic_count: int,
    branch_label: str,
    enhance_low_b: bool,
    save_preprocessed_path: str | None = None,
) -> tuple[pretty_midi.PrettyMIDI, dict[str, float]]:
    """Preprocesses one branch of audio and transcribes it to MIDI."""
    print(f"[*] Pre-processing {branch_label}: lowcut={lowcut:.1f}Hz highcut={highcut:.1f}Hz")
    temp_wav_path, preprocess_timings = preprocess_audio_array(
        source_audio,
        sr,
        lowcut,
        highcut,
        normalize_peak,
        low_b_boost_db,
        low_b_boost_q,
        harmonic_attenuation_db,
        harmonic_notch_q,
        harmonic_count,
        enhance_low_b=enhance_low_b,
        save_preprocessed_path=save_preprocessed_path,
    )

    try:
        midi_data, inference_timings = transcribe_preprocessed_wav(
            temp_wav_path,
            output_mid,
            onset_thresh,
            frame_thresh,
            min_note_len_ms,
            min_velocity,
            minimum_frequency,
            maximum_frequency,
        )
        timings = {**preprocess_timings, **inference_timings}
        print(f"[+] Successfully generated {branch_label}: {output_mid}")
        return midi_data, timings
    finally:
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)


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
    normalize_peak: float = DEFAULT_NORMALIZE_PEAK,
    low_b_boost_db: float = DEFAULT_LOW_B_BOOST_DB,
    low_b_boost_q: float = DEFAULT_LOW_B_BOOST_Q,
    harmonic_attenuation_db: float = DEFAULT_HARMONIC_ATTENUATION_DB,
    harmonic_notch_q: float = DEFAULT_HARMONIC_NOTCH_Q,
    harmonic_count: int = DEFAULT_HARMONIC_COUNT,
    enhance_low_b: bool = True,
    save_preprocessed_path: str | None = None,
):
    total_start = time.perf_counter()
    print(f"[*] Preparing source audio: {input_wav}")
    if transpose_semitones:
        print(f"[*] Transposing audio by {transpose_semitones:+.1f} semitones before inference...")
    if save_preprocessed_path:
        print(f"[*] Saving fully preprocessed audio to: {save_preprocessed_path}")
    source_audio, sr, timings, transpose_backend = load_source_audio(input_wav, transpose_semitones)
    if transpose_semitones:
        print(f"[*] Transpose backend: {transpose_backend}")

    _, branch_timings = convert_audio_segment_to_midi(
        source_audio,
        sr,
        output_mid,
        onset_thresh,
        frame_thresh,
        min_note_len_ms,
        min_velocity,
        lowcut,
        highcut,
        minimum_frequency,
        maximum_frequency,
        normalize_peak,
        low_b_boost_db,
        low_b_boost_q,
        harmonic_attenuation_db,
        harmonic_notch_q,
        harmonic_count,
        branch_label="full-range guitar pass",
        enhance_low_b=enhance_low_b,
        save_preprocessed_path=save_preprocessed_path,
    )

    timings.update(branch_timings)
    timings["total_sec"] = time.perf_counter() - total_start
    print_timings(timings)


def convert_wav_to_split_midi(
    input_wav: str,
    combined_output_mid: str | None,
    low_output_mid: str | None,
    high_output_mid: str | None,
    onset_thresh: float = 0.60,
    frame_thresh: float = 0.35,
    min_note_len_ms: int = DEFAULT_MIN_NOTE_LEN_MS,
    min_velocity: int = DEFAULT_MIN_VELOCITY,
    transpose_semitones: float = 0.0,
    normalize_peak: float = DEFAULT_NORMALIZE_PEAK,
    low_b_boost_db: float = DEFAULT_LOW_B_BOOST_DB,
    low_b_boost_q: float = DEFAULT_LOW_B_BOOST_Q,
    harmonic_attenuation_db: float = DEFAULT_HARMONIC_ATTENUATION_DB,
    harmonic_notch_q: float = DEFAULT_HARMONIC_NOTCH_Q,
    harmonic_count: int = DEFAULT_HARMONIC_COUNT,
    low_strings_lowcut: float = DEFAULT_SPLIT_LOW_STRINGS_LOWCUT_HZ,
    low_strings_highcut: float = DEFAULT_SPLIT_LOW_STRINGS_HIGHCUT_HZ,
    high_strings_lowcut: float = DEFAULT_SPLIT_HIGH_STRINGS_LOWCUT_HZ,
    high_strings_highcut: float = DEFAULT_SPLIT_HIGH_STRINGS_HIGHCUT_HZ,
    low_strings_max_frequency: float = DEFAULT_SPLIT_LOW_STRINGS_MAX_FREQUENCY_HZ,
    high_strings_min_frequency: float = DEFAULT_SPLIT_HIGH_STRINGS_MIN_FREQUENCY_HZ,
    maximum_frequency: float = DEFAULT_MAX_FREQUENCY_HZ,
    save_preprocessed_path: str | None = None,
) -> None:
    """Transcribes the low two strings and upper five strings separately, then merges them into two MIDI tracks."""
    total_start = time.perf_counter()
    low_mid, high_mid, combined_mid = derive_split_output_paths(
        input_wav,
        combined_output_mid,
        low_output_mid,
        high_output_mid,
    )
    low_preprocessed, high_preprocessed = derive_split_preprocessed_paths(save_preprocessed_path)

    print(f"[*] Preparing source audio for split transcription: {input_wav}")
    if transpose_semitones:
        print(f"[*] Transposing audio by {transpose_semitones:+.1f} semitones before inference...")
    source_audio, sr, source_timings, transpose_backend = load_source_audio(input_wav, transpose_semitones)
    if transpose_semitones:
        print(f"[*] Transpose backend: {transpose_backend}")

    low_midi, low_timings = convert_audio_segment_to_midi(
        source_audio,
        sr,
        low_mid,
        onset_thresh,
        frame_thresh,
        min_note_len_ms,
        min_velocity,
        low_strings_lowcut,
        low_strings_highcut,
        LOW_B1_MIDI_FLOOR_HZ,
        low_strings_max_frequency,
        normalize_peak,
        low_b_boost_db,
        low_b_boost_q,
        harmonic_attenuation_db,
        harmonic_notch_q,
        harmonic_count,
        branch_label="low two strings pass",
        enhance_low_b=True,
        save_preprocessed_path=low_preprocessed,
    )

    high_midi, high_timings = convert_audio_segment_to_midi(
        source_audio,
        sr,
        high_mid,
        onset_thresh,
        frame_thresh,
        min_note_len_ms,
        min_velocity,
        high_strings_lowcut,
        high_strings_highcut,
        high_strings_min_frequency,
        maximum_frequency,
        normalize_peak,
        low_b_boost_db,
        low_b_boost_q,
        harmonic_attenuation_db,
        harmonic_notch_q,
        harmonic_count,
        branch_label="upper five strings pass",
        enhance_low_b=False,
        save_preprocessed_path=high_preprocessed,
    )

    stage_start = time.perf_counter()
    combine_midi_tracks(low_midi, high_midi, combined_mid)
    combine_sec = time.perf_counter() - stage_start

    print(f"[+] Combined two-track MIDI: {combined_mid}")
    print(f"[+] Low strings MIDI: {low_mid}")
    print(f"[+] Upper strings MIDI: {high_mid}")

    split_timings = {}
    split_timings.update(source_timings)
    split_timings.update(prefix_timings("low_strings", low_timings))
    split_timings.update(prefix_timings("upper_strings", high_timings))
    split_timings["combine_midi_sec"] = combine_sec
    split_timings["total_sec"] = time.perf_counter() - total_start
    print_timings(split_timings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert guitar WAV audio to polyphonic MIDI.")
    parser.add_argument("input", help="Path to input .wav file")
    parser.add_argument("-o", "--output", help="Path to output .mid file, or combined output when using --split-seven-string", default=None)
    parser.add_argument("--onset", type=float, default=0.60, help="Onset sensitivity threshold [0.1 - 0.9] (default: 0.60)")
    parser.add_argument("--frame", type=float, default=0.35, help="Frame sustain threshold [0.1 - 0.9] (default: 0.35)")
    parser.add_argument("--min-len", type=int, default=DEFAULT_MIN_NOTE_LEN_MS, help="Minimum note length in ms (default: 20)")
    parser.add_argument("--min-vel", type=int, default=DEFAULT_MIN_VELOCITY, help="Minimum MIDI velocity threshold (default: 1)")
    parser.add_argument("--lowcut", type=float, default=DEFAULT_LOWCUT_HZ, help="Band-pass low cutoff in Hz (default: 40.0 for 7-string low B support)")
    parser.add_argument("--highcut", type=float, default=DEFAULT_HIGHCUT_HZ, help="Band-pass high cutoff in Hz (default: 5000.0)")
    parser.add_argument("--min-freq", type=float, default=LOW_B1_MIDI_FLOOR_HZ, help="Inference minimum frequency in Hz (default: 55.0 to include 7-string low B1 at 61.74 Hz)")
    parser.add_argument("--max-freq", type=float, default=DEFAULT_MAX_FREQUENCY_HZ, help="Inference maximum frequency in Hz (default: 1500.0)")
    parser.add_argument("--transpose", type=float, default=0.0, help="Pitch-shift the source audio before inference in semitones, from -12 to +12 (default: 0)")
    parser.add_argument("--normalize-peak", type=float, default=DEFAULT_NORMALIZE_PEAK, help="Peak normalization target applied after preprocessing, from 0 to 1 (default: 0.98)")
    parser.add_argument("--low-b-boost-db", type=float, default=DEFAULT_LOW_B_BOOST_DB, help="Peaking EQ boost in dB around low B1 after filtering (default: 8.0)")
    parser.add_argument("--low-b-boost-q", type=float, default=DEFAULT_LOW_B_BOOST_Q, help="Q for the low B1 peaking EQ boost (default: 0.8)")
    parser.add_argument("--harmonic-attenuation-db", type=float, default=DEFAULT_HARMONIC_ATTENUATION_DB, help="Approximate attenuation depth in dB for early low-B harmonics after filtering (default: 6.0)")
    parser.add_argument("--harmonic-notch-q", type=float, default=DEFAULT_HARMONIC_NOTCH_Q, help="Q for harmonic notch filters above low B1 (default: 5.0)")
    parser.add_argument("--harmonic-count", type=int, default=DEFAULT_HARMONIC_COUNT, help="How many low-B harmonics above the fundamental to attenuate (default: 3)")
    parser.add_argument("--save-preprocessed", help="Optional path to save the fully preprocessed WAV that will be fed into transcription", default=None)
    parser.add_argument("--split-seven-string", action="store_true", help="Run two separate Basic Pitch passes: one for the low B/E strings and one for the upper five strings, then combine them into a two-track MIDI file")
    parser.add_argument("--low-strings-output", help="Optional path for the low-strings MIDI file when using --split-seven-string", default=None)
    parser.add_argument("--high-strings-output", help="Optional path for the upper-strings MIDI file when using --split-seven-string", default=None)
    parser.add_argument("--low-strings-lowcut", type=float, default=DEFAULT_SPLIT_LOW_STRINGS_LOWCUT_HZ, help="Low cutoff for the low-strings split pass in Hz (default: 55.0)")
    parser.add_argument("--low-strings-highcut", type=float, default=DEFAULT_SPLIT_LOW_STRINGS_HIGHCUT_HZ, help="High cutoff for the low-strings split pass in Hz (default: 220.0)")
    parser.add_argument("--high-strings-lowcut", type=float, default=DEFAULT_SPLIT_HIGH_STRINGS_LOWCUT_HZ, help="Low cutoff for the upper-strings split pass in Hz (default: 95.0)")
    parser.add_argument("--high-strings-highcut", type=float, default=DEFAULT_SPLIT_HIGH_STRINGS_HIGHCUT_HZ, help="High cutoff for the upper-strings split pass in Hz (default: 5000.0)")
    parser.add_argument("--low-strings-max-freq", type=float, default=DEFAULT_SPLIT_LOW_STRINGS_MAX_FREQUENCY_HZ, help="Maximum inferred note frequency for the low-strings split pass in Hz (default: 350.0)")
    parser.add_argument("--high-strings-min-freq", type=float, default=DEFAULT_SPLIT_HIGH_STRINGS_MIN_FREQUENCY_HZ, help="Minimum inferred note frequency for the upper-strings split pass in Hz (default: 98.0)")
    
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
    if args.lowcut <= 0 or args.lowcut >= args.highcut:
        parser.error("--lowcut must be greater than 0 and less than --highcut")
    if args.low_strings_lowcut <= 0 or args.low_strings_lowcut >= args.low_strings_highcut:
        parser.error("--low-strings-lowcut must be greater than 0 and less than --low-strings-highcut")
    if args.high_strings_lowcut <= 0 or args.high_strings_lowcut >= args.high_strings_highcut:
        parser.error("--high-strings-lowcut must be greater than 0 and less than --high-strings-highcut")
    if args.low_strings_max_freq <= LOW_B1_MIDI_FLOOR_HZ:
        parser.error("--low-strings-max-freq must be greater than 55.0 Hz")
    if args.high_strings_min_freq <= 0:
        parser.error("--high-strings-min-freq must be greater than 0")

    out_file = args.output if args.output else os.path.splitext(args.input)[0] + ".mid"
    if args.split_seven_string:
        convert_wav_to_split_midi(
            args.input,
            combined_output_mid=args.output,
            low_output_mid=args.low_strings_output,
            high_output_mid=args.high_strings_output,
            onset_thresh=args.onset,
            frame_thresh=args.frame,
            min_note_len_ms=args.min_len,
            min_velocity=args.min_vel,
            transpose_semitones=args.transpose,
            normalize_peak=args.normalize_peak,
            low_b_boost_db=args.low_b_boost_db,
            low_b_boost_q=args.low_b_boost_q,
            harmonic_attenuation_db=args.harmonic_attenuation_db,
            harmonic_notch_q=args.harmonic_notch_q,
            harmonic_count=args.harmonic_count,
            low_strings_lowcut=args.low_strings_lowcut,
            low_strings_highcut=args.low_strings_highcut,
            high_strings_lowcut=args.high_strings_lowcut,
            high_strings_highcut=args.high_strings_highcut,
            low_strings_max_frequency=args.low_strings_max_freq,
            high_strings_min_frequency=args.high_strings_min_freq,
            maximum_frequency=args.max_freq,
            save_preprocessed_path=args.save_preprocessed,
        )
    else:
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
            normalize_peak=args.normalize_peak,
            low_b_boost_db=args.low_b_boost_db,
            low_b_boost_q=args.low_b_boost_q,
            harmonic_attenuation_db=args.harmonic_attenuation_db,
            harmonic_notch_q=args.harmonic_notch_q,
            harmonic_count=args.harmonic_count,
            save_preprocessed_path=args.save_preprocessed,
        )