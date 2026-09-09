#!/usr/bin/env python3
"""Generate expressive guitar lead suites as multi-lead MIDI files."""

import argparse
import math
import os
import random
import shutil
from dataclasses import dataclass

import pretty_midi


TEMPO_BPM = 80
BEAT_SECONDS = 60.0 / TEMPO_BPM
DEFAULT_FILE_COUNT = 1
LEADS_PER_FILE = 4
LEAD_LENGTH_BEATS = 16
SUITE_LENGTH_BEATS = LEADS_PER_FILE * LEAD_LENGTH_BEATS
DEFAULT_OUTPUT_DIR = os.path.join("Audio_Samples", "Generated_Guitar_Leads")
GUITAR_PROGRAM = 29
MUTED_GUITAR_PROGRAM = 28
PITCH_BEND_RANGE_SEMITONES = 2.0
MIN_PITCH = 40
MAX_PITCH = 76

NOTE_TO_PITCH_CLASS = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}

CHORD_NOTE_NAMES = {
    "Fmaj7": ["F", "A", "C", "E"],
    "G7": ["G", "B", "D", "F"],
    "Am9": ["A", "C", "E", "G", "B"],
}

COMMON_GUITAR_ARTICULATIONS = [
    "picked single note",
    "double-stop",
    "hammer-on",
    "pull-off",
    "legato chain",
    "slide into a note",
    "slide up between fretted notes",
    "slide down between fretted notes",
    "shift slide",
    "half-step bend",
    "whole-step bend",
    "bend to unison",
    "bend-and-hold",
    "bend release",
    "pre-bend",
    "pre-bend release / unbend",
    "double-stop with one bent note and one static picked note",
    "double-stop pre-bend release from unison",
    "vibrato",
    "trill",
    "rake into a note",
    "ghosted / muted attack",
    "natural harmonic",
    "pinch harmonic",
    "tremolo picking",
    "tap",
    "string skip",
    "rest with rapid slide-down falloff",
]

GUARANTEED_ARTICULATIONS = [
    "picked single note",
    "hammer-on",
    "pull-off",
    "slide into a note",
    "slide up between fretted notes",
    "slide down into a rest",
    "standard bend",
    "bend to unison",
    "pre-bend release / unbend",
    "double-stop with one bent note and one static picked note",
    "double-stop pre-bend release from unison",
    "vibrato",
    "rest with rapid slide-down falloff and diminishing velocity",
]

SLOT_OFFSETS = [0.0] * 8 + [0.5] * 4 + [0.25, 0.75, 0.25, 0.75]


@dataclass(frozen=True)
class Segment:
    chord_name: str
    start_beat: float
    duration_beats: float

    @property
    def end_beat(self) -> float:
        return self.start_beat + self.duration_beats


PROGRESSION = [
    Segment("Fmaj7", 0.0, 7.0),
    Segment("G7", 7.0, 1.0),
    Segment("Am9", 8.0, 8.0),
]


def beat_to_time(beat: float) -> float:
    return beat * BEAT_SECONDS


def build_pitch_pool(note_names: list[str], minimum_pitch: int, maximum_pitch: int) -> list[int]:
    pitch_classes = {NOTE_TO_PITCH_CLASS[name] for name in note_names}
    return [pitch for pitch in range(minimum_pitch, maximum_pitch + 1) if pitch % 12 in pitch_classes]


CHORD_PITCH_POOLS = {
    name: build_pitch_pool(notes, MIN_PITCH, MAX_PITCH) for name, notes in CHORD_NOTE_NAMES.items()
}


def get_segment_for_beat(beat: float) -> Segment:
    local_beat = beat % LEAD_LENGTH_BEATS
    for segment in PROGRESSION:
        if segment.start_beat <= local_beat < segment.end_beat:
            return segment
    return PROGRESSION[-1]


def clamp_pitch(pitch: int, minimum_pitch: int = MIN_PITCH, maximum_pitch: int = MAX_PITCH) -> int:
    return max(minimum_pitch, min(maximum_pitch, pitch))


def choose_chord_tone(chord_name: str, rng: random.Random, near_pitch: int | None = None) -> int:
    pool = CHORD_PITCH_POOLS[chord_name]
    if near_pitch is None:
        return rng.choice([pitch for pitch in pool if 50 <= pitch <= 68])
    return min(pool, key=lambda pitch: (abs(pitch - near_pitch), rng.random()))


def choose_motion_target(
    chord_name: str,
    rng: random.Random,
    anchor: int,
    motion: int,
    min_interval: int,
    max_interval: int,
) -> int:
    desired = anchor + motion * rng.randint(min_interval, max_interval)
    pool = CHORD_PITCH_POOLS[chord_name]
    candidates = [pitch for pitch in pool if (pitch - anchor) * motion >= 0]
    if not candidates:
        candidates = pool
    return min(candidates, key=lambda pitch: (abs(pitch - desired), rng.random()))


def semitones_to_pitch_bend(semitones: float) -> int:
    normalized = max(-PITCH_BEND_RANGE_SEMITONES, min(PITCH_BEND_RANGE_SEMITONES, semitones))
    return int(round((normalized / PITCH_BEND_RANGE_SEMITONES) * 8191))


def add_note(instrument: pretty_midi.Instrument, pitch: int, start: float, end: float, velocity: int) -> None:
    instrument.notes.append(
        pretty_midi.Note(
            velocity=max(1, min(127, velocity)),
            pitch=clamp_pitch(pitch),
            start=max(0.0, start),
            end=max(start + 0.01, end),
        )
    )


def add_pitch_bend(instrument: pretty_midi.Instrument, time_seconds: float, semitones: float) -> None:
    instrument.pitch_bends.append(
        pretty_midi.PitchBend(pitch=semitones_to_pitch_bend(semitones), time=max(0.0, time_seconds))
    )


def add_bend_curve(
    instrument: pretty_midi.Instrument,
    start: float,
    end: float,
    start_semitones: float,
    end_semitones: float,
    steps: int = 8,
) -> None:
    steps = max(2, steps)
    for index in range(steps + 1):
        position = index / steps
        semitones = start_semitones + (end_semitones - start_semitones) * position
        add_pitch_bend(instrument, start + (end - start) * position, semitones)


def add_vibrato(
    instrument: pretty_midi.Instrument,
    start: float,
    end: float,
    width_semitones: float,
    cycles: float,
    width_variation: float,
    rng: random.Random,
) -> None:
    total_points = int(max(14, round(cycles * 14)))
    width_phase = rng.uniform(0.0, math.tau)
    speed_phase = rng.uniform(0.0, math.tau)
    for index in range(total_points + 1):
        position = index / total_points
        drift = 1.0 + 0.08 * math.sin((position * math.pi) + speed_phase)
        phase = (position * cycles * drift * math.tau) + speed_phase
        width_scale = 1.0 + width_variation * math.sin((position * math.pi * 1.5) + width_phase)
        jitter_scale = 1.0 + rng.uniform(-0.06, 0.06)
        semitones = width_semitones * width_scale * jitter_scale * math.sin(phase)
        add_pitch_bend(instrument, start + (end - start) * position, semitones)
    add_pitch_bend(instrument, end, 0.0)


def add_tail_vibrato(
    instrument: pretty_midi.Instrument,
    start_beat: float,
    duration_beats: float,
    rng: random.Random,
) -> None:
    vib_start = beat_to_time(start_beat + duration_beats * 0.38)
    vib_end = beat_to_time(start_beat + duration_beats * 0.95)
    vibrato_cycles = rng.uniform(0.65, 1.15)
    vibrato_width = rng.uniform(0.14, 0.30)
    width_variation = rng.uniform(0.14, 0.30)
    add_vibrato(
        instrument,
        vib_start,
        vib_end,
        vibrato_width,
        cycles=vibrato_cycles,
        width_variation=width_variation,
        rng=rng,
    )


def enforce_monophonic_instrument(instrument: pretty_midi.Instrument, min_gap_seconds: float = 0.010) -> None:
    """Serializes notes inside one MIDI track so no two notes overlap in time."""
    ordered_notes = sorted(instrument.notes, key=lambda note: (note.start, note.pitch, note.end))
    monophonic_notes: list[pretty_midi.Note] = []

    for source_note in ordered_notes:
        note = pretty_midi.Note(
            velocity=source_note.velocity,
            pitch=source_note.pitch,
            start=source_note.start,
            end=source_note.end,
        )

        if monophonic_notes:
            previous_note = monophonic_notes[-1]
            min_start = previous_note.end + min_gap_seconds
            if note.start < min_start:
                note.start = min_start
            if note.start < previous_note.end + min_gap_seconds:
                previous_note.end = max(previous_note.start + 0.01, note.start - min_gap_seconds)

        note.end = max(note.start + 0.01, note.end)
        monophonic_notes.append(note)

    instrument.notes = monophonic_notes


def choose_start_profile(file_index: int, lead_index: int, rng: random.Random) -> tuple[list[int], int, int]:
    centers = [46, 50, 53, 57, 60, 62, 65]
    center = centers[(file_index * LEADS_PER_FILE + lead_index) % len(centers)] + rng.choice([-2, 0, 2])
    motion = 1 if (file_index + lead_index) % 2 == 0 else -1
    start_pitch = choose_chord_tone("Fmaj7", rng, near_pitch=center)
    if lead_index % 2 == 1:
        partner = choose_chord_tone("Fmaj7", rng, near_pitch=start_pitch + rng.choice([3, 5, 7]))
        return sorted({start_pitch, partner}), start_pitch, motion
    return [start_pitch], start_pitch, motion


def add_start_phrase(
    voice_a: pretty_midi.Instrument,
    voice_b: pretty_midi.Instrument,
    start_beat: float,
    start_pitches: list[int],
    rng: random.Random,
) -> int:
    duration_beats = 0.92
    velocities = [86, 72]
    for index, pitch in enumerate(start_pitches):
        target_voice = voice_a if index == 0 else voice_b
        add_note(
            target_voice,
            pitch,
            beat_to_time(start_beat),
            beat_to_time(start_beat + duration_beats),
            velocities[index],
        )
        add_tail_vibrato(target_voice, start_beat, duration_beats, rng)
    return start_pitches[0]


def phrase_picked(
    voice_a: pretty_midi.Instrument,
    start_beat: float,
    chord_name: str,
    rng: random.Random,
    anchor: int,
    motion: int,
) -> int:
    target = choose_motion_target(chord_name, rng, anchor, motion, 1, 4)
    duration_beats = 0.78
    add_note(voice_a, target, beat_to_time(start_beat), beat_to_time(start_beat + duration_beats), 84)
    return target


def phrase_hammer_on(
    voice_a: pretty_midi.Instrument,
    start_beat: float,
    chord_name: str,
    rng: random.Random,
    anchor: int,
    motion: int,
) -> int:
    target = choose_motion_target(chord_name, rng, anchor, motion, 2, 5)
    if motion > 0:
        source = clamp_pitch(target - rng.choice([1, 2]))
    else:
        source = clamp_pitch(target + rng.choice([1, 2]))
    add_note(voice_a, source, beat_to_time(start_beat), beat_to_time(start_beat + 0.24), 74)
    add_note(voice_a, target, beat_to_time(start_beat + 0.18), beat_to_time(start_beat + 0.88), 82)
    return target


def phrase_pull_off(
    voice_a: pretty_midi.Instrument,
    start_beat: float,
    chord_name: str,
    rng: random.Random,
    anchor: int,
    motion: int,
) -> int:
    upper = choose_motion_target(chord_name, rng, anchor, 1, 1, 4)
    lower = clamp_pitch(upper - rng.choice([1, 2]))
    add_note(voice_a, upper, beat_to_time(start_beat), beat_to_time(start_beat + 0.26), 86)
    add_note(voice_a, lower, beat_to_time(start_beat + 0.20), beat_to_time(start_beat + 0.86), 72)
    return lower if motion < 0 else upper


def phrase_slide_in(
    voice_a: pretty_midi.Instrument,
    start_beat: float,
    chord_name: str,
    rng: random.Random,
    anchor: int,
    motion: int,
) -> int:
    target = choose_motion_target(chord_name, rng, anchor, motion, 1, 3)
    source = clamp_pitch(target - motion * rng.choice([1, 2]))
    duration_beats = 0.88
    add_note(voice_a, source, beat_to_time(start_beat), beat_to_time(start_beat + duration_beats), 80)
    add_bend_curve(voice_a, beat_to_time(start_beat), beat_to_time(start_beat + 0.36), 0.0, float(target - source), steps=7)
    add_pitch_bend(voice_a, beat_to_time(start_beat + 0.42), 0.0)
    add_tail_vibrato(voice_a, start_beat, duration_beats, rng)
    return target


def phrase_slide_up(
    voice_a: pretty_midi.Instrument,
    start_beat: float,
    chord_name: str,
    rng: random.Random,
    anchor: int,
    motion: int,
) -> int:
    target = choose_motion_target(chord_name, rng, anchor, 1, 2, 4)
    source = clamp_pitch(target - rng.choice([2, 3]))
    duration_beats = 0.90
    add_note(voice_a, source, beat_to_time(start_beat), beat_to_time(start_beat + duration_beats), 82)
    add_bend_curve(voice_a, beat_to_time(start_beat + 0.08), beat_to_time(start_beat + 0.56), 0.0, float(target - source), steps=8)
    add_pitch_bend(voice_a, beat_to_time(start_beat + 0.64), 0.0)
    add_tail_vibrato(voice_a, start_beat, duration_beats, rng)
    return target if motion > 0 else source


def phrase_slide_down(
    voice_a: pretty_midi.Instrument,
    start_beat: float,
    chord_name: str,
    rng: random.Random,
    anchor: int,
) -> int:
    source = choose_chord_tone(chord_name, rng, near_pitch=anchor + 2)
    target = clamp_pitch(source - rng.choice([2, 3]))
    duration_beats = 0.70
    add_note(voice_a, source, beat_to_time(start_beat), beat_to_time(start_beat + duration_beats), 76)
    add_bend_curve(voice_a, beat_to_time(start_beat + 0.02), beat_to_time(start_beat + 0.46), 0.0, float(target - source), steps=8)
    add_pitch_bend(voice_a, beat_to_time(start_beat + 0.52), 0.0)
    return target


def phrase_standard_bend(
    voice_a: pretty_midi.Instrument,
    start_beat: float,
    chord_name: str,
    rng: random.Random,
    anchor: int,
    motion: int,
) -> int:
    base_pitch = choose_motion_target(chord_name, rng, anchor, motion, 1, 3)
    duration_beats = 0.92
    add_note(voice_a, base_pitch, beat_to_time(start_beat), beat_to_time(start_beat + duration_beats), 88)
    bend_size = rng.choice([0.5, 1.0])
    add_bend_curve(voice_a, beat_to_time(start_beat + 0.14), beat_to_time(start_beat + 0.48), 0.0, bend_size, steps=7)
    add_bend_curve(voice_a, beat_to_time(start_beat + 0.48), beat_to_time(start_beat + 0.78), bend_size, 0.25, steps=6)
    add_tail_vibrato(voice_a, start_beat + 0.50, 0.40, rng)
    add_pitch_bend(voice_a, beat_to_time(start_beat + duration_beats), 0.0)
    return base_pitch


def phrase_unison_bend(
    voice_a: pretty_midi.Instrument,
    voice_b: pretty_midi.Instrument,
    start_beat: float,
    chord_name: str,
    rng: random.Random,
    anchor: int,
    motion: int,
) -> int:
    static_pitch = choose_motion_target(chord_name, rng, anchor, motion, 2, 5)
    bent_pitch = clamp_pitch(static_pitch - rng.choice([1, 2]))
    duration_beats = 0.94
    add_note(voice_a, bent_pitch, beat_to_time(start_beat), beat_to_time(start_beat + duration_beats), 84)
    add_note(voice_b, static_pitch, beat_to_time(start_beat), beat_to_time(start_beat + duration_beats), 76)
    add_bend_curve(voice_a, beat_to_time(start_beat + 0.10), beat_to_time(start_beat + 0.56), 0.0, float(static_pitch - bent_pitch), steps=8)
    add_tail_vibrato(voice_a, start_beat + 0.54, 0.34, rng)
    add_pitch_bend(voice_a, beat_to_time(start_beat + duration_beats), 0.0)
    return static_pitch


def phrase_pre_bend_release(
    voice_a: pretty_midi.Instrument,
    start_beat: float,
    chord_name: str,
    rng: random.Random,
    anchor: int,
    motion: int,
) -> int:
    base_pitch = choose_motion_target(chord_name, rng, anchor, motion, 1, 4)
    bend_size = rng.choice([1.0, 2.0])
    duration_beats = 0.90
    add_pitch_bend(voice_a, beat_to_time(start_beat), bend_size)
    add_note(voice_a, base_pitch, beat_to_time(start_beat), beat_to_time(start_beat + duration_beats), 85)
    add_bend_curve(voice_a, beat_to_time(start_beat + 0.03), beat_to_time(start_beat + 0.62), bend_size, 0.0, steps=8)
    add_tail_vibrato(voice_a, start_beat + 0.54, 0.26, rng)
    add_pitch_bend(voice_a, beat_to_time(start_beat + duration_beats), 0.0)
    return base_pitch


def phrase_double_stop_bend(
    voice_a: pretty_midi.Instrument,
    voice_b: pretty_midi.Instrument,
    start_beat: float,
    chord_name: str,
    rng: random.Random,
    anchor: int,
    motion: int,
) -> int:
    static_pitch = choose_motion_target(chord_name, rng, anchor + 4, motion, 1, 4)
    bent_pitch = choose_motion_target(chord_name, rng, anchor - 2, motion, 1, 3)
    interval = rng.choice([1.0, 2.0])
    duration_beats = 0.92
    add_note(voice_a, bent_pitch, beat_to_time(start_beat), beat_to_time(start_beat + duration_beats), 86)
    add_note(voice_b, static_pitch, beat_to_time(start_beat), beat_to_time(start_beat + duration_beats), 72)
    add_bend_curve(voice_a, beat_to_time(start_beat + 0.12), beat_to_time(start_beat + 0.50), 0.0, interval, steps=7)
    add_tail_vibrato(voice_a, start_beat + 0.48, 0.32, rng)
    add_pitch_bend(voice_a, beat_to_time(start_beat + duration_beats), 0.0)
    return static_pitch


def phrase_double_stop_pre_bend_release(
    voice_a: pretty_midi.Instrument,
    voice_b: pretty_midi.Instrument,
    start_beat: float,
    chord_name: str,
    rng: random.Random,
    anchor: int,
    motion: int,
) -> int:
    static_pitch = choose_motion_target(chord_name, rng, anchor + 3, motion, 1, 4)
    release_pitch = clamp_pitch(static_pitch - rng.choice([1, 2]))
    bend_size = float(static_pitch - release_pitch)
    duration_beats = 0.95
    add_pitch_bend(voice_a, beat_to_time(start_beat), bend_size)
    add_note(voice_a, release_pitch, beat_to_time(start_beat), beat_to_time(start_beat + duration_beats), 84)
    add_note(voice_b, static_pitch, beat_to_time(start_beat), beat_to_time(start_beat + duration_beats), 74)
    add_bend_curve(voice_a, beat_to_time(start_beat + 0.04), beat_to_time(start_beat + 0.70), bend_size, 0.0, steps=9)
    add_tail_vibrato(voice_b, start_beat, duration_beats, rng)
    add_pitch_bend(voice_a, beat_to_time(start_beat + duration_beats), 0.0)
    return release_pitch


def phrase_vibrato(
    voice_a: pretty_midi.Instrument,
    start_beat: float,
    chord_name: str,
    rng: random.Random,
    anchor: int,
    motion: int,
) -> int:
    pitch = choose_motion_target(chord_name, rng, anchor, motion, 1, 3)
    duration_beats = 0.96
    add_note(voice_a, pitch, beat_to_time(start_beat), beat_to_time(start_beat + duration_beats), 82)
    add_tail_vibrato(voice_a, start_beat, duration_beats, rng)
    return pitch


def phrase_legato_chain(
    voice_a: pretty_midi.Instrument,
    start_beat: float,
    chord_name: str,
    rng: random.Random,
    anchor: int,
    motion: int,
) -> int:
    first = choose_motion_target(chord_name, rng, anchor, motion, 1, 2)
    second = choose_motion_target(chord_name, rng, first, motion, 1, 2)
    third = choose_motion_target(chord_name, rng, second, -motion, 1, 2)
    add_note(voice_a, first, beat_to_time(start_beat), beat_to_time(start_beat + 0.18), 70)
    add_note(voice_a, second, beat_to_time(start_beat + 0.14), beat_to_time(start_beat + 0.42), 78)
    add_note(voice_a, third, beat_to_time(start_beat + 0.38), beat_to_time(start_beat + 0.88), 73)
    return third


def phrase_two_voice_cadence(
    voice_a: pretty_midi.Instrument,
    voice_b: pretty_midi.Instrument,
    start_beat: float,
    chord_name: str,
    rng: random.Random,
    anchor: int,
    motion: int,
) -> int:
    high_pitch = choose_motion_target(chord_name, rng, anchor + 2, motion, 1, 4)
    low_pitch = choose_motion_target(chord_name, rng, anchor - 5, -motion, 1, 3)
    add_note(voice_a, high_pitch, beat_to_time(start_beat), beat_to_time(start_beat + 0.82), 78)
    add_note(voice_b, low_pitch, beat_to_time(start_beat + 0.08), beat_to_time(start_beat + 0.76), 64)
    add_tail_vibrato(voice_a, start_beat, 0.82, rng)
    return high_pitch


def phrase_rest_falloff(
    voice_a: pretty_midi.Instrument,
    falloff_voice: pretty_midi.Instrument,
    start_beat: float,
    chord_name: str,
    rng: random.Random,
    anchor: int,
) -> int:
    pitch = phrase_slide_down(voice_a, start_beat, chord_name, rng, anchor)
    falloff_start = start_beat + 0.72
    note_count = 5
    falloff_window = 0.34
    current_pitch = pitch
    for index in range(note_count):
        note_start = falloff_start + (index / note_count) * falloff_window
        note_end = falloff_start + ((index + 0.65) / note_count) * falloff_window
        velocity = max(6, 48 - index * 11)
        if index > 0:
            current_pitch = clamp_pitch(current_pitch - rng.choice([1, 2]))
        add_note(falloff_voice, current_pitch, beat_to_time(note_start), beat_to_time(note_end), velocity)
    return current_pitch


ARTICULATION_FUNCTIONS = {
    "picked": lambda a, b, f, s, c, r, p, m: phrase_picked(a, s, c, r, p, m),
    "hammer_on": lambda a, b, f, s, c, r, p, m: phrase_hammer_on(a, s, c, r, p, m),
    "pull_off": lambda a, b, f, s, c, r, p, m: phrase_pull_off(a, s, c, r, p, m),
    "slide_in": lambda a, b, f, s, c, r, p, m: phrase_slide_in(a, s, c, r, p, m),
    "slide_up": lambda a, b, f, s, c, r, p, m: phrase_slide_up(a, s, c, r, p, m),
    "standard_bend": lambda a, b, f, s, c, r, p, m: phrase_standard_bend(a, s, c, r, p, m),
    "unison_bend": lambda a, b, f, s, c, r, p, m: phrase_unison_bend(a, b, s, c, r, p, m),
    "pre_bend_release": lambda a, b, f, s, c, r, p, m: phrase_pre_bend_release(a, s, c, r, p, m),
    "double_stop_bend": lambda a, b, f, s, c, r, p, m: phrase_double_stop_bend(a, b, s, c, r, p, m),
    "double_stop_pre_bend": lambda a, b, f, s, c, r, p, m: phrase_double_stop_pre_bend_release(a, b, s, c, r, p, m),
    "vibrato": lambda a, b, f, s, c, r, p, m: phrase_vibrato(a, s, c, r, p, m),
    "legato_chain": lambda a, b, f, s, c, r, p, m: phrase_legato_chain(a, s, c, r, p, m),
    "cadence": lambda a, b, f, s, c, r, p, m: phrase_two_voice_cadence(a, b, s, c, r, p, m),
    "rest_falloff": lambda a, b, f, s, c, r, p, m: phrase_rest_falloff(a, f, s, c, r, p),
}


def build_articulation_order(rng: random.Random) -> list[str]:
    order = [
        "picked",
        "hammer_on",
        "pull_off",
        "slide_in",
        "slide_up",
        "standard_bend",
        "unison_bend",
        "pre_bend_release",
        "double_stop_bend",
        "double_stop_pre_bend",
        "vibrato",
        "vibrato",
        "vibrato",
        "legato_chain",
        "cadence",
    ]
    rng.shuffle(order)
    return order


def build_start_slots(rng: random.Random, lead_start: float) -> list[float]:
    offsets = list(SLOT_OFFSETS)
    rng.shuffle(offsets)
    return [lead_start + index + offsets[index] for index in range(len(offsets))]


def sort_events(instrument: pretty_midi.Instrument) -> None:
    instrument.notes.sort(key=lambda note: (note.start, note.pitch, note.end))
    instrument.pitch_bends.sort(key=lambda bend: bend.time)
    instrument.control_changes.sort(key=lambda control_change: control_change.time)


def print_reference_information() -> None:
    print("Chord tones for the progression:")
    print("  Fmaj7: F, A, C, E")
    print("  G7: G, B, D, F")
    print("  Am9: A, C, E, G, B")
    print()
    print("Common guitar lead articulations:")
    for articulation in COMMON_GUITAR_ARTICULATIONS:
        print(f"  - {articulation}")
    print()
    print("Articulations guaranteed in every generated lead:")
    for articulation in GUARANTEED_ARTICULATIONS:
        print(f"  - {articulation}")
    print()


def prepare_output_directory(output_dir: str) -> tuple[str | None, int]:
    if not os.path.isdir(output_dir) or not os.listdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        return None, 0

    suffix = 1
    backup_dir = f"{output_dir}_{suffix}"
    while os.path.exists(backup_dir):
        suffix += 1
        backup_dir = f"{output_dir}_{suffix}"

    shutil.copytree(output_dir, backup_dir)
    shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    return backup_dir, suffix


def create_lead_segment(
    voice_a: pretty_midi.Instrument,
    voice_b: pretty_midi.Instrument,
    falloff_voice: pretty_midi.Instrument,
    file_index: int,
    lead_index: int,
    rng: random.Random,
) -> None:
    lead_start = lead_index * LEAD_LENGTH_BEATS
    start_pitches, previous_anchor, motion = choose_start_profile(file_index, lead_index, rng)
    slots = build_start_slots(rng, lead_start)
    order = build_articulation_order(rng)
    articulation_slots = slots[1:13]
    rest_slots = sorted(slots[13:])

    previous_anchor = add_start_phrase(voice_a, voice_b, slots[0], start_pitches, rng)
    for slot_beat, articulation_name in zip(articulation_slots, order):
        chord_name = get_segment_for_beat(slot_beat).chord_name
        next_anchor = ARTICULATION_FUNCTIONS[articulation_name](
            voice_a,
            voice_b,
            falloff_voice,
            slot_beat,
            chord_name,
            rng,
            previous_anchor,
            motion,
        )
        if articulation_name == "vibrato":
            motion *= -1
        elif (next_anchor - previous_anchor) * motion < 0:
            motion *= -1
        previous_anchor = next_anchor

    for rest_index, rest_slot in enumerate(rest_slots):
        chord_name = get_segment_for_beat(rest_slot).chord_name
        previous_anchor = ARTICULATION_FUNCTIONS["rest_falloff"](
            voice_a,
            voice_b,
            falloff_voice,
            rest_slot,
            chord_name,
            rng,
            previous_anchor,
            motion,
        )
        if rest_index < len(rest_slots) - 1:
            motion *= -1


def create_lead_suite_midi(file_index: int, seed: int, output_dir: str) -> str:
    rng = random.Random(seed + file_index * 104729)
    midi = pretty_midi.PrettyMIDI(initial_tempo=TEMPO_BPM)
    midi.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0.0))

    voice_a = pretty_midi.Instrument(program=GUITAR_PROGRAM, name="Lead voice 1")
    voice_b = pretty_midi.Instrument(program=GUITAR_PROGRAM, name="Lead voice 2")
    falloff_voice = pretty_midi.Instrument(program=MUTED_GUITAR_PROGRAM, name="Rapid falloff muted guitar")

    for lead_index in range(LEADS_PER_FILE):
        create_lead_segment(voice_a, voice_b, falloff_voice, file_index, lead_index, rng)

    enforce_monophonic_instrument(voice_a)
    enforce_monophonic_instrument(voice_b)
    enforce_monophonic_instrument(falloff_voice)
    sort_events(voice_a)
    sort_events(voice_b)
    sort_events(falloff_voice)
    midi.instruments.extend([voice_a, voice_b, falloff_voice])

    output_path = os.path.join(output_dir, f"guitar_lead_suite_{file_index:02d}.mid")
    midi.write(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate expressive 64-beat guitar lead suite MIDI files.")
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_FILE_COUNT,
        help="How many 64-beat suite MIDI files to generate (default: 1)",
    )
    parser.add_argument("--seed", type=int, default=17, help="Base RNG seed used to vary the generated suites")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory where generated MIDI files are written")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise SystemExit("--count must be greater than 0")

    print_reference_information()
    print("Assumption: '80 bps' was interpreted as 80 BPM in 4/4 time.")
    backup_dir, generation_index = prepare_output_directory(args.output_dir)
    effective_seed = args.seed + generation_index * 1009
    if backup_dir:
        print(f"Backed up existing output folder to: {backup_dir}")
    print(f"Using effective seed: {effective_seed}")
    print(
        f"Generating {args.count} suite file(s) into: {args.output_dir} "
        f"with {LEADS_PER_FILE} leads per file and {SUITE_LENGTH_BEATS} beats each"
    )
    for file_index in range(1, args.count + 1):
        output_path = create_lead_suite_midi(file_index, effective_seed, args.output_dir)
        print(f"  wrote {output_path}")


if __name__ == "__main__":
    main()