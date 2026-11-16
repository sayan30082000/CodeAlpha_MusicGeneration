"""Shared helpers: turning music21 objects into text tokens and back.

A token describes one musical event and its length in quarter notes:
    "E5_1.0"          -> the note E5 for one beat
    "C4.E4.G4_2.0"    -> a C-major chord for two beats
    "R_0.5"           -> a rest for half a beat
    "END"             -> boundary between two pieces in the training stream
"""
from music21 import chord, instrument, meter, note, pitch, stream, tempo

DURATIONS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
END_TOKEN = "END"


def quantize_duration(quarter_length: float) -> float | None:
    """Snap a duration to the nearest allowed value (None for grace notes)."""
    if quarter_length < 0.125:
        return None
    return min(DURATIONS, key=lambda d: abs(d - quarter_length))


def pitch_name(midi_number: int) -> str:
    """Canonical spelling for a MIDI number (61 -> 'C#4'), so C#4 and D-4 become one token."""
    return pitch.Pitch(midi=midi_number).nameWithOctave


def element_to_token(el) -> str | None:
    dur = quantize_duration(float(el.duration.quarterLength))
    if dur is None:
        return None
    if isinstance(el, note.Note):
        return f"{pitch_name(el.pitch.midi)}_{dur}"
    if isinstance(el, chord.Chord):
        midis = sorted({p.midi for p in el.pitches})
        return ".".join(pitch_name(m) for m in midis) + f"_{dur}"
    if isinstance(el, note.Rest):
        return f"R_{dur}"
    return None


def parse_token(token: str) -> tuple[list[str], float]:
    """'C4.E4.G4_2.0' -> (['C4', 'E4', 'G4'], 2.0); a rest returns an empty pitch list."""
    body, dur = token.rsplit("_", 1)
    pitches = [] if body == "R" else body.split(".")
    return pitches, float(dur)


_transpose_cache: dict[tuple[str, int], str] = {}


def transpose_token(token: str, semitones: int) -> str:
    """Shift every pitch in a token up/down — used for data augmentation."""
    if semitones == 0 or token == END_TOKEN or token.startswith("R_"):
        return token
    key = (token, semitones)
    if key not in _transpose_cache:
        pitches, dur = parse_token(token)
        moved = [pitch_name(pitch.Pitch(p).midi + semitones) for p in pitches]
        _transpose_cache[key] = ".".join(moved) + f"_{dur}"
    return _transpose_cache[key]


def tokens_to_stream(tokens: list[str], bpm: int = 90) -> stream.Stream:
    """Build a playable music21 stream (piano) from tokens."""
    out = stream.Stream()
    out.append(instrument.Piano())
    out.append(tempo.MetronomeMark(number=bpm))
    out.append(meter.TimeSignature("4/4"))
    for tok in tokens:
        if tok == END_TOKEN:
            continue
        pitches, dur = parse_token(tok)
        if not pitches:
            out.append(note.Rest(quarterLength=dur))
        elif len(pitches) == 1:
            out.append(note.Note(pitches[0], quarterLength=dur))
        else:
            out.append(chord.Chord(pitches, quarterLength=dur))
    return out


def tokens_to_events(tokens: list[str], bpm: int = 90) -> list[tuple[float, float, list[int]]]:
    """(start_seconds, duration_seconds, [midi pitches]) for every event — used by the audio synth."""
    sec_per_beat = 60.0 / bpm
    events, t = [], 0.0
    for tok in tokens:
        if tok == END_TOKEN:
            continue
        pitches, dur = parse_token(tok)
        seconds = dur * sec_per_beat
        if pitches:
            events.append((t, seconds, [pitch.Pitch(p).midi for p in pitches]))
        t += seconds
    return events
