"""A tiny additive synthesizer: renders note events to a .wav file with NumPy.

MIDI files need a synth to be heard; this makes a WAV that plays anywhere
(no SoundFont or FluidSynth required). The tone is a few sine harmonics with a
piano-like decay envelope.
"""
import numpy as np
from scipy.io import wavfile

SAMPLE_RATE = 22050
HARMONICS = [(1, 1.0), (2, 0.45), (3, 0.25), (4, 0.12), (5, 0.06)]
RELEASE = 0.35  # seconds of fade-out after a note ends


def midi_to_hz(midi: int) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def render_note(freq: float, seconds: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    n = int((seconds + RELEASE) * sr)
    t = np.arange(n) / sr
    wave = sum(amp * np.sin(2 * np.pi * freq * h * t) for h, amp in HARMONICS)

    attack = np.clip(t / 0.01, 0, 1)                             # 10 ms attack
    decay = np.exp(-t * 2.5)                                      # piano-like decay
    release = np.clip(1 - (t - seconds) / RELEASE, 0, 1)          # fade out at the end
    return wave * attack * decay * release


def render_events(events, path: str, sr: int = SAMPLE_RATE) -> float:
    """events: [(start_seconds, duration_seconds, [midi, ...]), ...]. Returns length in seconds."""
    if not events:
        raise ValueError("No notes to render.")
    total = max(start + dur for start, dur, _ in events) + RELEASE + 0.5
    audio = np.zeros(int(total * sr))
    for start, dur, pitches in events:
        begin = int(start * sr)
        for midi in pitches:
            tone = render_note(midi_to_hz(midi), dur, sr) * (0.6 / np.sqrt(len(pitches)))
            audio[begin:begin + len(tone)] += tone[: len(audio) - begin]

    peak = np.max(np.abs(audio)) or 1.0
    audio = (audio / peak * 0.9 * 32767).astype(np.int16)
    wavfile.write(path, sr, audio)
    return total
