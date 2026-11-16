"""Step 1 + 2: collect MIDI files and turn them into note sequences for training.

1. Collect: if data/midi/ is empty, export the Bach chorales that ship with the
   music21 corpus to MIDI files (no internet needed). You can also drop your own
   .mid files (classical, jazz, ...) into data/midi/ instead.
2. Preprocess: parse every MIDI file with music21, transpose it to C major /
   A minor (so the model learns patterns, not keys), take the melody part and
   convert each note / chord / rest to a token like "E5_1.0".
   The result is saved to data/notes.json.

    python prepare_data.py                 # all Bach chorales
    python prepare_data.py --limit 100     # fewer pieces = faster
    python prepare_data.py --part -1       # use all parts merged (for piano MIDI)
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from music21 import converter, corpus

from music_utils import END_TOKEN, element_to_token

DATA_DIR = Path(__file__).parent / "data"
MIDI_DIR = DATA_DIR / "midi"
NOTES_FILE = DATA_DIR / "notes.json"


def collect_bach_chorales(limit: int | None) -> int:
    """Export the 4-part Bach chorales from the music21 corpus as MIDI files."""
    MIDI_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for path in corpus.getComposer("bach"):
        if limit and written >= limit:
            break
        out = MIDI_DIR / f"{Path(path).stem}.mid"
        if out.exists():
            written += 1
            continue
        try:
            score = converter.parse(path)
            if len(score.parts) != 4:  # keep proper SATB chorales only
                continue
            score.write("midi", fp=str(out))
        except Exception:  # a few corpus files have malformed repeats etc.
            out.unlink(missing_ok=True)
            continue
        written += 1
        if written % 25 == 0:
            print(f"  exported {written} MIDI files...")
    return written


def transpose_to_c(score):
    """Transpose a piece to C major or A minor (by the smallest interval)."""
    key = score.analyze("key")
    target_pc = 0 if key.mode == "major" else 9  # C or A
    semitones = (target_pc - key.tonic.pitchClass) % 12
    if semitones > 6:
        semitones -= 12
    return score.transpose(semitones) if semitones else score


def midi_to_tokens(path: Path, part_index: int) -> list[str]:
    score = converter.parse(str(path))
    score = transpose_to_c(score)

    if part_index >= 0 and len(score.parts) > 0:
        source = score.parts[min(part_index, len(score.parts) - 1)]
        elements = source.flatten().notesAndRests
    else:
        elements = score.chordify().flatten().notesAndRests

    tokens = []
    for el in elements:
        tok = element_to_token(el)
        if tok is None:
            continue
        if tok.startswith("R_") and tokens and tokens[-1].startswith("R_"):
            continue  # merge consecutive rests
        tokens.append(tok)

    while tokens and tokens[0].startswith("R_"):
        tokens.pop(0)
    while tokens and tokens[-1].startswith("R_"):
        tokens.pop()
    return tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None, help="max number of pieces to use")
    parser.add_argument("--part", type=int, default=0,
                        help="which part/track to learn (0 = top voice / melody, -1 = all parts as chords)")
    args = parser.parse_args()

    midi_files = sorted(MIDI_DIR.glob("*.mid")) + sorted(MIDI_DIR.glob("*.midi"))
    if not midi_files:
        print("data/midi/ is empty -> exporting Bach chorales from the music21 corpus ...")
        collect_bach_chorales(args.limit)
        midi_files = sorted(MIDI_DIR.glob("*.mid"))
    if args.limit:
        midi_files = midi_files[: args.limit]
    print(f"Preprocessing {len(midi_files)} MIDI files ...")

    pieces = []
    for i, path in enumerate(midi_files, 1):
        try:
            tokens = midi_to_tokens(path, args.part)
        except Exception as exc:
            print(f"  skipped {path.name}: {exc}")
            continue
        if len(tokens) >= 16:
            pieces.append(tokens)
        if i % 25 == 0:
            print(f"  parsed {i}/{len(midi_files)}")

    counts = Counter(tok for piece in pieces for tok in piece)
    vocab = [END_TOKEN] + sorted(counts)
    DATA_DIR.mkdir(exist_ok=True)
    NOTES_FILE.write_text(json.dumps({"part": args.part, "vocab": vocab, "pieces": pieces}), encoding="utf-8")

    total = sum(len(p) for p in pieces)
    print(f"\nSaved {len(pieces)} pieces / {total} tokens / vocabulary of {len(vocab)} to {NOTES_FILE}")
    print("Most common tokens:", ", ".join(f"{t} ({n})" for t, n in counts.most_common(8)))


if __name__ == "__main__":
    main()
