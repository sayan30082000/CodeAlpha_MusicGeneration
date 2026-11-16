"""Step 5: generate new music with the trained LSTM and save it as MIDI + WAV.

    python generate.py                          # 1 piece, 160 notes
    python generate.py --count 3 --length 200   # 3 longer pieces
    python generate.py --temperature 1.2        # more adventurous (0.6 = safer)
    python generate.py --play                   # open the result in your media player

Output goes to output/ as generated_<time>.mid and .wav
"""
import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import torch

from model import MusicLSTM
from music_utils import END_TOKEN, tokens_to_events, tokens_to_stream
from synth import render_events

ROOT = Path(__file__).parent
MODEL_FILE = ROOT / "models" / "music_lstm.pt"
NOTES_FILE = ROOT / "data" / "notes.json"
OUTPUT_DIR = ROOT / "output"


def load_model():
    if not MODEL_FILE.exists():
        raise SystemExit("models/music_lstm.pt not found — run `python train.py` first.")
    ckpt = torch.load(MODEL_FILE, map_location="cpu")
    model = MusicLSTM(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["vocab"], ckpt["seq_len"]


def pick_seed(vocab: list[str], seq_len: int) -> list[str]:
    """A random snippet from the training data to start the melody."""
    pieces = json.loads(NOTES_FILE.read_text(encoding="utf-8"))["pieces"]
    known = set(vocab)
    candidates = [p for p in pieces if len(p) > seq_len]
    piece = random.choice(candidates or pieces)
    start = random.randint(0, max(0, len(piece) - seq_len))
    return [t for t in piece[start:start + seq_len] if t in known]


@torch.no_grad()
def generate(model, vocab, seed_tokens, length, temperature=1.0, top_k=8) -> list[str]:
    token_to_id = {t: i for i, t in enumerate(vocab)}
    end_id = token_to_id[END_TOKEN]

    ids = torch.tensor([[token_to_id[t] for t in seed_tokens]])
    logits, hidden = model(ids)          # warm up the LSTM memory with the seed
    next_logits = logits[0, -1]

    generated = []
    for _ in range(length):
        next_logits = next_logits / max(temperature, 1e-3)
        next_logits[end_id] = float("-inf")             # keep going, don't end the piece
        if top_k:
            cutoff = torch.topk(next_logits, min(top_k, len(vocab))).values[-1]
            next_logits[next_logits < cutoff] = float("-inf")
        probs = torch.softmax(next_logits, dim=-1)
        token_id = torch.multinomial(probs, 1).item()
        generated.append(vocab[token_id])

        logits, hidden = model(torch.tensor([[token_id]]), hidden)
        next_logits = logits[0, -1]
    return generated


def open_file(path: Path) -> None:
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606 - opens with the default media player
    elif sys.platform == "darwin":
        os.system(f'open "{path}"')
    else:
        os.system(f'xdg-open "{path}"')


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--length", type=int, default=160, help="number of notes/chords to generate")
    p.add_argument("--count", type=int, default=1, help="how many pieces to generate")
    p.add_argument("--temperature", type=float, default=1.0, help="randomness (0.5 = safe, 1.3 = wild)")
    p.add_argument("--top-k", type=int, default=8, help="sample only from the k most likely tokens")
    p.add_argument("--bpm", type=int, default=90, help="tempo of the output")
    p.add_argument("--seed", type=int, default=None, help="random seed for reproducible output")
    p.add_argument("--no-wav", action="store_true", help="only write the MIDI file")
    p.add_argument("--play", action="store_true", help="open the result after generating")
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    model, vocab, seq_len = load_model()
    OUTPUT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    last = None
    for n in range(1, args.count + 1):
        seed_tokens = pick_seed(vocab, seq_len)
        tokens = generate(model, vocab, seed_tokens, args.length, args.temperature, args.top_k)

        name = f"generated_{stamp}" + (f"_{n}" if args.count > 1 else "")
        midi_path = OUTPUT_DIR / f"{name}.mid"
        tokens_to_stream(tokens, bpm=args.bpm).write("midi", fp=str(midi_path))
        print(f"[{n}] MIDI  -> {midi_path}")
        print("    first notes:", " ".join(tokens[:12]), "...")

        last = midi_path
        if not args.no_wav:
            wav_path = OUTPUT_DIR / f"{name}.wav"
            seconds = render_events(tokens_to_events(tokens, bpm=args.bpm), str(wav_path))
            print(f"    Audio -> {wav_path}  ({seconds:.0f} s)")
            last = wav_path

    if args.play and last:
        open_file(last)


if __name__ == "__main__":
    main()
