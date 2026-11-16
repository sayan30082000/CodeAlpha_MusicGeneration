"""Export the trained LSTM for the browser version in web/ (the live demo on Netlify).

Netlify only hosts static files, so the generator runs in JavaScript:
  web/model.bin   all weights as little-endian float32
  web/model.json  vocabulary, layer sizes, where each weight tensor starts in model.bin,
                  and seed phrases taken from the Bach training data
web/lstm.js implements the same LSTM forward pass, so the browser produces the
same probabilities as PyTorch.

    python export_web.py
"""
import json
import random
from pathlib import Path

import numpy as np
import torch

from generate import load_model

ROOT = Path(__file__).parent
WEB = ROOT / "web"
NOTES_FILE = ROOT / "data" / "notes.json"
N_SEEDS = 80


def main():
    model, vocab, seq_len = load_model()
    state = model.state_dict()
    cfg = model.config

    order = ["embed.weight"]
    for layer in range(cfg["num_layers"]):
        order += [f"lstm.weight_ih_l{layer}", f"lstm.weight_hh_l{layer}",
                  f"lstm.bias_ih_l{layer}", f"lstm.bias_hh_l{layer}"]
    order += ["head.weight", "head.bias"]

    WEB.mkdir(exist_ok=True)
    tensors, offset = [], 0
    with open(WEB / "model.bin", "wb") as f:
        for name in order:
            array = state[name].detach().cpu().numpy().astype("<f4")
            f.write(array.tobytes())
            tensors.append({"name": name, "shape": list(array.shape), "offset": offset})
            offset += array.size

    # Seed phrases: random snippets of real Bach melodies to start each new piece.
    random.seed(0)
    known = set(vocab)
    pieces = [p for p in json.loads(NOTES_FILE.read_text(encoding="utf-8"))["pieces"] if len(p) > seq_len]
    seeds = []
    for piece in random.sample(pieces, min(N_SEEDS, len(pieces))):
        start = random.randint(0, len(piece) - seq_len)
        snippet = [t for t in piece[start:start + seq_len] if t in known]
        if len(snippet) == seq_len:
            seeds.append(snippet)

    ckpt = torch.load(ROOT / "models" / "music_lstm.pt", map_location="cpu")
    meta = {
        "config": cfg,
        "vocab": vocab,
        "seq_len": seq_len,
        "tensors": tensors,
        "seeds": seeds,
        "training": {"epoch": ckpt.get("epoch"), "val_loss": round(float(ckpt.get("val_loss", 0)), 3),
                     "val_acc": round(float(ckpt.get("val_acc", 0)), 3)},
    }
    (WEB / "model.json").write_text(json.dumps(meta, separators=(",", ":")), encoding="utf-8")

    size_mb = (WEB / "model.bin").stat().st_size / 1e6
    print(f"web/model.bin: {size_mb:.1f} MB ({offset:,} weights) | web/model.json: {len(vocab)} tokens, {len(seeds)} seeds")


if __name__ == "__main__":
    main()
