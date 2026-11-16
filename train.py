"""Step 3 + 4: build the LSTM and train it on the note sequences.

    python train.py                     # default settings
    python train.py --epochs 60         # train longer for better music

Saves the best model to models/music_lstm.pt and a loss chart to models/training_loss.png.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import MusicLSTM
from music_utils import END_TOKEN, transpose_token

ROOT = Path(__file__).parent
NOTES_FILE = ROOT / "data" / "notes.json"
MODEL_DIR = ROOT / "models"
MODEL_FILE = MODEL_DIR / "music_lstm.pt"


def make_windows(stream_ids: list[int], seq_len: int, stride: int):
    """Slice the token stream into (input, target) pairs; target = input shifted by one."""
    xs, ys = [], []
    for start in range(0, len(stream_ids) - seq_len - 1, stride):
        chunk = stream_ids[start:start + seq_len + 1]
        xs.append(chunk[:-1])
        ys.append(chunk[1:])
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def plot_losses(history: dict, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    epochs = range(1, len(history["train"]) + 1)
    best = int(np.argmin(history["val"])) + 1
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, history["train"], label="train")
    plt.plot(epochs, history["val"], label="validation")
    plt.axvline(best, color="gray", linestyle="--", linewidth=1, label=f"best model (epoch {best})")
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy loss")
    plt.title("Music LSTM training")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--augment", type=int, default=2,
                   help="data augmentation: also train on copies transposed by ±1..N semitones")
    p.add_argument("--patience", type=int, default=6, help="early stopping: epochs without improvement")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not NOTES_FILE.exists():
        raise SystemExit("data/notes.json not found — run `python prepare_data.py` first.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = json.loads(NOTES_FILE.read_text(encoding="utf-8"))

    # Hold out 10% of the pieces for validation, so the val loss measures generalisation.
    pieces = data["pieces"][:]
    random.shuffle(pieces)
    n_val = max(1, len(pieces) // 10)
    val_pieces, train_pieces = pieces[:n_val], pieces[n_val:]

    # Data augmentation: the same melodies shifted a few semitones up and down.
    # The dataset is small (~350 melodies), so this greatly reduces overfitting.
    shifts = [s for s in range(-args.augment, args.augment + 1)]
    train_pieces = [[transpose_token(t, s) for t in piece] for s in shifts for piece in train_pieces]

    vocab = [END_TOKEN] + sorted({t for piece in train_pieces + val_pieces for t in piece} - {END_TOKEN})
    token_to_id = {tok: i for i, tok in enumerate(vocab)}

    def encode(piece_list):
        ids = []
        for piece in piece_list:
            ids.extend(token_to_id[t] for t in piece)
            ids.append(token_to_id[END_TOKEN])
        return ids

    x_train, y_train = make_windows(encode(train_pieces), args.seq_len, args.stride)
    x_val, y_val = make_windows(encode(val_pieces), args.seq_len, args.stride)
    train_dl = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(TensorDataset(x_val, y_val), batch_size=args.batch_size)

    model = MusicLSTM(len(vocab), hidden_size=args.hidden, dropout=args.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    loss_fn = nn.CrossEntropyLoss()
    n_params = sum(p.numel() for p in model.parameters())

    print(f"Pieces: {len(train_pieces)} train (incl. {len(shifts)}x transposition) / {len(val_pieces)} val")
    print(f"Vocabulary: {len(vocab)} tokens | train windows: {len(x_train)} | val windows: {len(x_val)}")
    print(f"Model parameters: {n_params:,} | epochs: {args.epochs}\n")

    MODEL_DIR.mkdir(exist_ok=True)
    history = {"train": [], "val": []}
    best_val = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        total = 0.0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(xb)
        train_loss = total / len(x_train)

        model.eval()
        total, correct, count = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                logits, _ = model(xb)
                total += loss_fn(logits.reshape(-1, logits.size(-1)), yb.reshape(-1)).item() * len(xb)
                correct += (logits.argmax(-1) == yb).sum().item()
                count += yb.numel()
        val_loss = total / max(len(x_val), 1)
        val_acc = correct / max(count, 1)
        scheduler.step(val_loss)

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            epochs_without_improvement = 0
            torch.save({"state_dict": model.state_dict(), "config": model.config, "vocab": vocab,
                        "seq_len": args.seq_len, "val_loss": val_loss, "val_acc": val_acc,
                        "epoch": epoch}, MODEL_FILE)
            marker = "  ✓ saved"
        else:
            epochs_without_improvement += 1
        print(f"epoch {epoch:3d}/{args.epochs} | train loss {train_loss:.3f} | val loss {val_loss:.3f} "
              f"| val acc {val_acc:.1%} | {time.time() - start:.0f}s{marker}", flush=True)
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping: no improvement for {args.patience} epochs.")
            break

    plot_losses(history, MODEL_DIR / "training_loss.png")
    (MODEL_DIR / "history.json").write_text(json.dumps(history), encoding="utf-8")
    print(f"\nBest validation loss {best_val:.3f}. Model saved to {MODEL_FILE}")


if __name__ == "__main__":
    main()
