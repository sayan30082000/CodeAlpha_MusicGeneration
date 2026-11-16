"""The deep-learning model: an LSTM that predicts the next musical token."""
import torch
from torch import nn


class MusicLSTM(nn.Module):
    """
    token ids -> Embedding -> 2-layer LSTM -> Dropout -> Linear -> logits for the next token

    The LSTM reads the sequence one token at a time and keeps a memory (hidden state)
    of what came before, which lets it learn melodic patterns, rhythm and phrase shapes.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 96, hidden_size: int = 256,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, embed_dim=embed_dim, hidden_size=hidden_size,
                           num_layers=num_layers, dropout=dropout)
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers,
                            dropout=dropout if num_layers > 1 else 0.0, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, hidden=None):
        out, hidden = self.lstm(self.embed(x), hidden)
        return self.head(self.dropout(out)), hidden
