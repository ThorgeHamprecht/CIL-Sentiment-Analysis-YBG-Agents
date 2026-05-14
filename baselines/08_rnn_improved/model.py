"""Improved BiLSTM with attention pooling for sentiment rating prediction."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBiLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 384,
        num_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

        # Attention pooling: learns which timesteps matter for classification
        self.attn = nn.Linear(hidden_dim * 2, 1, bias=False)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.dropout(self.embedding(x))
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # out: (B, L, hidden*2)

        # Masked attention: ignore PAD positions
        mask = torch.arange(out.size(1), device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        scores = self.attn(out).squeeze(-1)  # (B, L)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = F.softmax(scores, dim=1).unsqueeze(-1)  # (B, L, 1)
        context = (weights * out).sum(dim=1)  # (B, hidden*2)

        return self.classifier(self.dropout(context))
