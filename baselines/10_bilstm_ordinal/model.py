"""Two-stream BiLSTM with gated fusion, OLL loss, and EV decode."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def oll_loss(logits: torch.Tensor, labels: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    """Ordinal Log-Loss (Castagnos et al., COLING 2022).

    L = CE(logits, y) + Σ_{i≠y} |y-i|^α · (-log(1 - p_i))
    Penalises wrong predictions in proportion to their ordinal distance.
    """
    B, C = logits.shape
    probs = F.softmax(logits, dim=1)
    classes = torch.arange(C, device=logits.device, dtype=torch.float)
    distances = torch.abs(labels.float().unsqueeze(1) - classes)  # (B, C)
    wrong = (distances > 0).float()
    penalty = (distances.pow(alpha) * wrong * (-torch.log(1 - probs + 1e-8))).sum(1).mean()
    return F.cross_entropy(logits, labels) + penalty


def ev_decode(logits: torch.Tensor) -> torch.Tensor:
    """Bayes-optimal decoder under MAE: E[class] rounded to nearest integer."""
    probs = F.softmax(logits, dim=1)
    classes = torch.arange(probs.shape[1], device=probs.device, dtype=torch.float)
    return (probs * classes).sum(1).round().long().clamp(0, probs.shape[1] - 1)


class TwoStreamBiLSTM(nn.Module):
    """
    Shared-weight BiLSTM encodes title and body independently.
    A learned sigmoid gate fuses the two representations:
        h = σ(W[h_title; h_body]) * h_title + (1 − σ) * h_body
    An auxiliary title-only head provides multi-task supervision during training.
    """

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
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.Linear(hidden_dim * 2, 1, bias=False)

        h = hidden_dim * 2
        self.gate = nn.Linear(h * 2, h)
        self.classifier = nn.Sequential(
            nn.Linear(h, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes),
        )
        self.title_head = nn.Linear(h, num_classes)

    def _encode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.dropout(self.embedding(x))
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        mask = torch.arange(out.size(1), device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        scores = self.attn(out).squeeze(-1).masked_fill(mask, float("-inf"))
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        return (weights * out).sum(1)

    def forward(self, x_t, l_t, x_b, l_b):
        h_t = self._encode(x_t, l_t)
        h_b = self._encode(x_b, l_b)
        g = torch.sigmoid(self.gate(torch.cat([h_t, h_b], dim=1)))
        fused = g * h_t + (1 - g) * h_b
        return self.classifier(self.dropout(fused)), self.title_head(self.dropout(h_t))
