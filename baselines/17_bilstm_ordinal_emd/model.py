"""Two-stream BiLSTM with gated fusion, EMD² loss, and median decode."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def emd_loss(logits: torch.Tensor, labels: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    """EMD² loss: squared sum of absolute CDF differences (Wasserstein-1 squared)."""
    probs = F.softmax(logits, dim=1)
    cdf = torch.cumsum(probs, dim=1)[:, :-1]  # (B, K-1), CDF at k=0..K-2
    k_vals = torch.arange(num_classes - 1, device=labels.device)
    targets = (labels.unsqueeze(1) <= k_vals).float()  # 1[k >= y]
    return ((cdf - targets) ** 2).sum(dim=1).mean()


def median_decode(logits: torch.Tensor) -> torch.Tensor:
    """Bayes-optimal decoder under MAE: median of the predicted distribution."""
    cdf = torch.cumsum(F.softmax(logits, dim=1), dim=1)
    return (cdf < 0.5).sum(dim=1).clamp(0, logits.shape[1] - 1)


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
