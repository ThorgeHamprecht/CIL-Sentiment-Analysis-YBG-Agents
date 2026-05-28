"""Bidirectional LSTM classifier for sentiment rating prediction — EMD² loss + median decode."""
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


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
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
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.dropout(self.embedding(x))
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        # h_n: (num_layers*2, batch, hidden_dim) — take last layer, both directions
        h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.classifier(self.dropout(h))
