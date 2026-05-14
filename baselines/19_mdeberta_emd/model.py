"""mDeBERTa-v3-base with standard classification head, EMD² loss, and median decode."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


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


class mDeBERTaEMD(nn.Module):
    """mDeBERTa-v3-base with K-class softmax head, trained with EMD² loss."""

    def __init__(self, model_name: str = "microsoft/mdeberta-v3-base", num_classes: int = 5, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0])
        return self.classifier(cls)  # (B, K)
