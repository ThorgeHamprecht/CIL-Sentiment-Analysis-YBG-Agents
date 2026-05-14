"""mDeBERTa-v3-base with CORAL ordinal regression head."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class mDeBERTaCORAL(nn.Module):
    """
    CORAL head: K-1 binary classifiers sharing one weight vector but with
    separate per-threshold biases. Prediction = number of thresholds exceeded.
    """

    def __init__(self, model_name: str = "microsoft/mdeberta-v3-base", num_classes: int = 5, dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        # Shared projection + K-1 independent bias terms
        self.fc = nn.Linear(hidden, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token (position 0)
        cls = self.dropout(out.last_hidden_state[:, 0])
        logits = self.fc(cls) + self.bias  # (B, K-1)
        return logits

    @torch.no_grad()
    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        # Count how many cumulative thresholds are exceeded
        return (torch.sigmoid(logits) > 0.5).sum(dim=1)


def coral_loss(logits: torch.Tensor, labels: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    """Binary cross-entropy across all K-1 ordinal thresholds."""
    targets = torch.stack(
        [labels > k for k in range(num_classes - 1)], dim=1
    ).float()
    return F.binary_cross_entropy_with_logits(logits, targets)
