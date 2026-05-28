"""mDeBERTa-v3-base: mean pooling, multi-sample dropout, EMD² loss, EMA."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

DROPOUT_SAMPLES = 5


def emd_loss(logits: torch.Tensor, labels: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    cdf = torch.cumsum(probs, dim=1)[:, :-1]
    k_vals = torch.arange(num_classes - 1, device=labels.device)
    targets = (labels.unsqueeze(1) <= k_vals).float()
    return ((cdf - targets) ** 2).sum(dim=1).mean()


def median_decode(logits: torch.Tensor) -> torch.Tensor:
    cdf = torch.cumsum(F.softmax(logits, dim=1), dim=1)
    return (cdf < 0.5).sum(dim=1).clamp(0, logits.shape[1] - 1)


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup: dict = {}

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


class mDeBERTaEMD(nn.Module):
    def __init__(self, model_name: str = "microsoft/mdeberta-v3-base", num_classes: int = 5, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        return torch.stack([self.classifier(self.dropout(pooled)) for _ in range(DROPOUT_SAMPLES)]).mean(0)
