"""mDeBERTa-v3-base: SORD loss, FGM, EMA, mean pool, two-layer head, multi-sample dropout."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

DROPOUT_SAMPLES = 5


# ── Losses ────────────────────────────────────────────────────────────────────

def sord_loss(logits: torch.Tensor, labels: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    """Soft Ordinal Regression Distribution: KL-div against exp(-|i-y|) soft targets."""
    k = torch.arange(num_classes, device=labels.device).float()
    targets = torch.exp(-torch.abs(k.unsqueeze(0) - labels.float().unsqueeze(1)))
    targets = targets / targets.sum(dim=1, keepdim=True)
    return F.kl_div(F.log_softmax(logits, dim=1), targets, reduction="batchmean")


def emd_loss(logits: torch.Tensor, labels: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    """EMD² loss (kept for ablation)."""
    probs = F.softmax(logits, dim=1)
    cdf = torch.cumsum(probs, dim=1)[:, :-1]
    targets = (labels.unsqueeze(1) <= torch.arange(num_classes - 1, device=labels.device)).float()
    return ((cdf - targets) ** 2).sum(dim=1).mean()


# ── Decoders ──────────────────────────────────────────────────────────────────

def ev_decode(logits: torch.Tensor) -> torch.Tensor:
    """Expected-value decode: Σ i·p_i, rounded and clipped."""
    probs = F.softmax(logits, dim=1)
    classes = torch.arange(logits.shape[1], device=logits.device).float()
    return (probs * classes).sum(dim=1).round().long().clamp(0, logits.shape[1] - 1)


def median_decode(logits: torch.Tensor) -> torch.Tensor:
    """Median decode (kept for ablation)."""
    cdf = torch.cumsum(F.softmax(logits, dim=1), dim=1)
    return (cdf < 0.5).sum(dim=1).clamp(0, logits.shape[1] - 1)


# ── FGM ───────────────────────────────────────────────────────────────────────

class FGM:
    """Fast Gradient Method: perturb word embeddings along gradient direction."""

    def __init__(self, model: nn.Module, eps: float = 1.0):
        self.model = model
        self.eps = eps
        self.backup: dict = {}

    def attack(self, emb_name: str = "word_embeddings"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    param.data.add_(self.eps * param.grad / norm)

    def restore(self, emb_name: str = "word_embeddings"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


# ── EMA ───────────────────────────────────────────────────────────────────────

class EMA:
    """Exponential moving average of model weights."""

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
        self.backup = {}


# ── Model ─────────────────────────────────────────────────────────────────────

class mDeBERTaAdvanced(nn.Module):
    """mDeBERTa-v3-base + mean pool + two-layer head + K=5 multi-sample dropout."""

    def __init__(self, model_name: str = "microsoft/mdeberta-v3-base", num_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size  # 768
        self.dense = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        hidden = F.gelu(self.norm(self.dense(pooled)))
        return torch.stack([self.classifier(self.dropout(hidden)) for _ in range(DROPOUT_SAMPLES)]).mean(0)
