"""mDeBERTa-v3-base contrastive encoder with mean pooling and SupCon loss."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Attention-mask-aware mean pooling over final hidden states."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


class SupervisedContrastiveLoss(nn.Module):
    """Supervised contrastive loss with optional distance-weighted negatives."""

    def __init__(self, temperature: float = 0.07, variant: str = "normal", eps: float = 1e-12):
        super().__init__()
        if variant not in {"normal", "distance_weighted"}:
            raise ValueError(f"Unknown SupCon variant: {variant}")
        self.temperature = temperature
        self.variant = variant
        self.eps = eps

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.view(-1)
        batch_size = embeddings.size(0)
        if batch_size <= 1:
            return embeddings.new_tensor(0.0)

        z = F.normalize(embeddings, p=2, dim=-1)
        logits = torch.matmul(z, z.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        device = embeddings.device
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        not_self = ~self_mask
        same_label = labels.unsqueeze(0).eq(labels.unsqueeze(1))
        positive_mask = same_label & not_self
        negative_mask = (~same_label) & not_self

        exp_logits = torch.exp(logits) * not_self.float()
        if self.variant == "normal":
            denominator = exp_logits.sum(dim=1, keepdim=True).clamp(min=self.eps)
        else:
            label_dist = (labels.unsqueeze(0) - labels.unsqueeze(1)).abs().float()
            neg_weights = (label_dist / 4.0).clamp(min=0.0, max=1.0)
            weights = torch.ones_like(exp_logits)
            weights = torch.where(negative_mask, neg_weights, weights)
            weights = torch.where(self_mask, torch.zeros_like(weights), weights)
            weighted_exp_logits = exp_logits * weights
            denominator = weighted_exp_logits.sum(dim=1, keepdim=True).clamp(min=self.eps)

        log_prob = logits - torch.log(denominator)
        positives_per_anchor = positive_mask.sum(dim=1)
        valid_anchor = positives_per_anchor > 0
        if valid_anchor.sum() == 0:
            return embeddings.new_tensor(0.0)

        loss_per_anchor = -(log_prob * positive_mask.float()).sum(dim=1) / positives_per_anchor.clamp(min=1)
        return loss_per_anchor[valid_anchor].mean()


class EMA:
    """Exponential moving average of model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

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


class PureContrastiveMDeBERTa(nn.Module):
    """mDeBERTa-v3-base with mean pooling and a contrastive projection head."""

    def __init__(
        self,
        model_name: str = "microsoft/mdeberta-v3-base",
        projection_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.contrastive_head = nn.Sequential(
            nn.Linear(hidden, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, projection_dim),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pool(out.last_hidden_state, attention_mask)
        proj = self.contrastive_head(pooled)
        embeddings = F.normalize(proj, p=2, dim=-1)
        return {
            "embeddings": embeddings,
            "pooled": pooled,
        }
