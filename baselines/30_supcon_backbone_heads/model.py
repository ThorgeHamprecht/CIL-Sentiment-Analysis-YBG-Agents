"""SupCon-pretrained mDeBERTa backbone with frozen/fine-tuned heads."""
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


class EMA:
    """Exponential moving average helper for trainable weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    def update(self):
        """Update the shadow weights from the model's current parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self):
        """Temporarily swap EMA weights into the model."""
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore live weights after an EMA evaluation or save."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


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
        """Return scalar SupCon loss, skipping anchors without positives."""
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
            denominator = (exp_logits * weights).sum(dim=1, keepdim=True).clamp(min=self.eps)

        log_prob = logits - torch.log(denominator)
        positives_per_anchor = positive_mask.sum(dim=1)
        valid_anchor = positives_per_anchor > 0
        if valid_anchor.sum() == 0:
            return embeddings.new_tensor(0.0)
        loss_per_anchor = -(log_prob * positive_mask.float()).sum(dim=1) / positives_per_anchor.clamp(min=1)
        return loss_per_anchor[valid_anchor].mean()


class PureContrastiveMDeBERTa(nn.Module):
    """mDeBERTa encoder plus projection head used only for SupCon pretraining."""

    def __init__(self, model_name: str = "microsoft/mdeberta-v3-base", projection_dim: int = 128, dropout: float = 0.1):
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
        """Return normalized projection embeddings and pooled backbone states."""
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pool(out.last_hidden_state, attention_mask)
        projection = self.contrastive_head(pooled)
        return {
            "embeddings": F.normalize(projection, p=2, dim=-1),
            "pooled": pooled,
        }


class CORALHead(nn.Module):
    """Linear CORAL ordinal-regression head for pooled frozen features."""

    def __init__(self, hidden_size: int, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(hidden_size, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return K-1 threshold logits for pooled features."""
        return self.fc(features) + self.bias


def coral_loss(logits: torch.Tensor, labels: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    """Binary cross-entropy over CORAL ordinal thresholds."""
    targets = torch.stack([labels > k for k in range(num_classes - 1)], dim=1).float()
    return F.binary_cross_entropy_with_logits(logits, targets)


def coral_decode(logits: torch.Tensor) -> torch.Tensor:
    """Decode CORAL logits by counting passed thresholds."""
    return (torch.sigmoid(logits) > 0.5).sum(dim=1).long()


class MDeBERTaRegressor(nn.Module):
    """SupCon-initialized mDeBERTa with either linear or one-hidden-layer regression head."""

    def __init__(
        self,
        model_name: str = "microsoft/mdeberta-v3-base",
        head_type: str = "linear",
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        if head_type not in {"linear", "mlp"}:
            raise ValueError(f"Unknown regression head_type: {head_type}")
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        if head_type == "linear":
            self.regression_head = nn.Linear(hidden, 1)
        else:
            self.regression_head = nn.Sequential(
                nn.Linear(hidden, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, input_ids, attention_mask):
        """Return scalar rating predictions on the 0..4 scale."""
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pool(out.last_hidden_state, attention_mask)
        return self.regression_head(self.dropout(pooled)).squeeze(-1)


def regression_loss(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean-squared error between scalar predictions and ordinal labels."""
    return F.mse_loss(preds.float(), labels.float())


def regression_decode(preds: torch.Tensor) -> torch.Tensor:
    """Round scalar predictions and clamp them to valid label ids."""
    return torch.round(preds.float()).clamp(0, 4).long()
