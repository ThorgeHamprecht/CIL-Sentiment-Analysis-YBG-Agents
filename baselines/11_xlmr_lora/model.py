"""XLM-R-base with LoRA adapter (r=8, α=16, attention layers only)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

try:
    from peft import LoraConfig, TaskType, get_peft_model
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False


def oll_loss(logits: torch.Tensor, labels: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    B, C = logits.shape
    probs = F.softmax(logits, dim=1)
    classes = torch.arange(C, device=logits.device, dtype=torch.float)
    distances = torch.abs(labels.float().unsqueeze(1) - classes)
    wrong = (distances > 0).float()
    penalty = (distances.pow(alpha) * wrong * (-torch.log(1 - probs + 1e-8))).sum(1).mean()
    return F.cross_entropy(logits, labels) + penalty


def ev_decode(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    classes = torch.arange(probs.shape[1], device=probs.device, dtype=torch.float)
    return (probs * classes).sum(1).round().long().clamp(0, probs.shape[1] - 1)


class XLMRLoRA(nn.Module):
    def __init__(self, model_name: str = "xlm-roberta-base", num_classes: int = 5, dropout: float = 0.1):
        super().__init__()
        if not _PEFT_AVAILABLE:
            raise ImportError("peft is required: pip install peft")

        base = AutoModel.from_pretrained(model_name)
        lora_cfg = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["query", "key", "value"],
            lora_dropout=0.05,
            bias="none",
        )
        self.encoder = get_peft_model(base, lora_cfg)
        self.encoder.print_trainable_parameters()

        hidden = base.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0])
        return self.classifier(cls)
