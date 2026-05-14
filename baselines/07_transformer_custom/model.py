"""Custom Transformer encoder classifier trained from scratch."""
import math

import torch
import torch.nn as nn


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.dropout(x + self.pe(positions))


class CustomTransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        num_classes: int = 5,
        dropout: float = 0.1,
        max_len: int = 256,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = LearnedPositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.scale = math.sqrt(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for stable training from scratch
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len), padding_mask: (batch, seq_len) True=PAD
        emb = self.embedding(x) * self.scale
        emb = self.pos_encoding(emb)
        out = self.encoder(emb, src_key_padding_mask=padding_mask)
        out = self.norm(out)
        cls_repr = out[:, 0, :]  # [CLS] token at position 0
        return self.classifier(cls_repr)
