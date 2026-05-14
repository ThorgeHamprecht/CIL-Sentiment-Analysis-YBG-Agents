"""Two-stream Transformer encoder with FastText-initialised embeddings.

Same interface as 13_bilstm_fasttext but replaces the BiLSTM with a
4-layer Pre-LN Transformer encoder. d_model=300 matches the FastText
embedding dimension directly — no projection needed.

Title and body are encoded independently with a shared encoder.
A [CLS] token is prepended to each stream; its output is the
stream representation. Gated fusion and OLL/EV decode are unchanged.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.dropout(x + self.pe(positions))


class TwoStreamTransformer(nn.Module):
    """
    Shared-weight Transformer encoder for title and body streams.
    [CLS] token prepended to each stream; its output feeds the classifier.
    Gated fusion and auxiliary title head identical to 13_bilstm_fasttext.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 300,
        nhead: int = 6,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        num_classes: int = 5,
        dropout: float = 0.2,
        max_len: int = 257,   # max(64, 192) + 1 for [CLS]
        pad_idx: int = 0,
        embeddings_path: str = None,
    ):
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        if embeddings_path is not None:
            matrix = torch.tensor(np.load(embeddings_path), dtype=torch.float)
            assert matrix.shape == (vocab_size, d_model)
            self.embedding.weight.data.copy_(matrix)
            print(f"Loaded pretrained embeddings from {embeddings_path}")

        # Learned [CLS] token — separate from vocab embeddings
        self.cls_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_emb, std=0.02)

        self.pos_encoding = LearnedPositionalEncoding(d_model, max_len, dropout)
        self.scale = math.sqrt(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN: stable training from scratch
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.gate = nn.Linear(d_model * 2, d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )
        self.title_head = nn.Linear(d_model, num_classes)

    def _encode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        # Scale embeddings, prepend [CLS]
        emb = self.embedding(x) * self.scale               # (B, L, d)
        cls = self.cls_emb.expand(B, 1, -1)               # (B, 1, d)
        emb = torch.cat([cls, emb], dim=1)                 # (B, L+1, d)
        emb = self.pos_encoding(emb)

        # Padding mask: [CLS] is never padded, rest follows lengths
        pad_mask = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=x.device),
            torch.arange(L, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1),
        ], dim=1)                                           # (B, L+1)

        out = self.encoder(emb, src_key_padding_mask=pad_mask)
        out = self.norm(out)
        return out[:, 0]                                   # [CLS] representation

    def forward(self, x_t, l_t, x_b, l_b):
        h_t = self._encode(x_t, l_t)
        h_b = self._encode(x_b, l_b)
        g = torch.sigmoid(self.gate(torch.cat([h_t, h_b], dim=1)))
        fused = g * h_t + (1 - g) * h_b
        return self.classifier(self.dropout(fused)), self.title_head(self.dropout(h_t))
