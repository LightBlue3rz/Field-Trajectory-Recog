"""Point classifier with banded attention, distance decay and optional LSTM."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

POINT_VARIANTS = ("improved", "standard", "window-only", "decay-only")


@dataclass(frozen=True)
class PointModelConfig:
    input_dim: int = 8
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 3
    num_classes: int = 2
    dropout: float = 0.1
    local_window: int = 6
    time_decay: float = 0.1
    lstm_layers: int = 1
    variant: str = "improved"

    def __post_init__(self):
        if min(self.input_dim, self.d_model, self.nhead, self.num_layers, self.lstm_layers) < 1:
            raise ValueError("Model dimensions and layer counts must be positive")
        if self.d_model % self.nhead or self.d_model % 2:
            raise ValueError("d_model must be even and divisible by nhead")
        if self.local_window < 0 or not math.isfinite(self.time_decay) or self.time_decay < 0:
            raise ValueError("local_window and time_decay must be nonnegative")
        if not 0 <= self.dropout < 1 or self.variant not in POINT_VARIANTS:
            raise ValueError("Invalid dropout or point model variant")
        if self.num_classes != 2:
            raise ValueError("Point labels are fixed to turn=0 and straight=1")


class PositionalEncoding(nn.Module):
    def forward(self, values):
        length, channels = values.shape[1:]
        positions = torch.arange(length, device=values.device, dtype=values.dtype)[:, None]
        factors = torch.exp(
            torch.arange(0, channels, 2, device=values.device, dtype=values.dtype)
            * (-math.log(10000.0) / channels)
        )
        encoding = torch.zeros(length, channels, device=values.device, dtype=values.dtype)
        encoding[:, 0::2] = torch.sin(positions * factors)
        encoding[:, 1::2] = torch.cos(positions * factors)
        return values + encoding


class DistanceAttention(nn.Module):
    """Multiply scaled dot-product logits by exp(-alpha * |i-j|) before softmax.

    Local mode gathers only 2*radius+1 keys per query: no dense T-by-T tensor
    is allocated. Windows are symmetric, including future neighbors.
    """

    def __init__(self, d_model, nhead, radius=6, decay=0.1, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.radius = radius
        self.decay = decay
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, values, valid_mask):
        values = values.masked_fill(~valid_mask[..., None], 0)
        batch, length, channels = values.shape
        qkv = self.qkv(values).reshape(batch, length, 3, self.nhead, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        if self.radius is None:
            scores = torch.einsum("bihd,bjhd->bihj", query, key) / math.sqrt(self.head_dim)
            positions = torch.arange(length, device=values.device)
            distance = (positions[:, None] - positions[None, :]).abs()
            scores = scores * torch.exp(-self.decay * distance).to(scores.dtype)[None, :, None, :]
            allowed = valid_mask[:, None, None, :]
            scores = scores.masked_fill(~allowed, -torch.inf)
            scores = scores.masked_fill(~valid_mask[:, :, None, None], 0)
            weights = F.softmax(scores, dim=-1).masked_fill(~allowed, 0)
            weights = F.dropout(weights, self.dropout, self.training)
            attended = torch.einsum("bihj,bjhd->bihd", weights, value)
        else:
            offsets = torch.arange(-self.radius, self.radius + 1, device=values.device)
            neighbors = torch.arange(length, device=values.device)[:, None] + offsets[None, :]
            in_bounds = (neighbors >= 0) & (neighbors < length)
            indices = neighbors.clamp(0, length - 1)
            keys, values_local = key[:, indices], value[:, indices]
            scores = torch.einsum("bihd,biwhd->bihw", query, keys) / math.sqrt(self.head_dim)
            scores = scores * torch.exp(-self.decay * offsets.abs()).to(scores.dtype)[None, None, None, :]
            allowed = in_bounds[None, :, None, :] & valid_mask[:, indices][:, :, None, :]
            scores = scores.masked_fill(~allowed, -torch.inf)
            scores = scores.masked_fill(~valid_mask[:, :, None, None], 0)
            weights = F.softmax(scores, dim=-1).masked_fill(~allowed, 0)
            weights = F.dropout(weights, self.dropout, self.training)
            attended = torch.einsum("bihw,biwhd->bihd", weights, values_local)
        result = self.output(attended.reshape(batch, length, channels))
        return result.masked_fill(~valid_mask[..., None], 0)


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        radius = config.local_window if config.variant in {"improved", "window-only"} else None
        decay = config.time_decay if config.variant in {"improved", "decay-only"} else 0.0
        self.attention = DistanceAttention(config.d_model, config.nhead, radius, decay, config.dropout)
        self.norm_attention = nn.LayerNorm(config.d_model)
        self.norm_feedforward = nn.LayerNorm(config.d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(4 * config.d_model, config.d_model),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, values, mask):
        values = self.norm_attention(values + self.dropout(self.attention(values, mask)))
        values = values.masked_fill(~mask[..., None], 0)
        values = self.norm_feedforward(values + self.dropout(self.feedforward(values)))
        return values.masked_fill(~mask[..., None], 0)


class TransformerNodeClassifier(nn.Module):
    def __init__(self, config: PointModelConfig | None = None, **kwargs):
        super().__init__()
        self.config = config or PointModelConfig(**kwargs)
        config = self.config
        self.feature_projection = nn.Linear(config.input_dim, config.d_model)
        self.positional_encoding = PositionalEncoding()
        self.encoder = nn.ModuleList([EncoderBlock(config) for _ in range(config.num_layers)])
        self.lstm = (
            nn.LSTM(
                config.d_model,
                config.d_model,
                config.lstm_layers,
                batch_first=True,
                dropout=config.dropout if config.lstm_layers > 1 else 0,
            )
            if config.variant == "improved"
            else None
        )
        self.classifier = nn.Linear(config.d_model, config.num_classes)
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, values, valid_mask=None):
        if values.ndim != 3 or values.shape[-1] != self.config.input_dim:
            raise ValueError("Expected [batch, time, eight input features]")
        if valid_mask is None:
            valid_mask = torch.ones(values.shape[:2], device=values.device, dtype=torch.bool)
        if valid_mask.shape != values.shape[:2] or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be boolean with shape [batch, time]")
        if not valid_mask.any(dim=1).all() or ((~valid_mask[:, :-1]) & valid_mask[:, 1:]).any():
            raise ValueError("Each sequence must contain valid points followed by optional padding")
        values = values.masked_fill(~valid_mask[..., None], 0)
        values = self.positional_encoding(self.feature_projection(values))
        for layer in self.encoder:
            values = layer(values, valid_mask)
        if self.lstm is not None:
            packed = pack_padded_sequence(
                values, valid_mask.sum(dim=1).cpu(), batch_first=True, enforce_sorted=False
            )
            encoded, _ = self.lstm(packed)
            values, _ = pad_packed_sequence(encoded, batch_first=True, total_length=values.shape[1])
        return self.classifier(values).masked_fill(~valid_mask[..., None], 0)
