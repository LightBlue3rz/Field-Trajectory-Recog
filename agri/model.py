"""Agri-STGLNet and optional encoder variants, using only PyTorch.

Public input: [batch, graph_count, 160, features], plus a boolean point mask.
Output: one [batch, 3] logits tensor, irrespective of trajectory length.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

VARIANTS = ("full", "no-local", "no-global", "no-bilstm", "bilstm", "cnn", "tcn")


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int = 8
    num_classes: int = 3
    graph_size: int = 160
    local_nodes: int = 8
    hidden_dim: int = 64
    lstm_layers: int = 1
    dropout: float = 0.1
    variant: str = "full"

    def __post_init__(self) -> None:
        for name in ("input_dim", "num_classes", "graph_size", "local_nodes", "hidden_dim", "lstm_layers"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.graph_size % self.local_nodes:
            raise ValueError("graph_size must be divisible by local_nodes")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")


class ValidBatchNorm(nn.Module):
    """Batch normalization over real points; padding never updates statistics."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # values [..., channels], mask [...]. A single point uses running stats.
        valid = values[mask]
        if self.training and valid.shape[0] == 1:
            normalized = F.batch_norm(
                valid,
                self.norm.running_mean,
                self.norm.running_var,
                self.norm.weight,
                self.norm.bias,
                training=False,
                eps=self.norm.eps,
            )
        else:
            normalized = self.norm(valid)
        result = torch.zeros_like(values)
        result[mask] = normalized
        return result


class ChainGraphConv(nn.Module):
    """D^-1/2 (A+I) D^-1/2 X W for a bidirectional trajectory chain.

    Each batch item is an independent graph chain. The three diagonals avoid
    allocating a dense adjacency matrix.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, values: torch.Tensor, mask: torch.Tensor, segment_ids=None) -> torch.Tensor:
        values = values.masked_fill(~mask[..., None], 0)
        connected = mask[:, 1:] & mask[:, :-1]
        if segment_ids is not None:
            connected = connected & (segment_ids[:, 1:] == segment_ids[:, :-1])
        degree = mask.to(values.dtype)
        degree = degree + F.pad(connected, (1, 0)) + F.pad(connected, (0, 1))
        inverse = degree.clamp_min(1).rsqrt().unsqueeze(-1)
        projected = self.projection(values) * inverse
        left = F.pad(projected[:, :-1] * connected[..., None], (0, 0, 1, 0))
        right = F.pad(projected[:, 1:] * connected[..., None], (0, 0, 0, 1))
        return ((projected + left + right) * inverse).masked_fill(~mask[..., None], 0)


class GlobalEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ChainGraphConv(config.input_dim, config.hidden_dim),
                ChainGraphConv(config.hidden_dim, config.hidden_dim),
            ]
        )
        self.norms = nn.ModuleList([ValidBatchNorm(config.hidden_dim) for _ in self.layers])
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, features: torch.Tensor, mask: torch.Tensor, segment_ids=None) -> torch.Tensor:
        batch, graphs, points, _ = features.shape
        # Each fixed window is a separate graph; BiLSTM links graph features later.
        values = features.reshape(batch * graphs, points, -1)
        flat_mask = mask.reshape(batch * graphs, points)
        segments = segment_ids.reshape(batch * graphs, points) if segment_ids is not None else None
        for layer, norm in zip(self.layers, self.norms):
            values = self.dropout(F.relu(norm(layer(values, flat_mask, segments), flat_mask)))
        values = values.reshape(batch, graphs, points, -1)
        # Pool valid nodes into one representation per graph.
        return values.sum(dim=2) / mask.sum(dim=2, keepdim=True).clamp_min(1)


class LocalGraphBlock(nn.Module):
    """Learnable T/A matrices, Conv2d, BN, residual and PReLU.

    Fold a 160-point graph into T=20 consecutive groups of V=8 actual
    trajectory points. Features form the C axis: [graphs, C, T, V].
    """

    def __init__(self, input_dim: int, output_dim: int, config: ModelConfig):
        super().__init__()
        steps = config.graph_size // config.local_nodes
        self.time_matrix = nn.Parameter(torch.empty(config.local_nodes, steps, steps))
        self.space_matrix = nn.Parameter(torch.empty(steps, config.local_nodes, config.local_nodes))
        nn.init.normal_(self.time_matrix, std=(2 / steps) ** 0.5)
        nn.init.normal_(self.space_matrix, std=(2 / config.local_nodes) ** 0.5)
        self.convolution = nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1, bias=False)
        self.norm = ValidBatchNorm(output_dim)
        self.residual = (
            nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False)
            if input_dim != output_dim
            else nn.Identity()
        )
        self.activation = nn.PReLU(output_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values = values.masked_fill(~mask[:, None], 0)
        residual = self.residual(values)
        transformed = torch.einsum("nctv,vtq->ncqv", values, self.time_matrix)
        transformed = transformed.masked_fill(~mask[:, None], 0)
        transformed = torch.einsum("nctv,tvw->nctw", transformed, self.space_matrix)
        transformed = transformed.masked_fill(~mask[:, None], 0)
        convolved = self.convolution(transformed).permute(0, 2, 3, 1)
        normalized = self.norm(convolved, mask).permute(0, 3, 1, 2)
        return self.dropout(self.activation(normalized + residual)).masked_fill(~mask[:, None], 0)


class LocalEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList(
            [
                LocalGraphBlock(config.input_dim, config.hidden_dim // 2 or 1, config),
                LocalGraphBlock(config.hidden_dim // 2 or 1, config.hidden_dim, config),
            ]
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, graphs, _, channels = features.shape
        graph_mask = mask.any(dim=-1).flatten()
        # Omit whole padded graphs to keep memory proportional to real data.
        values = features.reshape(-1, self.config.graph_size, channels)[graph_mask]
        steps = self.config.graph_size // self.config.local_nodes
        values = values.reshape(-1, steps, self.config.local_nodes, channels).permute(0, 3, 1, 2)
        point_mask = mask.reshape(-1, steps, self.config.local_nodes)[graph_mask]
        for block in self.blocks:
            values = block(values, point_mask)
        pooled = values.sum(dim=(2, 3)) / point_mask.sum(dim=(1, 2))[:, None]
        output = features.new_zeros(batch * graphs, self.config.hidden_dim)
        output[graph_mask] = pooled
        return output.reshape(batch, graphs, -1)


def _recurrent_summary(lstm: nn.LSTM, values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    packed = pack_padded_sequence(values, lengths.cpu(), batch_first=True, enforce_sorted=False)
    _, (hidden, _) = lstm(packed)
    return torch.cat((hidden[-2], hidden[-1]), dim=-1)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask[..., None]).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)


class AgriSTGLNet(nn.Module):
    """One fused representation per graph, then BiLSTM across the entire file."""

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        config = self.config
        if config.variant in {"bilstm", "cnn", "tcn"}:
            raise ValueError("Use build_model for sequence-only baselines")
        self.global_encoder = GlobalEncoder(config) if config.variant != "no-global" else None
        self.local_encoder = LocalEncoder(config) if config.variant != "no-local" else None
        fusion_dim = config.hidden_dim * (
            int(self.global_encoder is not None) + int(self.local_encoder is not None)
        )
        self.recurrent = None
        if config.variant != "no-bilstm":
            self.recurrent = nn.LSTM(
                fusion_dim,
                config.hidden_dim,
                config.lstm_layers,
                batch_first=True,
                bidirectional=True,
                dropout=config.dropout if config.lstm_layers > 1 else 0,
            )
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(
            2 * config.hidden_dim if self.recurrent else fusion_dim, config.num_classes
        )
        self.apply(_initialize)

    def forward(self, features: torch.Tensor, mask: torch.Tensor, segment_ids=None) -> torch.Tensor:
        _validate_inputs(features, mask, self.config)
        features = features.masked_fill(~mask[..., None], 0)
        encodings = []
        if self.global_encoder is not None:
            encodings.append(self.global_encoder(features, mask, segment_ids))
        if self.local_encoder is not None:
            encodings.append(self.local_encoder(features, mask))
        fused = torch.cat(encodings, dim=-1)
        graph_mask = mask.any(dim=-1)
        if self.recurrent is None:
            summary = _masked_mean(fused, graph_mask)
        else:
            summary = _recurrent_summary(self.recurrent, fused, graph_mask.sum(dim=1))
        return self.classifier(self.dropout(summary))


class TemporalResidual(nn.Module):
    """Causal, dilated residual convolution for the TCN comparison model."""

    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.left_padding = 2 * dilation
        self.convolutions = nn.ModuleList(
            [nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation) for _ in range(2)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        residual = values
        for convolution in self.convolutions:
            values = self.dropout(F.relu(convolution(F.pad(values, (self.left_padding, 0)))))
            values = values.masked_fill(~mask[:, None], 0)
        return F.relu(values + residual).masked_fill(~mask[:, None], 0)


class SequenceBaseline(nn.Module):
    """BiLSTM, 1D-CNN or dilated TCN over original (unwindowed) points."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        if config.variant == "bilstm":
            self.recurrent = nn.LSTM(
                config.input_dim,
                config.hidden_dim,
                config.lstm_layers,
                batch_first=True,
                bidirectional=True,
                dropout=config.dropout if config.lstm_layers > 1 else 0,
            )
            output_dim = 2 * config.hidden_dim
        else:
            self.projection = nn.Conv1d(config.input_dim, config.hidden_dim, kernel_size=1)
            if config.variant == "cnn":
                self.layers = nn.ModuleList(
                    [
                        nn.Conv1d(config.hidden_dim, config.hidden_dim, kernel_size=5, padding=2)
                        for _ in range(3)
                    ]
                )
            else:
                self.layers = nn.ModuleList(
                    [
                        TemporalResidual(config.hidden_dim, dilation, config.dropout)
                        for dilation in (1, 2, 4, 8)
                    ]
                )
            output_dim = config.hidden_dim
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(output_dim, config.num_classes)
        self.apply(_initialize)

    def forward(self, features: torch.Tensor, mask: torch.Tensor, segment_ids=None) -> torch.Tensor:
        _validate_inputs(features, mask, self.config)
        values = features.flatten(1, 2)
        point_mask = mask.flatten(1, 2)
        # Graph padding may occur between continuous segments. Compact real points.
        lengths = point_mask.sum(dim=1)
        compact = values.new_zeros(len(values), int(lengths.max()), values.shape[-1])
        for index in range(len(values)):
            compact[index, : lengths[index]] = values[index, point_mask[index]]
        values = compact
        point_mask = torch.arange(values.shape[1], device=values.device)[None, :] < lengths[:, None]
        values = values.masked_fill(~point_mask[..., None], 0)
        if self.config.variant == "bilstm":
            summary = _recurrent_summary(self.recurrent, values, point_mask.sum(dim=1))
        else:
            values = self.projection(values.transpose(1, 2)).masked_fill(~point_mask[:, None], 0)
            for layer in self.layers:
                if self.config.variant == "tcn":
                    values = layer(values, point_mask)
                else:
                    values = self.dropout(F.relu(layer(values))).masked_fill(~point_mask[:, None], 0)
            summary = _masked_mean(values.transpose(1, 2), point_mask)
        return self.classifier(self.dropout(summary))


def _initialize(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LSTM):
        for name, parameter in module.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(parameter, nonlinearity="relu")
            else:
                nn.init.zeros_(parameter)


def _validate_inputs(features: torch.Tensor, mask: torch.Tensor, config: ModelConfig) -> None:
    if features.ndim != 4 or features.shape[2:] != (config.graph_size, config.input_dim):
        raise ValueError("features must be [batch, graph_count, graph_size, input_dim]")
    if mask.dtype != torch.bool or mask.shape != features.shape[:-1]:
        raise ValueError("mask must be boolean with shape features.shape[:-1]")
    point_mask = mask.flatten(1, 2)
    if not point_mask.any(dim=1).all():
        raise ValueError("Every trajectory must contain at least one valid point")
    if ((~mask[..., :-1]) & mask[..., 1:]).any():
        raise ValueError("Padding must follow valid points within each graph")
    graph_mask = mask.any(dim=-1)
    if ((~graph_mask[:, :-1]) & graph_mask[:, 1:]).any():
        raise ValueError("Empty graphs must follow valid graphs")


def build_model(config: ModelConfig) -> nn.Module:
    if config.variant in {"bilstm", "cnn", "tcn"}:
        return SequenceBaseline(config)
    return AgriSTGLNet(config)
