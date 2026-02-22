import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (precomputed and cached as a buffer)."""

    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        scale = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * scale)
        pe[:, 1::2] = torch.cos(position * scale)
        self.register_buffer("pe", pe)  # not a parameter, moves with .to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)]


class _BaseTextEncoder(nn.Module):
    """Shared transformer-based text encoder used by both subject and location heads."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 2,
        num_layers: int = 1,
        dim_ff: int = 512,
        max_len: int = 15,
        out_dim: int = 512,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = PositionalEncoding(d_model, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(max_len * d_model, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Linear(768, out_dim),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.token_embed(input_ids)
        x = self.pos_embed(x)

        pad_mask = (attention_mask == 0) if attention_mask is not None else None
        x = self.encoder(x, src_key_padding_mask=pad_mask)

        return self.head(x)


class SubjectTextEncoder(_BaseTextEncoder):
    """Text encoder for subject/attribute reasoning."""
    pass


class LocationTextEncoder(_BaseTextEncoder):
    """Text encoder for spatial/location reasoning."""
    pass

