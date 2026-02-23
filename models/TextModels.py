import torch
import torch.nn as nn


class TextProjectionHead(nn.Module):
    """Lightweight MLP that projects BERT [CLS] features (768-d) into the
    co-attention feature space (default 512-d).

    Three separate instances are used for subject, location, and
    relationship branches so each branch learns its own text view.
    """

    def __init__(self, in_dim: int = 768, out_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
            nn.Dropout(0.1),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, cls_features: torch.Tensor) -> torch.Tensor:
        """Map BERT [CLS] output (B, 768) → (B, out_dim)."""
        return self.net(cls_features)

