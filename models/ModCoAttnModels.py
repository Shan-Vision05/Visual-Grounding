import torch
import torch.nn as nn
from transformers import BertModel

from .TextModels import TextProjectionHead
from .VisionModels import VisionEncoder, LocationVisionEncoder, RelationshipVisionEncoder


class CoAttentionScorer(nn.Module):
    """Three-branch bidirectional co-attention fusion scoring module.

    Branches:
        1. Attribute  — visual RoI features ↔ subject text
        2. Location   — normalised box geometry ↔ location text
        3. Relationship — Δ-neighbour features ↔ relationship text
    """

    def __init__(self, feature_dim: int = 512, nheads: int = 2):
        super().__init__()

        # --- per-branch cross-attention ---
        self.cross_attn_img = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=nheads, dropout=0.1, batch_first=True
        )
        self.cross_attn_loc = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=nheads, dropout=0.1, batch_first=True
        )
        self.cross_attn_rel = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=nheads, dropout=0.1, batch_first=True
        )

        self.norm_img = nn.LayerNorm(feature_dim)
        self.norm_loc = nn.LayerNorm(feature_dim)
        self.norm_rel = nn.LayerNorm(feature_dim)

        # --- learnable branch weights ---
        self.w_img = nn.Parameter(torch.tensor(1.0 / 3))
        self.w_loc = nn.Parameter(torch.tensor(1.0 / 3))
        self.w_rel = nn.Parameter(torch.tensor(1.0 / 3))

        # --- classifier (3 * feature_dim → 1) ---
        self.classifier = nn.Sequential(
            nn.Linear(3 * feature_dim, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, feature_dim // 4),
            nn.GELU(),
            nn.LayerNorm(feature_dim // 4),
            nn.Linear(feature_dim // 4, 1),
        )

    def forward(
        self,
        text_img_feats: torch.Tensor,
        text_loc_feats: torch.Tensor,
        text_rel_feats: torch.Tensor,
        img_feats: torch.Tensor,
        loc_feats: torch.Tensor,
        rel_feats: torch.Tensor,
        text_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        obj_feats = img_feats.mean(dim=[3, 4])  # (B, N, C)

        # text_*_feats: (B, seq_len, C) — full token sequence
        # key_padding_mask: (B, seq_len), True = padded position
        attn_img, _ = self.cross_attn_img(
            query=obj_feats, key=text_img_feats, value=text_img_feats,
            key_padding_mask=text_pad_mask,
        )
        attn_img = self.norm_img(attn_img + obj_feats)

        attn_loc, _ = self.cross_attn_loc(
            query=loc_feats, key=text_loc_feats, value=text_loc_feats,
            key_padding_mask=text_pad_mask,
        )
        attn_loc = self.norm_loc(attn_loc + loc_feats)

        attn_rel, _ = self.cross_attn_rel(
            query=rel_feats, key=text_rel_feats, value=text_rel_feats,
            key_padding_mask=text_pad_mask,
        )
        attn_rel = self.norm_rel(attn_rel + rel_feats)

        weights = torch.softmax(
            torch.stack([self.w_img, self.w_loc, self.w_rel]), dim=0
        )
        fused = torch.cat(
            [weights[0] * attn_img, weights[1] * attn_loc, weights[2] * attn_rel],
            dim=-1,
        )

        return self.classifier(fused).squeeze(-1)


class VisualGrounding(nn.Module):
    """Full visual grounding model with three reasoning branches.

    Architecture:
        Vision:  Faster R-CNN proposals → attribute (RoI), location (box geom),
                 relationship (Δ-neighbour) features
        Text:    Shared frozen BERT backbone → three projection heads
        Fusion:  Bidirectional co-attention per branch → weighted concatenation → MLP scorer
    """

    def __init__(self, image_size: int = 512):
        super().__init__()

        # Shared frozen BERT backbone (single forward pass for all branches)
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for p in self.bert.parameters():
            p.requires_grad = False

        # Vision
        self.vision_encoder = VisionEncoder(input_size=(image_size, image_size))
        self.loc_vision_encoder = LocationVisionEncoder(image_size=image_size)
        self.rel_vision_encoder = RelationshipVisionEncoder(image_size=image_size)

        # Text projection heads (lightweight, trainable)
        self.subject_proj = TextProjectionHead(in_dim=768, out_dim=512)
        self.loc_proj = TextProjectionHead(in_dim=768, out_dim=512)
        self.rel_proj = TextProjectionHead(in_dim=768, out_dim=512)

        # Fusion
        self.co_attn_scorer = CoAttentionScorer()

    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor,
    ):
        # --- shared BERT forward (single pass, frozen) ---
        with torch.no_grad():
            bert_seq = self.bert(
                input_ids.squeeze(1),
                attention_mask=attn_mask.squeeze(1),
            ).last_hidden_state  # (B, seq_len, 768)

        # Padding mask for cross-attention (True = pad token)
        text_pad_mask = (attn_mask.squeeze(1) == 0)

        # --- branch-specific projections (B, seq_len, 768) → (B, seq_len, 512) ---
        text_img = self.subject_proj(bert_seq)
        text_loc = self.loc_proj(bert_seq)
        text_rel = self.rel_proj(bert_seq)

        # --- vision ---
        proposals, img_feats = self.vision_encoder(image)
        loc_feats = self.loc_vision_encoder(proposals)
        rel_feats = self.rel_vision_encoder(proposals)

        scores = self.co_attn_scorer(
            text_img, text_loc, text_rel,
            img_feats, loc_feats, rel_feats,
            text_pad_mask=text_pad_mask,
        )
        return proposals, scores