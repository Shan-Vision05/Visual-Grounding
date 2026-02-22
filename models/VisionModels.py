import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models._utils import IntermediateLayerGetter


# ImageNet normalization constants (shared across vision modules)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class VisionEncoder(nn.Module):
    """Extract region proposals and per-object visual features via Faster R-CNN + FPN."""

    MAX_OBJECTS = 10

    def __init__(self, input_size: tuple[int, int] = (512, 512)):
        super().__init__()
        self.input_size = input_size

        # Frozen pretrained Faster R-CNN (proposal generator only)
        self.faster_rcnn = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        self.faster_rcnn.eval()
        for p in self.faster_rcnn.parameters():
            p.requires_grad = False

        # Pre-compute denormalization buffers (dataset applies ImageNet norm,
        # but FRCNN expects [0, 1] range and applies its own normalization)
        self.register_buffer(
            "_denorm_mean",
            torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "_denorm_std",
            torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1),
        )

        body = self.faster_rcnn.backbone.body
        self.feature_extractor = IntermediateLayerGetter(
            body, return_layers={"layer2": "c3", "layer3": "c4"}
        )

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(1536, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((5, 5))

    def train(self, mode: bool = True):
        super().train(mode)
        self.faster_rcnn.eval()  # always keep frozen
        return self

    def _unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Undo ImageNet normalization so FRCNN receives [0, 1] range input."""
        return (x * self._denorm_std + self._denorm_mean).clamp(0, 1)

    def forward(self, x: torch.Tensor):
        device = x.device
        batch_size = x.size(0)
        input_h, input_w = self.input_size  # boxes are in this coordinate space

        # --- proposals (denormalize first: FRCNN expects [0,1]) ---
        with torch.no_grad():
            results = self.faster_rcnn(self._unnormalize(x))

        # --- intermediate feature maps (ImageNet-normalised input is correct here) ---
        inter = self.feature_extractor(x)
        inter["c4"] = self.upsample(inter["c4"])
        attr_blob = self.conv1x1(torch.cat([inter["c3"], inter["c4"]], dim=1))

        H, W = attr_blob.shape[-2:]
        n_obj = self.MAX_OBJECTS

        obj_blob = torch.zeros(batch_size, n_obj, 512, 5, 5, device=device)
        new_boxes: list[torch.Tensor] = []

        for i in range(batch_size):
            boxes = results[i]["boxes"][: n_obj]
            perm = torch.randperm(boxes.size(0), device=device)
            boxes = boxes[perm]
            new_boxes.append(boxes)

            for j in range(min(len(boxes), n_obj)):
                # BUG-FIX: scale from *input* coords to feature-map coords
                # (previously used FRCNN internal transform size ≈800, not 512)
                scaled = boxes[j].clone()
                scaled[0] *= W / input_w
                scaled[1] *= H / input_h
                scaled[2] *= W / input_w
                scaled[3] *= H / input_h

                x1 = scaled[0].floor().clamp(0, W - 1).long()
                y1 = scaled[1].floor().clamp(0, H - 1).long()
                x2 = scaled[2].ceil().clamp(0, W - 1).long()
                y2 = scaled[3].ceil().clamp(0, H - 1).long()
                if x2 <= x1:
                    x2 = min(x1 + 1, W)
                if y2 <= y1:
                    y2 = min(y1 + 1, H)
                obj_blob[i, j] = self.pool(attr_blob[i, :, y1:y2, x1:x2])

        return new_boxes, obj_blob


class LocationVisionEncoder(nn.Module):
    """Encode normalised bounding-box geometry (x1, y1, x2, y2, area) via a transformer."""

    MAX_OBJECTS = 10

    def __init__(self, image_size: int = 512):
        super().__init__()
        self.image_size = image_size

        self.input_proj = nn.Linear(5, 512)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=2,
            dim_feedforward=512,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.norm = nn.LayerNorm(512)

    def _process_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Normalise box coordinates and compute relative area."""
        n = min(len(boxes), self.MAX_OBJECTS)
        s = self.image_size
        out = torch.zeros(self.MAX_OBJECTS, 5, device=boxes.device)
        for i in range(n):
            b = boxes[i]
            w = (b[2] - b[0]).abs()
            h = (b[3] - b[1]).abs()
            out[i] = torch.tensor(
                [b[0] / s, b[1] / s, b[2] / s, b[3] / s, (w * h) / (s * s)],
                device=boxes.device,
            )
        return out

    def forward(self, rcnn_boxes: list[torch.Tensor]) -> torch.Tensor:
        batch_size = len(rcnn_boxes)
        device = rcnn_boxes[0].device
        loc = torch.zeros(batch_size, self.MAX_OBJECTS, 5, device=device)
        for i in range(batch_size):
            loc[i] = self._process_boxes(rcnn_boxes[i][: self.MAX_OBJECTS])

        x = self.input_proj(loc)
        x = self.transformer(x)
        return self.norm(x)


class RelationshipVisionEncoder(nn.Module):
    """Encode inter-object spatial relationships (Δcx, Δcy, Δw, Δh to K nearest neighbors).

    For each proposal, computes displacement vectors to its K nearest
    neighbours, giving relational cues like "to the left of", "above", etc.
    """

    MAX_OBJECTS = 10
    NUM_NEIGHBORS = 5

    def __init__(self, image_size: int = 512):
        super().__init__()
        self.image_size = image_size
        feat_dim = self.NUM_NEIGHBORS * 4  # 20-D per object

        self.input_proj = nn.Linear(feat_dim, 512)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=2,
            dim_feedforward=512,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.norm = nn.LayerNorm(512)

    def _compute_relationships(self, boxes: torch.Tensor) -> torch.Tensor:
        """Compute relative (Δcx, Δcy, Δw, Δh) to the K nearest neighbours."""
        device = boxes.device
        n = min(len(boxes), self.MAX_OBJECTS)
        rel = torch.zeros(self.MAX_OBJECTS, self.NUM_NEIGHBORS * 4, device=device)

        if n <= 1:
            return rel

        b = boxes[:n]
        s = self.image_size
        cx = (b[:, 0] + b[:, 2]) / 2.0 / s
        cy = (b[:, 1] + b[:, 3]) / 2.0 / s
        w = (b[:, 2] - b[:, 0]).abs() / s
        h = (b[:, 3] - b[:, 1]).abs() / s

        # Pairwise squared distances between box centres
        dx = cx.unsqueeze(1) - cx.unsqueeze(0)  # (n, n)
        dy = cy.unsqueeze(1) - cy.unsqueeze(0)
        dists = dx ** 2 + dy ** 2
        dists.fill_diagonal_(float("inf"))  # exclude self

        k = min(self.NUM_NEIGHBORS, n - 1)
        _, nn_idx = dists.topk(k, largest=False)  # (n, k)

        for i in range(n):
            for j_idx in range(k):
                j = nn_idx[i, j_idx]
                off = j_idx * 4
                rel[i, off] = cx[j] - cx[i]
                rel[i, off + 1] = cy[j] - cy[i]
                rel[i, off + 2] = w[j] - w[i]
                rel[i, off + 3] = h[j] - h[i]

        return rel

    def forward(self, rcnn_boxes: list[torch.Tensor]) -> torch.Tensor:
        batch_size = len(rcnn_boxes)
        device = rcnn_boxes[0].device
        rel = torch.zeros(
            batch_size, self.MAX_OBJECTS, self.NUM_NEIGHBORS * 4, device=device
        )
        for i in range(batch_size):
            rel[i] = self._compute_relationships(rcnn_boxes[i][: self.MAX_OBJECTS])

        x = self.input_proj(rel)
        x = self.transformer(x)
        return self.norm(x)