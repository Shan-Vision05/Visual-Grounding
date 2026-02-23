import torch
from torchvision.ops import box_iou
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for cluster / headless use
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import BertTokenizer


# Module-level tokenizer singleton (avoids re-downloading every call)
_TOKENIZER: BertTokenizer | None = None


def _get_tokenizer() -> BertTokenizer:
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
    return _TOKENIZER


# ---------------------------------------------------------------------------
def GetScores(boxes: torch.Tensor, label_box: torch.Tensor, image_size: int = 512) -> int:
    """Return the index of the proposal box with highest IoU to *label_box*."""
    true_box = label_box.clone() * image_size
    # label_box is [x, y, w, h] normalised → convert to [x1, y1, x2, y2]
    true_box[2] += true_box[0]
    true_box[3] += true_box[1]
    ious = box_iou(boxes, true_box.unsqueeze(0)).squeeze(1)
    return torch.argmax(ious).item()


def CreateBatchLabels(
    proposals: list[torch.Tensor], batch_boxes: torch.Tensor, image_size: int = 512,
) -> torch.Tensor:
    """Create a (B,) tensor of ground-truth proposal indices for each sample."""
    batch_size = len(proposals)
    indices = []
    for i in range(batch_size):
        n = min(len(proposals[i]), 10)
        boxes = proposals[i][:n]
        indices.append(GetScores(boxes, batch_boxes[i], image_size=image_size))
    # Pad if fewer proposals than expected (edge case)
    while len(indices) < batch_size:
        indices.append(0)
    return torch.tensor(indices)


def compute_acc_at_iou(
    proposals: list[torch.Tensor],
    pred_indices: torch.Tensor,
    gt_boxes_norm: torch.Tensor,
    image_size: int = 512,
    threshold: float = 0.5,
) -> float:
    """Compute Acc@threshold: fraction of predictions with IoU >= threshold to GT.

    This is the standard visual grounding metric (Acc@0.5).
    """
    batch_size = len(proposals)
    correct = 0
    for i in range(batch_size):
        pred_idx = pred_indices[i].item()
        n = min(len(proposals[i]), 10)
        if pred_idx >= n:
            continue
        pred_box = proposals[i][pred_idx].unsqueeze(0)  # (1, 4) x1,y1,x2,y2

        # GT: [x, y, w, h] normalised → [x1, y1, x2, y2] in image coords
        gt = gt_boxes_norm[i] * image_size
        gt_xyxy = torch.stack([gt[0], gt[1], gt[0] + gt[2], gt[1] + gt[3]]).unsqueeze(0)

        iou = box_iou(pred_box, gt_xyxy.to(pred_box.device)).item()
        if iou >= threshold:
            correct += 1
    return correct / max(batch_size, 1) * 100.0


# ---------------------------------------------------------------------------
def plot_region_with_text(
    batch_imgs: torch.Tensor,
    batch_ids: torch.Tensor,
    batch_boxes,
    result: list[torch.Tensor],
    predict: bool = False,
    save_path: str | None = None,
):
    """Visualise predicted or ground-truth boxes overlaid on images.

    When *save_path* is given the figure is saved to disk instead of shown
    (useful for headless cluster runs).
    """
    indices = batch_boxes if predict else CreateBatchLabels(result, batch_boxes)
    tokenizer = _get_tokenizer()

    for sample, idx in enumerate(indices):
        decoded = tokenizer.decode(
            batch_ids[sample][0], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        fig, ax = plt.subplots(1, 1)
        ax.imshow(batch_imgs[sample].permute(1, 2, 0).cpu().numpy())
        ax.set_title(decoded)

        bbox = result[sample][idx]
        box = bbox.cpu().numpy()
        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)

        if save_path:
            fig.savefig(f"{save_path}_sample{sample}.png", bbox_inches="tight")
        else:
            plt.show()
        plt.close(fig)
