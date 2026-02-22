"""Visual Grounding – Inference & Visualization.

Load a trained checkpoint, run on test images, and display/save the
predicted bounding box overlaid on each image alongside the referring
expression.

Usage
-----
  # Interactive (local machine with display)
  python visualize.py --checkpoint outputs/best_model.pt --data_dir ./data/test --num_samples 10

  # Headless (cluster / SSH) – saves PNGs to disk
  python visualize.py --checkpoint outputs/best_model.pt --data_dir ./data/test --save_dir ./viz_outputs

  # Use a single image + custom query
  python visualize.py --checkpoint outputs/best_model.pt --image photo.jpg --query "the red car on the left"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer

from models.ModCoAttnModels import VisualGrounding

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_model(checkpoint_path: str, image_size: int, device: torch.device) -> VisualGrounding:
    """Load trained VisualGrounding model from a state_dict checkpoint."""
    model = VisualGrounding(image_size=image_size)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info("Loaded checkpoint from %s", checkpoint_path)
    return model


def get_transforms(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def unnormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised image tensor back to [0, 1] numpy for display."""
    img = tensor.clone().cpu()
    for c in range(3):
        img[c] = img[c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def tokenize_query(text: str, tokenizer: BertTokenizer, max_length: int = 15):
    """Tokenize a single query string into model-ready tensors."""
    enc = tokenizer(
        text,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        return_tensors="pt",
    )
    return enc["input_ids"].unsqueeze(0), enc["attention_mask"].unsqueeze(0)


# ---------------------------------------------------------------------------
# Core prediction
# ---------------------------------------------------------------------------
@torch.inference_mode()
def predict(
    model: VisualGrounding,
    image_tensor: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    """Run the model on a single image and return (boxes, best_idx, scores)."""
    img = image_tensor.unsqueeze(0).to(device)
    ids = input_ids.to(device)
    mask = attention_mask.to(device)

    proposals, scores = model(img, ids, mask)
    scores = scores.squeeze(0)  # (N,)

    best_idx = scores.argmax().item()
    best_box = proposals[0][best_idx].cpu()  # (4,) — x1, y1, x2, y2

    return proposals[0].cpu(), best_idx, scores.cpu()


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def visualize_prediction(
    image_np: np.ndarray,
    query: str,
    proposals: torch.Tensor,
    best_idx: int,
    scores: torch.Tensor,
    gt_bbox: torch.Tensor | None = None,
    image_size: int = 512,
    save_path: str | None = None,
    show_top_k: int = 3,
):
    """Plot image with predicted bounding box, query text, and optional GT box."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image_np)
    ax.set_title(f'Query: "{query}"', fontsize=13, pad=10)

    # Draw top-K proposals in light blue
    probs = torch.softmax(scores[: len(proposals)], dim=0)
    sorted_idx = probs.argsort(descending=True)

    for rank, idx in enumerate(sorted_idx[:show_top_k]):
        idx = idx.item()
        if idx == best_idx:
            continue  # draw best one separately
        box = proposals[idx].numpy()
        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=1,
            edgecolor="deepskyblue",
            facecolor="none",
            linestyle="--",
            alpha=0.5,
        )
        ax.add_patch(rect)
        ax.text(
            box[0], box[1] - 4,
            f"#{rank + 1} ({probs[idx]:.2f})",
            fontsize=8, color="deepskyblue",
        )

    # Draw predicted best box in green
    best_box = proposals[best_idx].numpy()
    pred_rect = patches.Rectangle(
        (best_box[0], best_box[1]),
        best_box[2] - best_box[0],
        best_box[3] - best_box[1],
        linewidth=3,
        edgecolor="lime",
        facecolor="none",
    )
    ax.add_patch(pred_rect)
    ax.text(
        best_box[0], best_box[1] - 6,
        f"Predicted ({probs[best_idx]:.2f})",
        fontsize=10, color="lime", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6),
    )

    # Draw ground-truth box in red (if available)
    if gt_bbox is not None:
        gt = gt_bbox.clone() * image_size
        gt[2] += gt[0]  # x,y,w,h → x1,y1,x2,y2
        gt[3] += gt[1]
        gt = gt.numpy()
        gt_rect = patches.Rectangle(
            (gt[0], gt[1]),
            gt[2] - gt[0],
            gt[3] - gt[1],
            linewidth=2,
            edgecolor="red",
            facecolor="none",
            linestyle="--",
        )
        ax.add_patch(gt_rect)
        ax.text(
            gt[0], gt[3] + 14,
            "Ground Truth",
            fontsize=10, color="red", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6),
        )

    ax.axis("off")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved: %s", save_path)
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Dataset mode – run on test set
# ---------------------------------------------------------------------------
def run_on_dataset(
    model: VisualGrounding,
    data_dir: str,
    device: torch.device,
    image_size: int,
    num_samples: int,
    save_dir: str | None,
    tokenizer: BertTokenizer,
):
    """Run inference on samples from the dataset directory."""
    # Load annotations
    annotations = []
    for fname in ("annotations_1.json", "annotations_2.json"):
        path = os.path.join(data_dir, fname)
        if os.path.isfile(path):
            with open(path) as f:
                annotations.extend(json.load(f))

    if not annotations:
        logger.error("No annotation files found in %s", data_dir)
        return

    num_samples = min(num_samples, len(annotations))
    # Sample evenly across the dataset
    indices = np.linspace(0, len(annotations) - 1, num_samples, dtype=int)

    transform = get_transforms(image_size)

    for i, idx in enumerate(indices):
        ann = annotations[idx]
        img_path = os.path.join(data_dir, "images", f"{ann['image_id']}.jpg")
        if not os.path.isfile(img_path):
            logger.warning("Image not found: %s — skipping", img_path)
            continue

        pil_img = Image.open(img_path)
        img_tensor = transform(pil_img)
        img_np = unnormalize(img_tensor)

        input_ids, attention_mask = tokenize_query(ann["text"], tokenizer)
        proposals, best_idx, scores = predict(model, img_tensor, input_ids, attention_mask, device)

        gt_bbox = torch.tensor(ann["bbox"], dtype=torch.float32) / image_size

        save_path = os.path.join(save_dir, f"pred_{i:04d}.png") if save_dir else None

        visualize_prediction(
            img_np, ann["text"], proposals, best_idx, scores,
            gt_bbox=gt_bbox, image_size=image_size, save_path=save_path,
        )

    logger.info("Done! Visualised %d samples.", num_samples)


# ---------------------------------------------------------------------------
# Single-image mode
# ---------------------------------------------------------------------------
def run_single_image(
    model: VisualGrounding,
    image_path: str,
    query: str,
    device: torch.device,
    image_size: int,
    save_dir: str | None,
    tokenizer: BertTokenizer,
):
    """Run inference on a single image with a custom query."""
    if not os.path.isfile(image_path):
        logger.error("Image not found: %s", image_path)
        return

    transform = get_transforms(image_size)
    pil_img = Image.open(image_path)
    img_tensor = transform(pil_img)
    img_np = unnormalize(img_tensor)

    input_ids, attention_mask = tokenize_query(query, tokenizer)
    proposals, best_idx, scores = predict(model, img_tensor, input_ids, attention_mask, device)

    save_path = os.path.join(save_dir, "pred_single.png") if save_dir else None

    visualize_prediction(
        img_np, query, proposals, best_idx, scores,
        image_size=image_size, save_path=save_path,
    )
    logger.info("Done!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visual Grounding – Inference & Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    p.add_argument("--image_size", type=int, default=512)

    # Dataset mode
    p.add_argument("--data_dir", type=str, default=None,
                   help="Path to test data directory (with images/ and annotations_*.json)")
    p.add_argument("--num_samples", type=int, default=10,
                   help="Number of samples to visualise from dataset")

    # Single-image mode
    p.add_argument("--image", type=str, default=None,
                   help="Path to a single image for custom-query inference")
    p.add_argument("--query", type=str, default=None,
                   help="Referring expression for single-image mode")

    # Output
    p.add_argument("--save_dir", type=str, default=None,
                   help="Save visualisations to this directory (omit to show interactively)")
    p.add_argument("--show_top_k", type=int, default=3,
                   help="Show top-K proposals on the plot")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Choose backend: interactive or headless
    if args.save_dir:
        matplotlib.use("Agg")
    else:
        try:
            matplotlib.use("TkAgg")
        except ImportError:
            matplotlib.use("Agg")
            logger.warning("No display backend available; falling back to Agg. "
                           "Use --save_dir to save outputs to disk.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    model = load_model(args.checkpoint, args.image_size, device)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    if args.image and args.query:
        # Single-image mode
        run_single_image(
            model, args.image, args.query, device,
            args.image_size, args.save_dir, tokenizer,
        )
    elif args.data_dir:
        # Dataset mode
        run_on_dataset(
            model, args.data_dir, device,
            args.image_size, args.num_samples, args.save_dir, tokenizer,
        )
    else:
        logger.error("Provide either --data_dir (dataset mode) or --image + --query (single-image mode)")
        sys.exit(1)


if __name__ == "__main__":
    main()
