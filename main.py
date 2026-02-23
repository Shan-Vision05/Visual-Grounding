"""Visual Grounding – training entry-point.

Usage examples
--------------
  # Local
  python main.py --data_dir ./data --epochs 40 --batch_size 16

  # On CU Blanca cluster
  module load slurm/blanca && sbatch run_training.sbatch
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys

import numpy as np
import torch

from models.ModCoAttnModels import VisualGrounding
from Trainer import VisualGroundingTrainer
from utils.Dataset import GetDataloader

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
# Reproducibility
# ---------------------------------------------------------------------------
def seed_everything(seed: int = 42, fast: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if fast:
        # Maximise throughput (H100 / A100) — minor non-determinism
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Visual Grounding model")
    p.add_argument("--data_dir", type=str, default="./data",
                    help="Root directory containing train/ and test/ sub-dirs")
    p.add_argument("--output_dir", type=str, default="./outputs",
                    help="Where to save checkpoints and logs")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8,
                    help="Early-stopping patience (epochs)")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--amp", action="store_true", default=True,
                    help="Use automatic mixed precision (default: on)")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--max_samples", type=int, default=None,
                    help="Cap train/test samples (default: use full dataset)")
    p.add_argument("--fast", action="store_true", default=False,
                    help="Enable TF32 + cuDNN benchmark for max GPU throughput")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    seed_everything(args.seed, fast=args.fast)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # --- data ---
    train_dir = os.path.join(args.data_dir, "train")
    test_dir = os.path.join(args.data_dir, "test")
    train_loader = GetDataloader(
        train_dir, batch_size=args.batch_size, split="train",
        num_workers=args.num_workers, max_samples=args.max_samples,
        image_size=args.image_size,
    )
    test_loader = GetDataloader(
        test_dir, batch_size=args.batch_size, split="test",
        num_workers=args.num_workers, image_size=args.image_size,
    )
    logger.info("Train batches: %d | Test batches: %d", len(train_loader), len(test_loader))

    # --- model & trainer ---
    model = VisualGrounding(image_size=args.image_size)
    trainer = VisualGroundingTrainer(
        model, device, train_loader, test_loader,
        lr=args.lr, weight_decay=args.weight_decay,
        use_amp=args.amp, image_size=args.image_size,
        epochs=args.epochs,
    )

    # --- training loop ---
    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = trainer.train_step(epoch)
        val_loss, val_acc = trainer.eval_step(epoch)

        logger.info(
            "Epoch %3d | TrainLoss %.4f | TrainAcc %.2f%% | ValLoss %.4f | ValAcc %.2f%% | LR %.2e",
            epoch, train_loss, train_acc, val_loss, val_acc, trainer.get_lr(),
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
            ckpt_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(best_state, ckpt_path)
            logger.info("  ✓ Saved best checkpoint (val_loss=%.4f)", val_loss)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            logger.info("Early stopping triggered after %d epochs without improvement.", args.patience)
            break

    logger.info("Training complete. Best val loss: %.4f", best_val_loss)


if __name__ == "__main__":
    main()