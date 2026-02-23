import logging
import time

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from utils.Util import CreateBatchLabels, compute_acc_at_iou

logger = logging.getLogger(__name__)


class VisualGroundingTrainer:
    """Handles training and evaluation loops for the VisualGrounding model."""

    def __init__(self, model, device, train_dataloader, test_dataloader,
                 lr=3e-4, weight_decay=1e-4, use_amp=True, image_size=512,
                 log_every: int = 20, epochs: int = 40):
        self.device = device
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.image_size = image_size
        self.log_every = log_every

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Mixed-precision training (huge speed-up on H100 / A100)
        self.use_amp = use_amp and device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        # LR schedule: linear warmup (2 epochs) + cosine decay
        warmup = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=2,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(epochs - 2, 1), eta_min=1e-6,
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer, schedulers=[warmup, cosine], milestones=[2],
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _move_text_to_device(text_batch: dict, device: torch.device) -> dict:
        """Move every tensor inside the tokenizer output dict to *device*."""
        return {k: v.to(device) for k, v in text_batch.items()}

    @staticmethod
    def _accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        return (torch.eq(y_true, y_pred).sum().item() / len(y_pred)) * 100.0

    def get_lr(self) -> float:
        """Return current learning rate."""
        return self.scheduler.get_last_lr()[0]

    # ------------------------------------------------------------------
    def train_step(self, epoch: int) -> tuple[float, float, float]:
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_iou_acc = 0.0
        n_batches = len(self.train_dataloader)
        t0 = time.time()

        for i, (X_Img, X_Text, y_bbox) in enumerate(self.train_dataloader, 1):
            X_Img = X_Img.to(self.device, non_blocking=True)
            X_Text = self._move_text_to_device(X_Text, self.device)
            y_bbox = y_bbox.to(self.device, non_blocking=True)

            with autocast("cuda", enabled=self.use_amp):
                roi, y_pred = self.model(X_Img, X_Text["input_ids"], X_Text["attention_mask"])
                y = CreateBatchLabels(roi, y_bbox, image_size=self.image_size).to(self.device)
                loss = self.loss_fn(y_pred.squeeze(-1), y.long())

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            pred_idx = y_pred.squeeze(-1).argmax(dim=1)
            total_loss += loss.item()
            total_acc += self._accuracy(y, pred_idx)
            total_iou_acc += compute_acc_at_iou(roi, pred_idx, y_bbox, self.image_size)

            if i % self.log_every == 0 or i == n_batches:
                elapsed = time.time() - t0
                avg_loss = total_loss / i
                avg_acc = total_acc / i
                avg_iou = total_iou_acc / i
                eta = elapsed / i * (n_batches - i)
                logger.info(
                    "  Epoch %2d Train [%3d/%d] | loss %.4f | acc %.1f%% | IoU@.5 %.1f%% | %.1fs | ETA %.0fs",
                    epoch, i, n_batches, avg_loss, avg_acc, avg_iou, elapsed, eta,
                )

        self.scheduler.step()

        n = len(self.train_dataloader)
        return total_loss / n, total_acc / n, total_iou_acc / n

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def eval_step(self, epoch: int) -> tuple[float, float, float]:
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        total_iou_acc = 0.0
        n_batches = len(self.test_dataloader)
        t0 = time.time()

        for i, (X_Img, X_Text, y_bbox) in enumerate(self.test_dataloader, 1):
            X_Img = X_Img.to(self.device, non_blocking=True)
            X_Text = self._move_text_to_device(X_Text, self.device)
            y_bbox = y_bbox.to(self.device, non_blocking=True)

            with autocast("cuda", enabled=self.use_amp):
                roi, y_pred = self.model(X_Img, X_Text["input_ids"], X_Text["attention_mask"])
                y = CreateBatchLabels(roi, y_bbox, image_size=self.image_size).to(self.device)
                loss = self.loss_fn(y_pred.squeeze(-1), y.long())

            pred_idx = y_pred.squeeze(-1).argmax(dim=1)
            total_loss += loss.item()
            total_acc += self._accuracy(y, pred_idx)
            total_iou_acc += compute_acc_at_iou(roi, pred_idx, y_bbox, self.image_size)

            if i % self.log_every == 0 or i == n_batches:
                elapsed = time.time() - t0
                avg_loss = total_loss / i
                avg_acc = total_acc / i
                avg_iou = total_iou_acc / i
                eta = elapsed / i * (n_batches - i)
                logger.info(
                    "  Epoch %2d Eval  [%3d/%d] | loss %.4f | acc %.1f%% | IoU@.5 %.1f%% | %.1fs | ETA %.0fs",
                    epoch, i, n_batches, avg_loss, avg_acc, avg_iou, elapsed, eta,
                )

        n = len(self.test_dataloader)
        return total_loss / n, total_acc / n, total_iou_acc / n