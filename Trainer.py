import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from utils.Util import CreateBatchLabels


class VisualGroundingTrainer:
    """Handles training and evaluation loops for the VisualGrounding model."""

    def __init__(self, model, device, train_dataloader, test_dataloader, lr=3e-4, weight_decay=1e-4, use_amp=True):
        self.device = device
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Mixed-precision training (huge speed-up on H100 / A100)
        self.use_amp = use_amp and device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

    # ------------------------------------------------------------------
    @staticmethod
    def _move_text_to_device(text_batch: dict, device: torch.device) -> dict:
        """Move every tensor inside the tokenizer output dict to *device*."""
        return {k: v.to(device) for k, v in text_batch.items()}

    @staticmethod
    def _accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        return (torch.eq(y_true, y_pred).sum().item() / len(y_pred)) * 100.0

    # ------------------------------------------------------------------
    def train_step(self) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0

        for X_Img, X_Text, y_bbox in self.train_dataloader:
            X_Img = X_Img.to(self.device, non_blocking=True)
            X_Text = self._move_text_to_device(X_Text, self.device)
            y_bbox = y_bbox.to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                roi, y_pred = self.model(X_Img, X_Text["input_ids"], X_Text["attention_mask"])
                y = CreateBatchLabels(roi, y_bbox).to(self.device)
                loss = self.loss_fn(y_pred.squeeze(-1), y.long())

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            total_acc += self._accuracy(y, y_pred.squeeze(-1).argmax(dim=1))

        n = len(self.train_dataloader)
        return total_loss / n, total_acc / n

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def eval_step(self) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0

        for X_Img, X_Text, y_bbox in self.test_dataloader:
            X_Img = X_Img.to(self.device, non_blocking=True)
            X_Text = self._move_text_to_device(X_Text, self.device)
            y_bbox = y_bbox.to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                roi, y_pred = self.model(X_Img, X_Text["input_ids"], X_Text["attention_mask"])
                y = CreateBatchLabels(roi, y_bbox).to(self.device)
                loss = self.loss_fn(y_pred.squeeze(-1), y.long())

            total_loss += loss.item()
            total_acc += self._accuracy(y, y_pred.squeeze(-1).argmax(dim=1))

        n = len(self.test_dataloader)
        return total_loss / n, total_acc / n