import json
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from transformers import BertTokenizer


class VG_Dataset(Dataset):
    """Visual Grounding dataset.

    Expects *root_dir* to contain:
      - ``images/`` folder with ``<image_id>.jpg`` files
      - ``annotations_1.json`` and (optionally) ``annotations_2.json``

    Each annotation entry: ``{"image_id": ..., "text": ..., "bbox": [x,y,w,h]}``
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, root_dir: str, image_size: int = 512, max_length: int = 40):
        self.root_dir = root_dir
        self.max_length = max_length
        self.image_size = image_size

        # --- load annotations ---
        self.annotations: list[dict] = []
        for fname in ("annotations_1.json", "annotations_2.json"):
            path = os.path.join(root_dir, fname)
            if os.path.isfile(path):
                with open(path, "r") as f:
                    self.annotations.extend(json.load(f))
        if not self.annotations:
            raise FileNotFoundError(
                f"No annotation files found in {root_dir}. "
                "Expected annotations_1.json and/or annotations_2.json."
            )

        self.transforms = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int):
        ann = self.annotations[index]
        image_id = ann["image_id"]
        pil_img = Image.open(os.path.join(self.root_dir, "images", f"{image_id}.jpg"))
        orig_w, orig_h = pil_img.size
        image = self.transforms(pil_img)

        text_encoded = self.tokenizer(
            ann["text"],
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Normalise bbox [x, y, w, h] by *original* image dims so that
        # bbox * image_size gives coordinates in the resized image space
        bbox = torch.tensor(ann["bbox"], dtype=torch.float32)
        bbox[0] /= orig_w   # x
        bbox[1] /= orig_h   # y
        bbox[2] /= orig_w   # w
        bbox[3] /= orig_h   # h
        return image, text_encoded, bbox


def GetDataloader(
    root_dir: str,
    batch_size: int = 16,
    split: str = "train",
    num_workers: int = 4,
    max_samples: int | None = None,
    image_size: int = 512,
) -> DataLoader:
    """Build a DataLoader for *split* with optional sample-count capping."""
    full_ds = VG_Dataset(root_dir, image_size=image_size)

    limit = min(max_samples, len(full_ds)) if max_samples is not None else len(full_ds)

    ds: Dataset = Subset(full_ds, list(range(limit))) if limit < len(full_ds) else full_ds

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )