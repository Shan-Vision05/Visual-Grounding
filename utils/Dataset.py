from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import os
from PIL import Image
from transformers import BertTokenizer
import json

from torch.utils.data import DataLoader, Subset

class VG_Dataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

        with open(os.path.join(root_dir, 'annotations_1.json'), 'r') as file:
            data1 = json.load(file)
        with open(os.path.join(root_dir, 'annotations_2.json'), 'r') as file:
            data2 = json.load(file)
        self.annotations = data1 + data2

        self.transforms = transforms.Compose([
            transforms.Lambda(lambda image: image.convert('RGB')),
            transforms.ToTensor(),
        ])
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = 15

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):

        ann = self.annotations[index]
        image_id = ann['image_id']
        image = self.transforms(
            Image.open(os.path.join(self.root_dir, 'images', f'{image_id}.jpg'))
        )

        text_encoded = self.tokenizer(
            ann['text'],
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors ='pt'
        )

        bbox = ann['bbox']
        bbox = torch.tensor(bbox, dtype=torch.float32)/512

        return image, text_encoded, bbox

def GetDataloader(root_dir, batch_size, typ):
    full_ds = VG_Dataset(root_dir)

    if typ == 'train':
        limit = min(10000, len(full_ds))
    elif typ == 'test':
        limit = min(2000, len(full_ds))
    else:
        limit = len(full_ds)

    indices = list(range(limit))
    ds = Subset(full_ds, indices)

    return DataLoader(ds, batch_size=batch_size, shuffle=True)