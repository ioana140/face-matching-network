from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import (
    BATCH_SIZE,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    RANDOM_SEED,
)


# ImageNet normalisation
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])


class SiamesePairDataset(Dataset):
    def __init__(self, pairs: List[Tuple[Path, Path, int]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        path_a, path_b, label = self.pairs[idx]

        img_a = _TRANSFORM(Image.open(str(path_a)).convert("RGB"))
        img_b = _TRANSFORM(Image.open(str(path_b)).convert("RGB"))
        label = torch.tensor(float(label), dtype=torch.float32)

        return img_a, img_b, label


def build_train_loader(
    train_pairs: List[Tuple[Path, Path, int]],
    batch_size:  int = BATCH_SIZE,
    num_workers: int = 4,
) -> DataLoader:
    ds = SiamesePairDataset(train_pairs)
    return DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = False,
        generator   = torch.Generator().manual_seed(RANDOM_SEED),
    )


def build_test_loader(
    test_pairs:  List[Tuple[Path, Path, int]],
    batch_size:  int = BATCH_SIZE,
    num_workers: int = 4,
) -> DataLoader:
    ds = SiamesePairDataset(test_pairs)
    return DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = False,
    )
