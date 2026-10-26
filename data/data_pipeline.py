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
    VGGFACE_IMAGE_HEIGHT,
    VGGFACE_IMAGE_WIDTH,
    RANDOM_SEED,
)

# ImageNet normalisation (required for pretrained VGG-16 in PyTorch)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# VGG-Face normalisation
# The original model was trained with BGR channel order (OpenCV/Caffe convention)
# and [0,255] pixel range with mean subtraction: B=129.1863, G=104.7624, R=93.5940.
#
# Since PIL loads RGB and ToTensor() scales to [0,1]:
#   1. We swap R↔B channels to get BGR (via _RGBtoBGR transform)
#   2. std=1/255 scales [0,1] back to [0,255]
#   3. Mean subtracted in BGR order: [129.1863, 104.7624, 93.5940]
_VGGFACE_MEAN_BGR = [129.1863 / 255.0, 104.7624 / 255.0, 93.5940 / 255.0]
_VGGFACE_STD_BGR  = [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0]


class _RGBtoBGR:
    def __call__(self, tensor):
        return tensor[[2, 1, 0], ...]

# Train transform — stronger augmentation for cross-domain generalization
_TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.RandomGrayscale(p=0.1),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])

# Test/val transform — no augmentation
_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])

# VGG-Face train transform (224x224, BGR + [0,255] mean subtraction)
_VGGFACE_TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((VGGFACE_IMAGE_HEIGHT, VGGFACE_IMAGE_WIDTH)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),          # RGB [0,1]
    _RGBtoBGR(),                    # BGR [0,1]
    transforms.Normalize(_VGGFACE_MEAN_BGR, _VGGFACE_STD_BGR),  # BGR [0,255] - mean
])

# VGG-Face test/val transform (224x224)
_VGGFACE_TRANSFORM = transforms.Compose([
    transforms.Resize((VGGFACE_IMAGE_HEIGHT, VGGFACE_IMAGE_WIDTH)),
    transforms.ToTensor(),          # RGB [0,1]
    _RGBtoBGR(),                    # BGR [0,1]
    transforms.Normalize(_VGGFACE_MEAN_BGR, _VGGFACE_STD_BGR),  # BGR [0,255] - mean
])


class SiamesePairDataset(Dataset):

    def __init__(self, pairs: List[Tuple[Path, Path, int]], augment: bool = False,
                 preprocessing: str = "imagenet"):
        self.pairs = pairs
        if preprocessing == "vggface":
            self.transform = _VGGFACE_TRAIN_TRANSFORM if augment else _VGGFACE_TRANSFORM
        else:
            self.transform = _TRAIN_TRANSFORM if augment else _TRANSFORM

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        path_a, path_b, label = self.pairs[idx]

        img_a = self.transform(Image.open(str(path_a)).convert("RGB"))
        img_b = self.transform(Image.open(str(path_b)).convert("RGB"))
        label = torch.tensor(float(label), dtype=torch.float32)

        return img_a, img_b, label


class TripletDataset(Dataset):

    def __init__(self, triplets: List[Tuple[Path, Path, Path]], augment: bool = False):
        self.triplets  = triplets
        self.transform = _TRAIN_TRANSFORM if augment else _TRANSFORM

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int):
        path_a, path_p, path_n = self.triplets[idx]

        anchor   = self.transform(Image.open(str(path_a)).convert("RGB"))
        positive = self.transform(Image.open(str(path_p)).convert("RGB"))
        negative = self.transform(Image.open(str(path_n)).convert("RGB"))

        return anchor, positive, negative


def build_train_loader(
    train_pairs:    List[Tuple[Path, Path, int]],
    batch_size:     int = BATCH_SIZE,
    num_workers:    int = 4,
    preprocessing:  str = "imagenet",
) -> DataLoader:
    ds = SiamesePairDataset(train_pairs, augment=True, preprocessing=preprocessing)
    return DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = False,
        generator   = torch.Generator().manual_seed(RANDOM_SEED),
    )


def build_triplet_train_loader(
    triplets:    List[Tuple[Path, Path, Path]],
    batch_size:  int = BATCH_SIZE,
    num_workers: int = 4,
) -> DataLoader:
    ds = TripletDataset(triplets, augment=True)
    return DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = False,
        generator   = torch.Generator().manual_seed(RANDOM_SEED),
    )


def build_triplet_val_loader(
    triplets:    List[Tuple[Path, Path, Path]],
    batch_size:  int = BATCH_SIZE,
    num_workers: int = 4,
) -> DataLoader:
    ds = TripletDataset(triplets)
    return DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = False,
    )


def build_test_loader(
    test_pairs:     List[Tuple[Path, Path, int]],
    batch_size:     int = BATCH_SIZE,
    num_workers:    int = 4,
    preprocessing:  str = "imagenet",
) -> DataLoader:
    ds = SiamesePairDataset(test_pairs, preprocessing=preprocessing)
    return DataLoader(
        ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
        drop_last   = False,
    )
