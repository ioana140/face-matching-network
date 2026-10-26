
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

from config import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    RANDOM_SEED,
)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])

_VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])


class IdentityDataset(Dataset):
    def __init__(
        self,
        data:    List[Tuple[Path, int]],
        augment: bool = False,
    ):
        self.data      = data
        self.transform = _TRAIN_TRANSFORM if augment else _VAL_TRANSFORM

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        path, label = self.data[idx]
        image = self.transform(Image.open(str(path)).convert("RGB"))
        return image, label


class PKSampler(Sampler):
    def __init__(
        self,
        data:   List[Tuple[Path, int]],
        p:      int = 16,
        k:      int = 4,
        seed:   int = RANDOM_SEED,
    ):
        self.p    = p
        self.k    = k
        self.seed = seed

        # Group indices by identity
        self.id_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, (_, label) in enumerate(data):
            self.id_to_indices[label].append(idx)

        # Only keep identities with at least 2 images
        self.valid_ids = [
            ident for ident, indices in self.id_to_indices.items()
            if len(indices) >= 2
        ]

        self.n_batches = len(self.valid_ids) // self.p

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed)
        ids = list(self.valid_ids)
        rng.shuffle(ids)

        for batch_start in range(0, len(ids) - self.p + 1, self.p):
            batch_ids = ids[batch_start : batch_start + self.p]
            batch_indices = []

            for ident in batch_ids:
                indices = list(self.id_to_indices[ident])
                rng.shuffle(indices)

                if len(indices) >= self.k:
                    chosen = indices[:self.k]
                else:
                    # Pad by repeating images from same identity
                    chosen = indices.copy()
                    while len(chosen) < self.k:
                        chosen.append(rng.choice(indices))

                batch_indices.extend(chosen)

            yield batch_indices

    def __len__(self) -> int:
        return self.n_batches


def build_identity_train_loader(
    data:        List[Tuple[Path, int]],
    p:           int = 16,
    k:           int = 4,
    num_workers: int = 4,
) -> DataLoader:
    dataset = IdentityDataset(data, augment=True)
    sampler = PKSampler(data, p=p, k=k)

    return DataLoader(
        dataset,
        batch_sampler = sampler,
        num_workers   = num_workers,
        pin_memory    = True,
    )


def build_identity_val_loader(
    data:        List[Tuple[Path, int]],
    p:           int = 16,
    k:           int = 4,
    num_workers: int = 4,
) -> DataLoader:
    dataset = IdentityDataset(data, augment=False)
    sampler = PKSampler(data, p=p, k=k)

    return DataLoader(
        dataset,
        batch_sampler = sampler,
        num_workers   = num_workers,
        pin_memory    = True,
    )