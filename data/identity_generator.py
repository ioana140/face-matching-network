
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from config import (
    CELEBA_IDENTITY_FILE,
    CELEBA_PARTITION_FILE,
    RANDOM_SEED,
)


def build_celeba_identity_data(
    celeba_root:      Path,
    identity_file:    Path = CELEBA_IDENTITY_FILE,
    partition_file:   Path = CELEBA_PARTITION_FILE,
    min_images:       int  = 2,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    celeba_root    = Path(celeba_root)
    identity_file  = Path(identity_file)
    partition_file = Path(partition_file)

    # Read partition assignments: {img_name: partition_id}
    partition: Dict[str, int] = {}
    with open(partition_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                partition[parts[0]] = int(parts[1])

    # Read identity assignments: {img_name: identity_str}
    identity: Dict[str, str] = {}
    with open(identity_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                identity[parts[0]] = parts[1]

    # Build catalogues per partition
    train_catalogue: Dict[str, List[Path]] = defaultdict(list)
    val_catalogue:   Dict[str, List[Path]] = defaultdict(list)

    for img_name, part_id in partition.items():
        if part_id not in (0, 1):
            continue
        img_path = celeba_root / img_name
        if not img_path.exists():
            continue
        ident = identity.get(img_name)
        if ident is None:
            continue
        if part_id == 0:
            train_catalogue[ident].append(img_path)
        else:
            val_catalogue[ident].append(img_path)

    # Filter: only identities with >= min_images
    train_catalogue = {k: v for k, v in train_catalogue.items() if len(v) >= min_images}
    val_catalogue   = {k: v for k, v in val_catalogue.items()   if len(v) >= min_images}

    # Map identity strings to contiguous ints.
    train_ids_sorted = sorted(train_catalogue.keys())
    val_only_ids     = sorted(set(val_catalogue.keys()) - set(train_catalogue.keys()))

    id_to_int = {}
    for i, ident in enumerate(train_ids_sorted):
        id_to_int[ident] = i
    offset = len(train_ids_sorted)
    for i, ident in enumerate(val_only_ids):
        id_to_int[ident] = offset + i

    train_data = []
    for ident, paths in train_catalogue.items():
        int_id = id_to_int[ident]
        for p in paths:
            train_data.append((p, int_id))

    val_data = []
    for ident, paths in val_catalogue.items():
        int_id = id_to_int[ident]
        for p in paths:
            val_data.append((p, int_id))

    n_train_ids = len(train_catalogue)
    n_val_ids   = len(val_catalogue)
    n_shared    = len(set(train_catalogue.keys()) & set(val_catalogue.keys()))
    print(f"[identity_generator] CelebA identity data (min_images={min_images}):")
    print(f"  train: {len(train_data)} images, {n_train_ids} identities (IDs 0..{n_train_ids-1})")
    print(f"  val:   {len(val_data)} images, {n_val_ids} identities "
          f"({n_shared} shared with train, {len(val_only_ids)} val-only)")

    return train_data, val_data


def build_celeba_identity_data_shared(
    celeba_root:      Path,
    identity_file:    Path = CELEBA_IDENTITY_FILE,
    partition_file:   Path = CELEBA_PARTITION_FILE,
    min_images:       int  = 3,
    val_ratio:        float = 0.15,
    seed:             int   = RANDOM_SEED,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    celeba_root    = Path(celeba_root)
    identity_file  = Path(identity_file)
    partition_file = Path(partition_file)

    # Read partition assignments (keep partitions 0 and 1 only)
    partition: Dict[str, int] = {}
    with open(partition_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                partition[parts[0]] = int(parts[1])

    # Read identity assignments
    identity: Dict[str, str] = {}
    with open(identity_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                identity[parts[0]] = parts[1]

    # Pool all images from partitions 0 + 1, grouped by identity
    catalogue: Dict[str, List[Path]] = defaultdict(list)
    for img_name, part_id in partition.items():
        if part_id not in (0, 1):
            continue
        img_path = celeba_root / img_name
        if not img_path.exists():
            continue
        ident = identity.get(img_name)
        if ident is None:
            continue
        catalogue[ident].append(img_path)

    # Filter: only identities with >= min_images
    catalogue = {k: v for k, v in catalogue.items() if len(v) >= min_images}

    # Map identity strings to contiguous ints
    all_ids = sorted(catalogue.keys())
    id_to_int = {ident: i for i, ident in enumerate(all_ids)}

    # Split each identity's images into train / val
    rng = random.Random(seed)
    train_data = []
    val_data   = []

    for ident, paths in catalogue.items():
        int_id = id_to_int[ident]
        shuffled = list(paths)
        rng.shuffle(shuffled)

        # At least 1 image for val, at least 2 for train
        n_val = max(1, int(len(shuffled) * val_ratio))
        n_val = min(n_val, len(shuffled) - 2)  # keep ≥2 for train

        val_imgs   = shuffled[:n_val]
        train_imgs = shuffled[n_val:]

        for p in train_imgs:
            train_data.append((p, int_id))
        for p in val_imgs:
            val_data.append((p, int_id))

    num_classes = len(catalogue)
    print(f"[identity_generator] CelebA identity data SHARED (min_images={min_images}, val_ratio={val_ratio}):")
    print(f"  identities: {num_classes} (all shared between train and val)")
    print(f"  train: {len(train_data)} images")
    print(f"  val:   {len(val_data)} images")

    return train_data, val_data