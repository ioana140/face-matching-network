"""Generates genuine/impostor image pairs for CelebA, CASIA-WebFace and LFW.

For LFW: seeds with the official pairs.csv benchmark split, then tops up
with additional pairs generated from disk (similar pairs enumerate all
same-identity combinations and sample from them; dissimilar pairs pick one
image each from two different identities), deduplicated against the CSV
pairs, then split into train/test.

Labels: 0 = same person (similar), 1 = different (dissimilar).
"""

import csv
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Set, Tuple

from config import (
    CASIA_TARGET_PAIRS,
    CASIA_TRAIN_SPLIT,
    CELEBA_IDENTITY_FILE,
    CELEBA_PARTITION_FILE,
    CELEBA_TARGET_PAIRS,
    LABEL_DISSIMILAR,
    LABEL_SIMILAR,
    RANDOM_SEED,
    TRAIN_SPLIT,
)

TARGET_PAIRS  = 15_000
SIMILAR_RATIO = 0.5          # 1:1 -> 50% similar, 50% dissimilar


def _image_path(lfw_root: Path, person_name: str, image_num: int) -> Path:
    filename = f"{person_name}_{int(image_num):04d}.jpg"
    return lfw_root / person_name / filename


def _pair_key(path_a: Path, path_b: Path) -> Tuple[str, str]:
    a, b = str(path_a), str(path_b)
    return (a, b) if a <= b else (b, a)


def _load_csv_pairs(
    csv_path: Path,
    lfw_root: Path,
    skip_missing: bool = True,
) -> Tuple[List[Tuple[Path, Path, int]], Set[Tuple[str, str]]]:
    pairs:     List[Tuple[Path, Path, int]]  = []
    seen_keys: Set[Tuple[str, str]]          = set()
    skipped = 0

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            cols = [c.strip() for c in row if c.strip()]
            if not cols or cols[0].lower() in ("name", "name1"):
                continue   # header row

            if len(cols) == 3:
                name, n1, n2 = cols
                path_a = _image_path(lfw_root, name, int(n1))
                path_b = _image_path(lfw_root, name, int(n2))
                label  = LABEL_SIMILAR

            elif len(cols) == 4:
                name1, n1, name2, n2 = cols
                path_a = _image_path(lfw_root, name1, int(n1))
                path_b = _image_path(lfw_root, name2, int(n2))
                label  = LABEL_DISSIMILAR
            else:
                continue

            if skip_missing and (not path_a.exists() or not path_b.exists()):
                skipped += 1
                continue

            key = _pair_key(path_a, path_b)
            if key not in seen_keys:
                seen_keys.add(key)
                pairs.append((path_a, path_b, label))

    n_sim = sum(1 for _, _, l in pairs if l == LABEL_SIMILAR)
    n_dis = sum(1 for _, _, l in pairs if l == LABEL_DISSIMILAR)
    print(f"[pair_generator] CSV loaded : {len(pairs)} pairs "
          f"({n_sim} similar, {n_dis} dissimilar)"
          + (f", {skipped} skipped (missing files)" if skipped else ""))

    return pairs, seen_keys


def _build_catalogue(lfw_root: Path) -> Dict[str, List[Path]]:
    catalogue: Dict[str, List[Path]] = {}
    for person_dir in sorted(lfw_root.iterdir()):
        if not person_dir.is_dir():
            continue
        imgs = sorted(person_dir.glob("*.jpg"))
        if imgs:
            catalogue[person_dir.name] = imgs
    return catalogue


def _generate_similar_pairs(
    catalogue: Dict[str, List[Path]],
    n_needed:  int,
    seen_keys: Set[Tuple[str, str]],
    rng:       random.Random,
) -> List[Tuple[Path, Path, int]]:
    pool: List[Tuple[Path, Path]] = []
    for name, images in catalogue.items():
        if len(images) < 2:
            continue
        for img_a, img_b in combinations(images, 2):
            key = _pair_key(img_a, img_b)
            if key not in seen_keys:
                pool.append((img_a, img_b))

    if len(pool) < n_needed:
        print(f"[pair_generator] WARNING: only {len(pool)} unique similar pairs "
              f"available, requested {n_needed}. Using all available.")
        n_needed = len(pool)

    rng.shuffle(pool)
    selected = pool[:n_needed]

    new_pairs: List[Tuple[Path, Path, int]] = []
    for img_a, img_b in selected:
        key = _pair_key(img_a, img_b)
        seen_keys.add(key)
        new_pairs.append((img_a, img_b, LABEL_SIMILAR))

    return new_pairs


def _generate_dissimilar_pairs(
    catalogue: Dict[str, List[Path]],
    n_needed:  int,
    seen_keys: Set[Tuple[str, str]],
    rng:       random.Random,
    max_attempts_multiplier: int = 10,
) -> List[Tuple[Path, Path, int]]:
    persons = list(catalogue.keys())
    new_pairs: List[Tuple[Path, Path, int]] = []
    attempts  = 0
    max_tries = n_needed * max_attempts_multiplier

    while len(new_pairs) < n_needed and attempts < max_tries:
        attempts += 1
        name_a, name_b = rng.sample(persons, 2)
        img_a = rng.choice(catalogue[name_a])
        img_b = rng.choice(catalogue[name_b])
        key   = _pair_key(img_a, img_b)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        new_pairs.append((img_a, img_b, LABEL_DISSIMILAR))

    if len(new_pairs) < n_needed:
        print(f"[pair_generator] WARNING: only generated {len(new_pairs)} "
              f"dissimilar pairs (requested {n_needed}).")

    return new_pairs


def build_pairs(
    csv_path:      Path,
    lfw_root:      Path,
    target_pairs:  int   = TARGET_PAIRS,
    similar_ratio: float = SIMILAR_RATIO,
    skip_missing:  bool  = True,
    seed:          int   = RANDOM_SEED,
    use_csv:       bool  = True,
) -> Tuple[List, List]:
    rng = random.Random(seed)
    lfw_root = Path(lfw_root)

    target_similar    = int(target_pairs * similar_ratio)
    target_dissimilar = target_pairs - target_similar

    print(f"[pair_generator] Target : {target_pairs} pairs "
          f"({target_similar} similar, {target_dissimilar} dissimilar)")

    catalogue = _build_catalogue(lfw_root)
    multi_catalogue = {k: v for k, v in catalogue.items() if len(v) >= 2}
    print(f"[pair_generator] Catalogue: {len(catalogue)} persons on disk "
          f"({len(multi_catalogue)} with >=2 images)")

    if use_csv:
        csv_pairs, seen_keys = _load_csv_pairs(csv_path, lfw_root, skip_missing)
        csv_similar    = [p for p in csv_pairs if p[2] == LABEL_SIMILAR]
        csv_dissimilar = [p for p in csv_pairs if p[2] == LABEL_DISSIMILAR]

        need_similar    = max(0, target_similar    - len(csv_similar))
        need_dissimilar = max(0, target_dissimilar - len(csv_dissimilar))

        print(f"[pair_generator] From CSV: {len(csv_similar)} similar, "
              f"{len(csv_dissimilar)} dissimilar")
        print(f"[pair_generator] Need to generate: {need_similar} more similar, "
              f"{need_dissimilar} more dissimilar")

        extra_similar    = []
        extra_dissimilar = []

        if need_similar > 0:
            extra_similar = _generate_similar_pairs(
                multi_catalogue, need_similar, seen_keys, rng
            )
        if need_dissimilar > 0:
            extra_dissimilar = _generate_dissimilar_pairs(
                catalogue, need_dissimilar, seen_keys, rng
            )

        all_pairs = csv_pairs + extra_similar + extra_dissimilar

    else:
        print("[pair_generator] Generating ALL pairs from scratch (no CSV)")
        seen_keys: Set[Tuple[str, str]] = set()

        similar_pairs = _generate_similar_pairs(
            multi_catalogue, target_similar, seen_keys, rng
        )
        dissimilar_pairs = _generate_dissimilar_pairs(
            catalogue, target_dissimilar, seen_keys, rng
        )

        all_pairs = similar_pairs + dissimilar_pairs

    rng.shuffle(all_pairs)

    total_sim = sum(1 for _, _, l in all_pairs if l == LABEL_SIMILAR)
    total_dis = sum(1 for _, _, l in all_pairs if l == LABEL_DISSIMILAR)
    print(f"[pair_generator] Final dataset: {len(all_pairs)} pairs "
          f"({total_sim} similar, {total_dis} dissimilar) "
          f"ratio={total_sim/max(total_dis,1):.2f}:1")

    rng.shuffle(all_pairs)
    n_train     = int(len(all_pairs) * TRAIN_SPLIT)
    train_pairs = all_pairs[:n_train]
    test_pairs  = all_pairs[n_train:]

    print(f"[pair_generator] Split → train: {len(train_pairs)}, "
          f"test: {len(test_pairs)}")

    return train_pairs, test_pairs


# CASIA-WebFace pair builder  (train + val splits, no CSV)

def build_casia_pairs(
    casia_root:    Path,
    target_pairs:  int   = CASIA_TARGET_PAIRS,
    similar_ratio: float = SIMILAR_RATIO,
    seed:          int   = RANDOM_SEED,
) -> Tuple[List, List]:
    rng = random.Random(seed)
    casia_root = Path(casia_root)

    target_similar    = int(target_pairs * similar_ratio)
    target_dissimilar = target_pairs - target_similar

    print(f"[pair_generator] CASIA target: {target_pairs} pairs "
          f"({target_similar} similar, {target_dissimilar} dissimilar)")

    catalogue = _build_catalogue(casia_root)
    multi_catalogue = {k: v for k, v in catalogue.items() if len(v) >= 2}
    print(f"[pair_generator] CASIA catalogue: {len(catalogue)} identities "
          f"({len(multi_catalogue)} with >=2 images)")

    seen_keys: Set[Tuple[str, str]] = set()

    similar_pairs = _generate_similar_pairs(
        multi_catalogue, target_similar, seen_keys, rng
    )
    dissimilar_pairs = _generate_dissimilar_pairs(
        catalogue, target_dissimilar, seen_keys, rng
    )

    all_pairs = similar_pairs + dissimilar_pairs
    rng.shuffle(all_pairs)

    total_sim = sum(1 for _, _, l in all_pairs if l == LABEL_SIMILAR)
    total_dis = sum(1 for _, _, l in all_pairs if l == LABEL_DISSIMILAR)
    print(f"[pair_generator] CASIA final: {len(all_pairs)} pairs "
          f"({total_sim} similar, {total_dis} dissimilar)")

    n_train     = int(len(all_pairs) * CASIA_TRAIN_SPLIT)
    train_pairs = all_pairs[:n_train]
    val_pairs   = all_pairs[n_train:]

    print(f"[pair_generator] CASIA split → train: {len(train_pairs)}, "
          f"val: {len(val_pairs)}")

    return train_pairs, val_pairs


# CelebA pair builder  (uses official list_eval_partition.txt splits)

def build_celeba_pairs(
    celeba_root:      Path,
    identity_file:    Path  = CELEBA_IDENTITY_FILE,
    partition_file:   Path  = CELEBA_PARTITION_FILE,
    target_pairs:     int   = CELEBA_TARGET_PAIRS,
    similar_ratio:    float = SIMILAR_RATIO,
    seed:             int   = RANDOM_SEED,
) -> Tuple[List, List]:
    rng = random.Random(seed)
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

    # Read identity assignments: {img_name: identity_id}
    identity: Dict[str, str] = {}
    with open(identity_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                identity[parts[0]] = parts[1]

    # Build separate catalogues per partition
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

    train_catalogue = dict(train_catalogue)
    val_catalogue   = dict(val_catalogue)
    train_multi     = {k: v for k, v in train_catalogue.items() if len(v) >= 2}
    val_multi       = {k: v for k, v in val_catalogue.items()   if len(v) >= 2}

    n_train_imgs = sum(len(v) for v in train_catalogue.values())
    n_val_imgs   = sum(len(v) for v in val_catalogue.values())
    print(f"[pair_generator] CelebA train partition: {len(train_catalogue)} identities, "
          f"{n_train_imgs} images ({len(train_multi)} with >=2)")
    print(f"[pair_generator] CelebA val   partition: {len(val_catalogue)} identities, "
          f"{n_val_imgs} images ({len(val_multi)} with >=2)")

    # Scale val target proportionally to partition sizes
    val_ratio       = n_val_imgs / max(n_train_imgs, 1)
    val_target      = max(1000, int(target_pairs * val_ratio))

    train_sim  = int(target_pairs * similar_ratio)
    train_dis  = target_pairs - train_sim
    val_sim    = int(val_target  * similar_ratio)
    val_dis    = val_target  - val_sim

    print(f"[pair_generator] CelebA generating → train: {target_pairs} pairs, "
          f"val: {val_target} pairs")

    seen_train: Set[Tuple[str, str]] = set()
    seen_val:   Set[Tuple[str, str]] = set()

    train_pairs = (
        _generate_similar_pairs(train_multi,    train_sim, seen_train, rng) +
        _generate_dissimilar_pairs(train_catalogue, train_dis, seen_train, rng)
    )
    val_pairs = (
        _generate_similar_pairs(val_multi,    val_sim, seen_val, rng) +
        _generate_dissimilar_pairs(val_catalogue, val_dis, seen_val, rng)
    )

    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)

    print(f"[pair_generator] CelebA pairs → train: {len(train_pairs)}, "
          f"val: {len(val_pairs)}")

    return train_pairs, val_pairs


# CelebA pooled pairs (ignore partition file, 80/20 split)

def build_celeba_pairs_pooled(
    celeba_root:      Path,
    identity_file:    Path  = CELEBA_IDENTITY_FILE,
    target_pairs:     int   = CELEBA_TARGET_PAIRS,
    similar_ratio:    float = SIMILAR_RATIO,
    train_ratio:      float = 0.80,
    seed:             int   = RANDOM_SEED,
) -> Tuple[List, List]:
    rng = random.Random(seed)
    celeba_root   = Path(celeba_root)
    identity_file = Path(identity_file)

    # Read identity assignments: {img_name: identity_id}
    identity: Dict[str, str] = {}
    with open(identity_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                identity[parts[0]] = parts[1]

    # Build catalogue from ALL images (no partition filtering)
    catalogue: Dict[str, List[Path]] = defaultdict(list)
    for img_name, ident in identity.items():
        img_path = celeba_root / img_name
        if img_path.exists():
            catalogue[ident].append(img_path)

    catalogue = dict(catalogue)
    multi = {k: v for k, v in catalogue.items() if len(v) >= 2}

    n_imgs = sum(len(v) for v in catalogue.values())
    print(f"[pair_generator] CelebA POOLED: {len(catalogue)} identities, "
          f"{n_imgs} images ({len(multi)} with >=2)")

    # Generate all pairs from the pooled catalogue
    n_sim = int(target_pairs * similar_ratio)
    n_dis = target_pairs - n_sim

    print(f"[pair_generator] Generating {target_pairs} pooled pairs "
          f"({n_sim} similar + {n_dis} dissimilar) ...")

    seen: Set[Tuple[str, str]] = set()
    all_pairs = (
        _generate_similar_pairs(multi, n_sim, seen, rng) +
        _generate_dissimilar_pairs(catalogue, n_dis, seen, rng)
    )
    rng.shuffle(all_pairs)

    # Split 80/20
    n_train = int(len(all_pairs) * train_ratio)
    train_pairs = all_pairs[:n_train]
    val_pairs   = all_pairs[n_train:]

    print(f"[pair_generator] CelebA pooled pairs → "
          f"train: {len(train_pairs)}, val: {len(val_pairs)}")

    return train_pairs, val_pairs


# Triplet generation from CelebA

def build_celeba_triplets(
    celeba_root:      Path,
    identity_file:    Path  = CELEBA_IDENTITY_FILE,
    partition_file:   Path  = CELEBA_PARTITION_FILE,
    target_triplets:  int   = CELEBA_TARGET_PAIRS,
    seed:             int   = RANDOM_SEED,
) -> Tuple[List, List]:
    rng = random.Random(seed)
    celeba_root    = Path(celeba_root)
    identity_file  = Path(identity_file)
    partition_file = Path(partition_file)

    # Read partition and identity assignments (same as build_celeba_pairs)
    partition: Dict[str, int] = {}
    with open(partition_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                partition[parts[0]] = int(parts[1])

    identity: Dict[str, str] = {}
    with open(identity_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                identity[parts[0]] = parts[1]

    # Build separate catalogues per partition
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

    train_catalogue = dict(train_catalogue)
    val_catalogue   = dict(val_catalogue)

    n_train_imgs = sum(len(v) for v in train_catalogue.values())
    n_val_imgs   = sum(len(v) for v in val_catalogue.values())
    print(f"[pair_generator] CelebA triplets — train: {len(train_catalogue)} identities, "
          f"{n_train_imgs} images")
    print(f"[pair_generator] CelebA triplets — val:   {len(val_catalogue)} identities, "
          f"{n_val_imgs} images")

    # Scale val target proportionally
    val_ratio  = n_val_imgs / max(n_train_imgs, 1)
    val_target = max(1000, int(target_triplets * val_ratio))

    print(f"[pair_generator] Generating → train: {target_triplets} triplets, "
          f"val: {val_target} triplets")

    train_triplets = _generate_triplets(train_catalogue, target_triplets, rng)
    val_triplets   = _generate_triplets(val_catalogue, val_target, rng)

    rng.shuffle(train_triplets)
    rng.shuffle(val_triplets)

    print(f"[pair_generator] CelebA triplets → train: {len(train_triplets)}, "
          f"val: {len(val_triplets)}")

    return train_triplets, val_triplets


def _generate_triplets(
    catalogue: Dict[str, List[Path]],
    n_needed:  int,
    rng:       random.Random,
) -> List[Tuple[Path, Path, Path]]:
    multi = {k: v for k, v in catalogue.items() if len(v) >= 2}
    if not multi:
        print("[pair_generator] WARNING: no identities with >=2 images for triplets")
        return []

    multi_ids = list(multi.keys())
    all_ids   = list(catalogue.keys())
    triplets: List[Tuple[Path, Path, Path]] = []

    for _ in range(n_needed):
        # Pick anchor identity (must have >=2 images)
        anchor_id = rng.choice(multi_ids)
        anchor, positive = rng.sample(multi[anchor_id], 2)

        # Pick negative identity (different from anchor)
        neg_id = anchor_id
        while neg_id == anchor_id:
            neg_id = rng.choice(all_ids)
        negative = rng.choice(catalogue[neg_id])

        triplets.append((anchor, positive, negative))

    return triplets


# LFW official CSV test set (6 000 pairs, no generation)

def build_lfw_test_pairs(
    csv_path:     Path,
    lfw_root:     Path,
    skip_missing: bool = True,
) -> List:
    pairs, _ = _load_csv_pairs(csv_path, lfw_root, skip_missing)
    print(f"[pair_generator] LFW test set: {len(pairs)} official pairs")
    return pairs
