
import csv
import random
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Set, Tuple

from config import (
    LABEL_DISSIMILAR,
    LABEL_SIMILAR,
    RANDOM_SEED,
    TRAIN_SPLIT,
)


TARGET_PAIRS  = 15_000
SIMILAR_RATIO = 0.5


def _image_path(lfw_root: Path, person_name: str, image_num: int) -> Path:
    filename = f"{person_name}_{int(image_num):04d}.jpg"
    return lfw_root / person_name / filename


def _pair_key(path_a: Path, path_b: Path) -> Tuple[str, str]:
    a, b = str(path_a), str(path_b)
    return (a, b) if a <= b else (b, a)


# Load official CSV pairs
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



# Build full image catalogue from disk
def _build_catalogue(lfw_root: Path) -> Dict[str, List[Path]]:
    catalogue: Dict[str, List[Path]] = {}
    for person_dir in sorted(lfw_root.iterdir()):
        if not person_dir.is_dir():
            continue
        imgs = sorted(person_dir.glob("*.jpg"))
        if imgs:
            catalogue[person_dir.name] = imgs
    return catalogue



# Generate additional similar pairs
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

# Generate additional dissimilar pairs
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



# Main entry point
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
        # Original mode: seed with official CSV, top up with generated pairs
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
        # Paper mode: generate ALL pairs from scratch
        print("[pair_generator] Generating ALL pairs from scratch (paper mode, no CSV)")
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

    # Random 60/40 train/test split
    rng.shuffle(all_pairs)
    n_train     = int(len(all_pairs) * TRAIN_SPLIT)
    train_pairs = all_pairs[:n_train]
    test_pairs  = all_pairs[n_train:]

    print(f"[pair_generator] Split → train: {len(train_pairs)}, "
          f"test: {len(test_pairs)}")

    return train_pairs, test_pairs
