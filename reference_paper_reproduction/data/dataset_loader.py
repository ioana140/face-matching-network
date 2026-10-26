
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np



# LFW root validation

def verify_lfw_root(lfw_root: Path) -> Path:
    lfw_root = Path(lfw_root).resolve()

    if not lfw_root.exists():
        raise FileNotFoundError(
            f"LFW root not found: {lfw_root}\n"
            "Set LFW_ROOT in config.py to the folder that contains "
            "one sub-directory per person."
        )

    # Sanity check: at least one sub-directory with at least one .jpg
    subdirs = [d for d in lfw_root.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(
            f"LFW root appears empty (no person sub-directories): {lfw_root}"
        )

    print(f"[dataset_loader] LFW root verified: {lfw_root}")
    print(f"[dataset_loader] Found {len(subdirs)} person directories.")
    return lfw_root



def build_image_catalogue(lfw_root: Path) -> Dict[str, List[Path]]:
    lfw_root  = Path(lfw_root)
    catalogue: Dict[str, List[Path]] = {}

    for person_dir in sorted(lfw_root.iterdir()):
        if not person_dir.is_dir():
            continue
        images = sorted(person_dir.glob("*.jpg")) + sorted(person_dir.glob("*.png"))
        if images:
            catalogue[person_dir.name] = images

    total_images = sum(len(v) for v in catalogue.values())
    multi        = sum(1 for v in catalogue.values() if len(v) >= 2)
    print(f"[dataset_loader] Catalogue: {len(catalogue)} identities, "
          f"{total_images} images total, {multi} with ≥2 images.")
    return catalogue


def load_image(path: Path, target_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    try:
        import cv2
        img = cv2.imread(str(path))
        if img is None:
            raise OSError(f"cv2 could not read: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (target_size[1], target_size[0]),
                         interpolation=cv2.INTER_AREA)
    except ImportError:
        from PIL import Image
        img = np.array(
            Image.open(str(path)).convert("RGB").resize(
                (target_size[1], target_size[0]), Image.BILINEAR
            )
        )
    return img.astype(np.float32) / 255.0
