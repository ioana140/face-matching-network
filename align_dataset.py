import argparse
import time
import warnings
from pathlib import Path

import torch
from PIL import Image

warnings.filterwarnings("ignore", category=FutureWarning)
from facenet_pytorch import MTCNN  # noqa: E402


DEFAULT_OUTPUT_SIZE = 224
DEFAULT_MARGIN      = 20
DEFAULT_BATCH_SIZE  = 32


def center_crop_square(img, size):
    w, h = img.size
    short = min(w, h)
    left = (w - short) // 2
    top  = (h - short) // 2
    img = img.crop((left, top, left + short, top + short))
    return img.resize((size, size), Image.LANCZOS)


def tensor_to_pil(tensor):
    arr = tensor.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()
    return Image.fromarray(arr)


def align_dataset(src_root, dst_root, size, margin, batch_size, device):
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    if not src_root.exists():
        raise FileNotFoundError(f"Source not found: {src_root}")
    dst_root.mkdir(parents=True, exist_ok=True)

    print(f"[align] Source : {src_root}")
    print(f"[align] Output : {dst_root}")
    print(f"[align] Size   : {size}x{size}, margin={margin}, batch={batch_size}")
    print(f"[align] Device : {device}")

    mtcnn = MTCNN(
        image_size=size,
        margin=margin,
        device=device,
        post_process=False,
        keep_all=False,
        select_largest=True,
    )

    print(f"[align] Scanning for *.jpg ...")
    images = sorted(src_root.rglob("*.jpg"))
    n_total = len(images)
    print(f"[align] Found {n_total} images")

    todo = []
    n_done = 0
    for p in images:
        rel = p.relative_to(src_root)
        if (dst_root / rel).exists():
            n_done += 1
        else:
            todo.append(p)

    print(f"[align] Resume: {n_done} already aligned, {len(todo)} to process")
    if not todo:
        print(f"[align] Nothing to do.")
        return

    n_aligned = 0
    n_fallback = 0
    n_error = 0
    t_start = time.time()

    for i in range(0, len(todo), batch_size):
        batch_paths = todo[i:i + batch_size]

        pil_imgs = []
        valid_paths = []
        for p in batch_paths:
            try:
                pil_imgs.append(Image.open(p).convert("RGB"))
                valid_paths.append(p)
            except Exception:
                n_error += 1

        if not pil_imgs:
            continue

        try:
            tensors = mtcnn(pil_imgs)
        except Exception as e:
            tensors = []
            for img in pil_imgs:
                try:
                    tensors.append(mtcnn(img))
                except Exception:
                    tensors.append(None)

        for src_path, src_img, t in zip(valid_paths, pil_imgs, tensors):
            rel = src_path.relative_to(src_root)
            out_path = dst_root / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                if t is not None:
                    aligned = tensor_to_pil(t)
                    aligned.save(out_path, "JPEG", quality=95)
                    n_aligned += 1
                else:
                    fallback = center_crop_square(src_img, size)
                    fallback.save(out_path, "JPEG", quality=95)
                    n_fallback += 1
            except Exception:
                n_error += 1

        n_processed = n_aligned + n_fallback + n_error
        if n_processed and (n_processed % (batch_size * 10) == 0 or i + batch_size >= len(todo)):
            elapsed = time.time() - t_start
            rate = n_processed / elapsed if elapsed > 0 else 0
            remaining = (len(todo) - n_processed) / rate if rate > 0 else 0
            print(f"[align]   {n_processed}/{len(todo)}  "
                  f"aligned={n_aligned}  fallback={n_fallback}  error={n_error}  "
                  f"({rate:.1f} img/s, ETA {remaining/60:.1f} min)")

    elapsed = time.time() - t_start
    print(f"\n[align] Done in {elapsed/60:.1f} min")
    print(f"        aligned    : {n_aligned}")
    print(f"        fallback   : {n_fallback}  (no face detected, center crop saved)")
    print(f"        error      : {n_error}")
    print(f"        already-done: {n_done}")
    print(f"        Output     : {dst_root}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src",    type=str, required=True, help="Source dataset root")
    p.add_argument("--dst",    type=str, required=True, help="Output (aligned) root")
    p.add_argument("--size",   type=int, default=DEFAULT_OUTPUT_SIZE)
    p.add_argument("--margin", type=int, default=DEFAULT_MARGIN)
    p.add_argument("--batch",  type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    if device != args.device:
        print(f"[align] WARNING: requested {args.device}, falling back to {device}")
    align_dataset(args.src, args.dst, args.size, args.margin, args.batch, device)


if __name__ == "__main__":
    main()
