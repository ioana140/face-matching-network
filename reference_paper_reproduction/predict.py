import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Predict whether two face images belong to the same person."
    )
    p.add_argument("--img1",       type=str, required=True)
    p.add_argument("--img2",       type=str, required=True)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--threshold",  type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    import torch
    from PIL import Image
    from torchvision import transforms

    from config import (
        CHECKPOINT_DIR, DECISION_THRESHOLD,
        IMAGE_HEIGHT, IMAGE_WIDTH,
    )
    from model import SiameseNetwork
    from training import Trainer


    img1_path  = Path(args.img1)
    img2_path  = Path(args.img2)
    threshold  = args.threshold if args.threshold is not None else DECISION_THRESHOLD
    checkpoint = Path(args.checkpoint) if args.checkpoint \
                 else CHECKPOINT_DIR / "best_model.pt"

    for p in (img1_path, img2_path):
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint}.\n"
            "Run 'python train.py' first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = SiameseNetwork()
    model = Trainer.load_checkpoint(model, checkpoint, device)
    model.eval()

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ),
    ])

    def _load(path: Path) -> torch.Tensor:
        img = Image.open(str(path)).convert("RGB")
        return transform(img).unsqueeze(0).to(device)   # (1, C, H, W)

    img_a = _load(img1_path)
    img_b = _load(img2_path)

    with torch.no_grad():
        d = model(img_a, img_b).item()   # scalar distance

    decision = 0 if d < threshold else 1
    verdict  = "SAME PERSON  (similar)"        if decision == 0 \
               else "DIFFERENT PEOPLE (dissimilar)"

    print("\n" + "=" * 55)
    print("  FACE VERIFICATION RESULT")
    print("=" * 55)
    print(f"  Image A    : {img1_path.name}")
    print(f"  Image B    : {img2_path.name}")
    print(f"  Distance d : {d:.4f}")
    print(f"  Threshold  : {threshold}")
    print(f"  Label D    : {decision}  →  {verdict}")
    print("=" * 55)

    _try_save_figure(img1_path, img2_path, d, verdict)
    return decision, d


def _try_save_figure(path_a, path_b, distance, verdict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        for ax, path, label in zip(axes, [path_a, path_b], ["Image A", "Image B"]):
            ax.imshow(Image.open(str(path)).convert("RGB"))
            ax.set_title(label, fontsize=12)
            ax.axis("off")

        color = "green" if "SAME" in verdict else "red"
        fig.suptitle(f"d = {distance:.4f}  |  {verdict}",
                     fontsize=13, color=color, fontweight="bold")
        plt.tight_layout()
        out = Path("prediction_result.png")
        plt.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[predict.py] Saved result image → {out}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
