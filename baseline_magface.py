"""
MagFace baseline (iResNet-100, MS1MV2)
"""
import csv

import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from sklearn.metrics import roc_auc_score, roc_curve
from torch.nn import functional as F
from torchvision.transforms import Compose, ToTensor

from gradcam_alignment import ROOT_DIR, LFW_ROOT_DEFAULT, PAIRS_CSV_DEFAULT
from data import verify_lfw_root, build_lfw_test_pairs
from align_arcface_112 import detect_and_align
from magface_iresnet_arch import iresnet100

CHECKPOINT_PATH = ROOT_DIR / "checkpoints" / "_external_baselines" / "magface_iresnet100_MS1MV2_ddp.pth"
_TRANSFORM = Compose([ToTensor()])  # official recipe: no mean/std shift, just [0,1]


def to_bgr(pil_112):
    arr = np.asarray(pil_112)[:, :, ::-1]
    return Image.fromarray(arr.copy())


def load_model(device):
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    assert ckpt["arch"] == "iresnet100", ckpt["arch"]
    sd = ckpt["state_dict"]
    stripped = {k[len("features.module."):]: v for k, v in sd.items() if k.startswith("features.module.")}
    model = iresnet100(num_classes=512)
    missing, unexpected = model.load_state_dict(stripped, strict=True)
    model.eval().to(device)
    return model


def embed(model, pil_112, device):
    x = _TRANSFORM(to_bgr(pil_112)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
    return F.normalize(out, p=2, dim=1).squeeze(0)


def threshold_sweep(similarities, labels):
    best_t, best_acc = 0.5, 0.0
    for t in np.arange(-1.0, 1.0, 0.005):
        preds = (similarities < t).astype(int)
        acc = float((preds == labels).mean())
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t, best_acc


def main():
    output_dir = ROOT_DIR / "results_summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[magface_baseline] Device: {device}")
    print(f"[magface_baseline] Loading MagFace iResNet-100 (MS1MV2) from {CHECKPOINT_PATH.name} ...")
    model = load_model(device)

    lfw_root = verify_lfw_root(LFW_ROOT_DEFAULT)
    pairs = build_lfw_test_pairs(csv_path=PAIRS_CSV_DEFAULT, lfw_root=lfw_root, skip_missing=True)

    mtcnn = MTCNN(keep_all=False, select_largest=True)

    similarities, labels = [], []
    n_fallback = [0]
    for i, (path_a, path_b, label) in enumerate(pairs):
        pil_a = Image.open(str(path_a)).convert("RGB")
        pil_b = Image.open(str(path_b)).convert("RGB")
        aligned_a = detect_and_align(mtcnn, pil_a)
        aligned_b = detect_and_align(mtcnn, pil_b)
        if aligned_a is None:
            aligned_a = pil_a.resize((112, 112)); n_fallback[0] += 1
        if aligned_b is None:
            aligned_b = pil_b.resize((112, 112)); n_fallback[0] += 1
        emb_a = embed(model, aligned_a, device)
        emb_b = embed(model, aligned_b, device)
        sim = float(torch.dot(emb_a, emb_b).item())
        similarities.append(sim)
        labels.append(label)
        if (i + 1) % 500 == 0:
            print(f"[magface_baseline]   {i + 1}/{len(pairs)} pairs")

    similarities = np.asarray(similarities)
    labels = np.asarray(labels)
    print(f"[magface_baseline] Done. {n_fallback[0]} / {2 * len(pairs)} images used the no-face fallback (plain resize).")

    auc = float(roc_auc_score(labels, 1.0 - similarities))
    fpr, tpr, _ = roc_curve(labels, 1.0 - similarities)
    fnr = 1 - tpr
    eer = float(fpr[int(np.nanargmin(np.abs(fpr - fnr)))])
    oracle_t, oracle_acc = threshold_sweep(similarities, labels)
    mean_sim = float(similarities[labels == 0].mean())
    mean_dis = float(similarities[labels == 1].mean())

    print("\n" + "=" * 70)
    print("  BASELINE -- MagFace (iResNet-100, MS1MV2, official checkpoint)")
    print("=" * 70)
    print(f"  AUC-ROC                          {auc:.4f}")
    print(f"  EER                              {eer * 100:.2f} %")
    print(f"  Mean cosine sim -- genuine        {mean_sim:.4f}")
    print(f"  Mean cosine sim -- impostor       {mean_dis:.4f}")
    print(f"  LFW-oracle threshold sweep -> {oracle_t:.3f}  acc={oracle_acc * 100:.2f}%  "
          f"(tuned ON the test set, cite as reference only, not an operating point)")
    print("=" * 70)

    csv_path = output_dir / "baseline_magface_scores.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cosine_similarity", "label"])
        for s, l in zip(similarities, labels):
            writer.writerow([f"{s:.6f}", int(l)])
    print(f"[magface_baseline] Raw scores -> {csv_path}")

    summary_path = output_dir / "baseline_magface_summary.md"
    summary_path.write_text(
        "## Baseline -- MagFace (iResNet-100, MS1MV2, official checkpoint, no fine-tuning)\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| AUC-ROC | {auc:.4f} |\n"
        f"| EER | {eer * 100:.2f}% |\n"
        f"| Mean cosine sim (genuine) | {mean_sim:.4f} |\n"
        f"| Mean cosine sim (impostor) | {mean_dis:.4f} |\n"
        f"| LFW-oracle accuracy (reference only) | {oracle_acc * 100:.2f}% (t={oracle_t:.3f}) |\n"
        f"| No-face fallback rate | {n_fallback[0]}/{2 * len(pairs)} images |\n",
        encoding="utf-8",
    )
    print(f"[magface_baseline] Summary -> {summary_path}")


if __name__ == "__main__":
    main()
