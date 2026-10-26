"""
Modern pretrained baseline #1: FaceNet (InceptionResnetV1, VGGFace2 weights)

"""
import csv

import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics import roc_auc_score, roc_curve

from gradcam_alignment import ROOT_DIR, LFW_ROOT_DEFAULT, PAIRS_CSV_DEFAULT
from data import verify_lfw_root, build_lfw_test_pairs
from align_dataset import center_crop_square


def load_pil(path):
    return Image.open(str(path)).convert("RGB")


def embed_pair(mtcnn, resnet, device, path_a, path_b, n_fallback):
    tensors = []
    for path in (path_a, path_b):
        pil = load_pil(path)
        t = mtcnn(pil)
        if t is None:
            # Same fallback as align_dataset.py: no face detected ->
            # center-crop + resize, then apply FaceNet's own normalization.
            from facenet_pytorch.models.mtcnn import fixed_image_standardization
            fallback = center_crop_square(pil, 160)
            arr = torch.from_numpy(np.asarray(fallback)).permute(2, 0, 1).float()
            t = fixed_image_standardization(arr)
            n_fallback[0] += 1
        tensors.append(t)
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        emb = resnet(batch)  # already L2-normalized (512-d)
    return emb[0], emb[1]


def threshold_sweep(similarities, labels):
    best_t, best_acc = 0.5, 0.0
    for t in np.arange(-1.0, 1.0, 0.005):
        preds = (similarities < t).astype(int)
        acc = float((preds == labels).mean())
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t, best_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[facenet_baseline] Device: {device}")

    output_dir = ROOT_DIR / "results_summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    lfw_root = verify_lfw_root(LFW_ROOT_DEFAULT)
    pairs = build_lfw_test_pairs(csv_path=PAIRS_CSV_DEFAULT, lfw_root=lfw_root, skip_missing=True)

    print("[facenet_baseline] Loading MTCNN(160, margin=0) + InceptionResnetV1(vggface2) ...")
    mtcnn = MTCNN(image_size=160, margin=0, device=device, keep_all=False, select_largest=True)
    resnet = InceptionResnetV1(pretrained="vggface2", device=device).eval()

    similarities, labels = [], []
    n_fallback = [0]
    for i, (path_a, path_b, label) in enumerate(pairs):
        emb_a, emb_b = embed_pair(mtcnn, resnet, device, path_a, path_b, n_fallback)
        sim = float(torch.dot(emb_a, emb_b).item())
        similarities.append(sim)
        labels.append(label)
        if (i + 1) % 1000 == 0:
            print(f"[facenet_baseline]   {i + 1}/{len(pairs)} pairs")

    similarities = np.asarray(similarities)
    labels = np.asarray(labels)
    print(f"[facenet_baseline] Done. {n_fallback[0]} / {2 * len(pairs)} images used the no-face fallback crop.")

    auc = float(roc_auc_score(labels, 1.0 - similarities))
    fpr, tpr, _ = roc_curve(labels, 1.0 - similarities)
    fnr = 1 - tpr
    eer = float(fpr[int(np.nanargmin(np.abs(fpr - fnr)))])

    oracle_t, oracle_acc = threshold_sweep(similarities, labels)
    mean_sim = float(similarities[labels == 0].mean())
    mean_dis = float(similarities[labels == 1].mean())

    print("\n" + "=" * 70)
    print("  BASELINE -- FaceNet (InceptionResnetV1, VGGFace2 pretrained)")
    print("=" * 70)
    print(f"  AUC-ROC                          {auc:.4f}")
    print(f"  EER                              {eer * 100:.2f} %")
    print(f"  Mean cosine sim -- genuine        {mean_sim:.4f}")
    print(f"  Mean cosine sim -- impostor       {mean_dis:.4f}")
    print(f"  LFW-oracle threshold sweep -> {oracle_t:.3f}  acc={oracle_acc * 100:.2f}%  "
          f"(tuned ON the test set, cite as reference only, not an operating point)")
    print("=" * 70)

    csv_path = output_dir / "baseline_facenet_scores.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cosine_similarity", "label"])
        for s, l in zip(similarities, labels):
            writer.writerow([f"{s:.6f}", int(l)])
    print(f"[facenet_baseline] Raw scores -> {csv_path}")

    summary_path = output_dir / "baseline_facenet_summary.md"
    summary_path.write_text(
        "## Baseline -- FaceNet (InceptionResnetV1, VGGFace2 pretrained, no fine-tuning)\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| AUC-ROC | {auc:.4f} |\n"
        f"| EER | {eer * 100:.2f}% |\n"
        f"| Mean cosine sim (genuine) | {mean_sim:.4f} |\n"
        f"| Mean cosine sim (impostor) | {mean_dis:.4f} |\n"
        f"| LFW-oracle accuracy (reference only) | {oracle_acc * 100:.2f}% (t={oracle_t:.3f}) |\n"
        f"| No-face fallback rate | {n_fallback[0]}/{2 * len(pairs)} images |\n",
        encoding="utf-8",
    )
    print(f"[facenet_baseline] Summary -> {summary_path}")


if __name__ == "__main__":
    main()
