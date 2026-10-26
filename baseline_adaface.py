"""
AdaFace baseline (IR-50, WebFace4M)

Standard ArcFace-family preprocessing: 112x112, 5-point similarity-transform
alignment (align_arcface_112.py), not a plain resize/bbox-crop.

"""
import csv
import os
import sys

import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from sklearn.metrics import roc_auc_score, roc_curve
from torch.nn import functional as F
from torchvision.transforms import Compose, Normalize, ToTensor

from gradcam_alignment import ROOT_DIR, LFW_ROOT_DEFAULT, PAIRS_CSV_DEFAULT
from data import verify_lfw_root, build_lfw_test_pairs
from align_arcface_112 import detect_and_align

HF_REPO_ID = "minchul/cvlface_adaface_ir50_webface4m"
CACHE_PATH = os.path.expanduser(f"~/.cvlface_cache/{HF_REPO_ID}")

_TRANSFORM = Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


def download_model(repo_id, path, hf_token=None):
    from huggingface_hub import hf_hub_download
    os.makedirs(path, exist_ok=True)
    files_path = os.path.join(path, "files.txt")
    if not os.path.exists(files_path):
        hf_hub_download(repo_id, "files.txt", token=hf_token, local_dir=path)
    with open(files_path, "r") as f:
        files = f.read().split("\n")
    for file in [f for f in files if f] + ["config.json", "wrapper.py", "model.safetensors"]:
        full_path = os.path.join(path, file)
        if not os.path.exists(full_path):
            hf_hub_download(repo_id, file, token=hf_token, local_dir=path)


def load_model(path):
    from transformers import AutoModel
    cwd = os.getcwd()
    os.chdir(path)
    sys.path.insert(0, path)
    model = AutoModel.from_pretrained(path, trust_remote_code=True)
    os.chdir(cwd)
    sys.path.pop(0)
    model.eval()
    return model


def embed(model, pil_112):
    device = next(model.parameters()).device
    x = _TRANSFORM(pil_112).unsqueeze(0).to(device)
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

    print(f"[adaface_baseline] Downloading/loading {HF_REPO_ID} ...")
    download_model(HF_REPO_ID, CACHE_PATH)
    model = load_model(CACHE_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"[adaface_baseline] Device: {device}")

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
        emb_a = embed(model, aligned_a).to(device)
        emb_b = embed(model, aligned_b).to(device)
        sim = float(torch.dot(emb_a, emb_b).item())
        similarities.append(sim)
        labels.append(label)
        if (i + 1) % 500 == 0:
            print(f"[adaface_baseline]   {i + 1}/{len(pairs)} pairs")

    similarities = np.asarray(similarities)
    labels = np.asarray(labels)
    print(f"[adaface_baseline] Done. {n_fallback[0]} / {2 * len(pairs)} images used the no-face fallback (plain resize).")

    auc = float(roc_auc_score(labels, 1.0 - similarities))
    fpr, tpr, _ = roc_curve(labels, 1.0 - similarities)
    fnr = 1 - tpr
    eer = float(fpr[int(np.nanargmin(np.abs(fpr - fnr)))])
    oracle_t, oracle_acc = threshold_sweep(similarities, labels)
    mean_sim = float(similarities[labels == 0].mean())
    mean_dis = float(similarities[labels == 1].mean())

    print("\n" + "=" * 70)
    print("  BASELINE -- AdaFace (IR-50, WebFace4M, CVLface official release)")
    print("=" * 70)
    print(f"  AUC-ROC                          {auc:.4f}")
    print(f"  EER                              {eer * 100:.2f} %")
    print(f"  Mean cosine sim -- genuine        {mean_sim:.4f}")
    print(f"  Mean cosine sim -- impostor       {mean_dis:.4f}")
    print(f"  LFW-oracle threshold sweep -> {oracle_t:.3f}  acc={oracle_acc * 100:.2f}%  "
          f"(tuned ON the test set, cite as reference only, not an operating point)")
    print("=" * 70)

    csv_path = output_dir / "baseline_adaface_scores.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cosine_similarity", "label"])
        for s, l in zip(similarities, labels):
            writer.writerow([f"{s:.6f}", int(l)])
    print(f"[adaface_baseline] Raw scores -> {csv_path}")

    summary_path = output_dir / "baseline_adaface_summary.md"
    summary_path.write_text(
        "## Baseline -- AdaFace (IR-50, WebFace4M, CVLface official release, no fine-tuning)\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| AUC-ROC | {auc:.4f} |\n"
        f"| EER | {eer * 100:.2f}% |\n"
        f"| Mean cosine sim (genuine) | {mean_sim:.4f} |\n"
        f"| Mean cosine sim (impostor) | {mean_dis:.4f} |\n"
        f"| LFW-oracle accuracy (reference only) | {oracle_acc * 100:.2f}% (t={oracle_t:.3f}) |\n"
        f"| No-face fallback rate | {n_fallback[0]}/{2 * len(pairs)} images |\n",
        encoding="utf-8",
    )
    print(f"[adaface_baseline] Summary -> {summary_path}")


if __name__ == "__main__":
    main()
