"""
Official InsightFace ArcFace baseline (buffalo_l, ResNet50-based)
"""
import csv
import time
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics import roc_auc_score, roc_curve

ROOT_DIR = Path(__file__).parent
LFW_ROOT = ROOT_DIR / "datasets" / "lfw" / "lfw"
PAIRS_CSV = ROOT_DIR / "datasets" / "lfw" / "pairs.csv"
OUTPUT_DIR = ROOT_DIR / "results_summary"

LABEL_SIMILAR = 0
LABEL_DISSIMILAR = 1


def image_path(lfw_root, person_name, image_num):
    filename = f"{person_name}_{int(image_num):04d}.jpg"
    return lfw_root / person_name / filename


def load_csv_pairs(csv_path, lfw_root, skip_missing=True):
    pairs = []
    skipped = 0
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            cols = [c.strip() for c in row if c.strip()]
            if not cols or cols[0].lower() in ("name", "name1"):
                continue
            if len(cols) == 3:
                name, n1, n2 = cols
                path_a = image_path(lfw_root, name, n1)
                path_b = image_path(lfw_root, name, n2)
                label = LABEL_SIMILAR
            elif len(cols) == 4:
                name1, n1, name2, n2 = cols
                path_a = image_path(lfw_root, name1, n1)
                path_b = image_path(lfw_root, name2, n2)
                label = LABEL_DISSIMILAR
            else:
                continue
            if skip_missing and (not path_a.exists() or not path_b.exists()):
                skipped += 1
                continue
            pairs.append((path_a, path_b, label))
    n_sim = sum(1 for _, _, l in pairs if l == LABEL_SIMILAR)
    n_dis = sum(1 for _, _, l in pairs if l == LABEL_DISSIMILAR)
    print(f"[insightface_baseline] CSV loaded: {len(pairs)} pairs ({n_sim} similar, {n_dis} dissimilar)"
          + (f", {skipped} skipped (missing files)" if skipped else ""))
    return pairs


def get_embedding(app, path, embed_cache):
    if path in embed_cache:
        return embed_cache[path]
    img = cv2.imread(str(path))
    if img is None:
        embed_cache[path] = None
        return None
    faces = app.get(img)
    if not faces:
        embed_cache[path] = None
        return None
    areas = [(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) for f in faces]
    face = faces[int(np.argmax(areas))]
    emb = face.normed_embedding
    embed_cache[path] = emb
    return emb


def threshold_sweep(similarities, labels):
    best_t, best_acc = 0.5, 0.0
    for t in np.arange(-1.0, 1.0, 0.005):
        preds = (similarities < t).astype(int)
        acc = float((preds == labels).mean())
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t, best_acc


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = load_csv_pairs(PAIRS_CSV, LFW_ROOT, skip_missing=True)

    checkpoint_path = OUTPUT_DIR / "baseline_insightface_scores.csv"
    done = []
    if checkpoint_path.exists():
        with open(checkpoint_path, newline="") as f:
            reader = csv.DictReader(f)
            done = [(float(r["cosine_similarity"]), int(r["label"]))
                    for r in reader if r.get("cosine_similarity") != "NA"]
        print(f"[insightface_baseline] Resuming: {len(done)} pairs already scored in a previous run")

    print("[insightface_baseline] Loading buffalo_l (detection + recognition only, no landmark/genderage) ...")
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=0, det_size=(160, 160))

    embed_cache = {}
    similarities = [s for s, _ in done]
    labels = [l for _, l in done]
    n_no_face = 0
    t0 = time.time()

    checkpoint_file = open(checkpoint_path, "a" if done else "w", newline="")
    writer = csv.writer(checkpoint_file)
    if not done:
        writer.writerow(["cosine_similarity", "label"])

    for i, (path_a, path_b, label) in enumerate(pairs):
        if i < len(done):
            continue
        emb_a = get_embedding(app, path_a, embed_cache)
        emb_b = get_embedding(app, path_b, embed_cache)
        if emb_a is None or emb_b is None:
            n_no_face += 1
            writer.writerow(["NA", label])
            checkpoint_file.flush()
            continue
        sim = float(np.dot(emb_a, emb_b))
        similarities.append(sim)
        labels.append(label)
        writer.writerow([f"{sim:.6f}", label])
        checkpoint_file.flush()
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            done_now = i + 1 - len(done)
            rate = done_now / elapsed if elapsed > 0 else 0
            eta = (len(pairs) - (i + 1)) / rate / 60 if rate > 0 else float("nan")
            print(f"[insightface_baseline]   {i + 1}/{len(pairs)} pairs "
                  f"({rate:.2f} pairs/s, ETA {eta:.1f} min)")

    checkpoint_file.close()
    similarities = np.asarray(similarities)
    labels = np.asarray(labels)
    print(f"\n[insightface_baseline] Scored {len(labels)} / {len(pairs)} pairs "
          f"({n_no_face} pairs dropped this run -- no face detected in at least one image)")

    auc = float(roc_auc_score(labels, 1.0 - similarities))
    fpr, tpr, _ = roc_curve(labels, 1.0 - similarities)
    fnr = 1 - tpr
    eer = float(fpr[int(np.nanargmin(np.abs(fpr - fnr)))])

    oracle_t, oracle_acc = threshold_sweep(similarities, labels)
    mean_sim = float(similarities[labels == 0].mean())
    mean_dis = float(similarities[labels == 1].mean())

    print("\n" + "=" * 70)
    print("  BASELINE -- InsightFace ArcFace (buffalo_l, official pretrained)")
    print("=" * 70)
    print(f"  AUC-ROC                          {auc:.4f}")
    print(f"  EER                              {eer * 100:.2f} %")
    print(f"  Mean cosine sim -- genuine        {mean_sim:.4f}")
    print(f"  Mean cosine sim -- impostor       {mean_dis:.4f}")
    print(f"  LFW-oracle threshold sweep -> {oracle_t:.3f}  acc={oracle_acc * 100:.2f}%  "
          f"(tuned ON the test set, cite as reference only, not an operating point)")
    print("=" * 70)

    csv_path = OUTPUT_DIR / "baseline_insightface_scores.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cosine_similarity", "label"])
        for s, l in zip(similarities, labels):
            writer.writerow([f"{s:.6f}", int(l)])
    print(f"[insightface_baseline] Raw scores -> {csv_path}")

    summary_path = OUTPUT_DIR / "baseline_insightface_summary.md"
    summary_path.write_text(
        "## Baseline -- InsightFace ArcFace (buffalo_l, official pretrained, no fine-tuning)\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| AUC-ROC | {auc:.4f} |\n"
        f"| EER | {eer * 100:.2f}% |\n"
        f"| Mean cosine sim (genuine) | {mean_sim:.4f} |\n"
        f"| Mean cosine sim (impostor) | {mean_dis:.4f} |\n"
        f"| LFW-oracle accuracy (reference only) | {oracle_acc * 100:.2f}% (t={oracle_t:.3f}) |\n"
        f"| Pairs dropped (no face detected) | {n_no_face}/{len(pairs)} |\n",
        encoding="utf-8",
    )
    print(f"[insightface_baseline] Summary -> {summary_path}")


if __name__ == "__main__":
    main()