
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from pytorch_grad_cam.utils.image import scale_cam_image
from scipy import stats

from gradcam_alignment import (
    SignedGradCAM, MODEL_SPECS, FAMILY_ORDER, FAMILY_LABEL,
    ROOT_DIR, LFW_ROOT_DEFAULT, LFW_ALIGNED_ROOT_DEFAULT, PAIRS_CSV_DEFAULT,
    load_encoder, target_layers_for, get_test_transform, load_pil, build_pairs,
    EuclideanSimilarityTarget, CosineSimilarityTarget,
)


class RawCaptureGradCAM(SignedGradCAM):

    def compute_cam_per_layer(self, input_tensor, targets, eigen_smooth):
        if self.detach:
            activations_list = [a.cpu().data.numpy() for a in self.activations_and_grads.activations]
            grads_list = [g.cpu().data.numpy() for g in self.activations_and_grads.gradients]
        else:
            activations_list = [a for a in self.activations_and_grads.activations]
            grads_list = [g for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        for i, target_layer in enumerate(self.target_layers):
            layer_activations = activations_list[i] if i < len(activations_list) else None
            layer_grads = grads_list[i] if i < len(grads_list) else None
            cam = self.get_cam_image(input_tensor, target_layer, targets,
                                      layer_activations, layer_grads, eigen_smooth)
            cam = np.abs(cam)  # (batch, H, W), raw units
            self.last_raw_stats = {"mean": float(np.mean(cam)), "max": float(np.max(cam))}
            self.last_raw_map = np.float32(
                [cv2.resize(np.float32(c), target_size) for c in cam]
            )
            scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])
        return cam_per_target_layer


def detect_face_box(mtcnn, pil_img, image_size):

    orig_w, orig_h = pil_img.size
    boxes, _ = mtcnn.detect(pil_img)
    if boxes is None or len(boxes) == 0:
        return None
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    box = boxes[int(np.argmax(areas))]
    sx, sy = image_size / orig_w, image_size / orig_h
    x1, x2 = sorted((box[0] * sx, box[2] * sx))
    y1, y2 = sorted((box[1] * sy, box[3] * sy))
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(image_size), x2), min(float(image_size), y2)
    return x1, y1, x2, y2


def box_mask(box, image_size):
    mask = np.zeros((image_size, image_size), dtype=bool)
    if box is None:
        return mask
    x1, y1, x2, y2 = box
    mask[int(round(y1)):int(round(y2)), int(round(x1)):int(round(x2))] = True
    return mask


def energy_ratio(raw_map, mask):
    total = raw_map.sum()
    if total <= 0:
        return None
    return float(raw_map[mask].sum() / total)


def run(args):
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[facebg] Device: {device}")

    pairs = build_pairs(args.n_pairs, args.seed, args.lfw_root, args.lfw_aligned_root, args.pairs_csv)
    mtcnn = MTCNN(device=device, select_largest=True, post_process=False, keep_all=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "face_region_ratios.csv"
    rows = []
    n_missing = 0

    for key, spec in MODEL_SPECS.items():
        print(f"\n[facebg] === {key} ({spec['family']}, aligned={spec['aligned']}) ===")
        encoder = load_encoder(spec, device)
        target_layers = target_layers_for(encoder, spec["family"])
        test_tf = get_test_transform(spec)
        image_size = spec["image_size"]
        target_cls = EuclideanSimilarityTarget if spec["metric"] == "euclidean" else CosineSimilarityTarget

        with RawCaptureGradCAM(model=encoder, target_layers=target_layers) as cam:
            for i, (raw_a, raw_b, aligned_a, aligned_b, label) in enumerate(pairs):
                path_a, path_b = (aligned_a, aligned_b) if spec["aligned"] else (raw_a, raw_b)
                pil_a, pil_b = load_pil(path_a), load_pil(path_b)
                tensor_a = test_tf(pil_a).unsqueeze(0).to(device)
                tensor_b = test_tf(pil_b).unsqueeze(0).to(device)

                with torch.no_grad():
                    emb_a = encoder(tensor_a).squeeze(0)
                    emb_b = encoder(tensor_b).squeeze(0)

                kind = "genuine" if label == 0 else "impostor"

                for tag, vis_tensor, vis_pil, anchor in (
                    ("imgA", tensor_a, pil_a, emb_b),
                    ("imgB", tensor_b, pil_b, emb_a),
                ):
                    _ = cam(input_tensor=vis_tensor, targets=[target_cls(anchor)])
                    raw_map = cam.last_raw_map[0]
                    box = detect_face_box(mtcnn, vis_pil, image_size)
                    if box is None:
                        n_missing += 1
                        ratio = None
                    else:
                        ratio = energy_ratio(raw_map, box_mask(box, image_size))
                    rows.append(dict(
                        pair_idx=i, kind=kind, model=key, family=spec["family"],
                        aligned=spec["aligned"], tag=tag, ratio=ratio,
                    ))
                if (i + 1) % 25 == 0:
                    print(f"[facebg]   {key}: {i + 1}/{len(pairs)} pairs")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_idx", "kind", "model", "family", "aligned", "tag", "ratio"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[facebg] Missing face detections: {n_missing} / {len(rows)} images")
    print(f"[facebg] Wrote {len(rows)} rows -> {csv_path}")
    return rows


def summarize(rows, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_model = {}
    for r in rows:
        if r["ratio"] is None:
            continue
        by_model.setdefault(r["model"], []).append(r["ratio"])

    print("\n" + "=" * 78)
    print("  IN-FACE-BBOX GRAD-CAM ENERGY RATIO  (1.0 = all energy inside the face box)")
    print("=" * 78)
    print(f"  {'Model':6} {'family':8} {'aligned':8} {'n':>5} {'mean':>7} {'std':>7} {'95% CI':>18}")
    summary = {}
    for key, spec in MODEL_SPECS.items():
        vals = np.array(by_model.get(key, []))
        if len(vals) == 0:
            continue
        mean, std = vals.mean(), vals.std(ddof=1)
        se = std / np.sqrt(len(vals))
        ci = (mean - 1.96 * se, mean + 1.96 * se)
        summary[key] = dict(mean=mean, std=std, n=len(vals), ci=ci)
        print(f"  {key:6} {spec['family']:8} {str(spec['aligned']):8} {len(vals):5d} "
              f"{mean:7.3f} {std:7.3f}   [{ci[0]:.3f}, {ci[1]:.3f}]")
    print("=" * 78)

    by_key_map = {}
    for r in rows:
        by_key_map.setdefault(r["model"], {})[(r["pair_idx"], r["tag"])] = r["ratio"]

    print("\n[facebg] Test 1 -- Wilcoxon signed-rank: aligned vs unaligned (paired per image, sanity check)")
    for family in FAMILY_ORDER:
        unaligned_key = next(k for k, s in MODEL_SPECS.items() if s["family"] == family and not s["aligned"])
        aligned_key   = next(k for k, s in MODEL_SPECS.items() if s["family"] == family and s["aligned"])
        common = set(by_key_map[unaligned_key]) & set(by_key_map[aligned_key])
        u_vals, a_vals = [], []
        for k in common:
            uv, av = by_key_map[unaligned_key][k], by_key_map[aligned_key][k]
            if uv is not None and av is not None:
                u_vals.append(uv); a_vals.append(av)
        if len(u_vals) < 10:
            print(f"  {FAMILY_LABEL[family]}: too few paired samples ({len(u_vals)})")
            continue
        _, p = stats.wilcoxon(u_vals, a_vals)
        direction = "LOWER" if np.mean(a_vals) < np.mean(u_vals) else "HIGHER"
        print(f"  {FAMILY_LABEL[family]:30}  n={len(u_vals):4d}  "
              f"unaligned={np.mean(u_vals):.3f}  aligned={np.mean(a_vals):.3f}  "
              f"aligned is {direction}  p={p:.2e}")

    print("\n[facebg] Test 2 -- Wilcoxon signed-rank: unaligned Euclidean vs unaligned ArcFace")
    print("           (SAME source images across families -- this is the decisive test)")
    af_key = next(k for k, s in MODEL_SPECS.items() if s["family"] == "arcface" and not s["aligned"])
    for family in ("vgg", "resnet"):
        eu_key = next(k for k, s in MODEL_SPECS.items() if s["family"] == family and not s["aligned"])
        common = set(by_key_map[eu_key]) & set(by_key_map[af_key])
        e_vals, a_vals = [], []
        for k in common:
            ev, av = by_key_map[eu_key][k], by_key_map[af_key][k]
            if ev is not None and av is not None:
                e_vals.append(ev); a_vals.append(av)
        if len(e_vals) < 10:
            print(f"  {FAMILY_LABEL[family]} vs ArcFace: too few paired samples ({len(e_vals)})")
            continue
        _, p = stats.wilcoxon(e_vals, a_vals)
        direction = "LOWER" if np.mean(e_vals) < np.mean(a_vals) else "HIGHER"
        print(f"  {FAMILY_LABEL[family]:30} vs ArcFace  n={len(e_vals):4d}  "
              f"{family}={np.mean(e_vals):.3f}  arcface={np.mean(a_vals):.3f}  "
              f"{family} is {direction} than ArcFace  p={p:.2e}")

    fig, ax = plt.subplots(figsize=(9, 5))
    order = list(MODEL_SPECS.keys())
    data = [by_model.get(k, []) for k in order]
    colors = ["#6f9fd8" if not MODEL_SPECS[k]["aligned"] else "#d87f6f" for k in order]
    bp = ax.boxplot(data, tick_labels=order, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_ylabel("Fraction of Grad-CAM energy inside face bbox")
    ax.set_title("Face vs. background Grad-CAM energy, by model\n(blue = unaligned, red = aligned)")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    out_path = output_dir / "face_region_ratio_boxplot.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n[facebg] Plot -> {out_path}")

    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-pairs", type=int, default=200)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--lfw-root", type=Path, default=LFW_ROOT_DEFAULT)
    p.add_argument("--lfw-aligned-root", type=Path, default=LFW_ALIGNED_ROOT_DEFAULT)
    p.add_argument("--pairs-csv", type=Path, default=PAIRS_CSV_DEFAULT)
    p.add_argument("--output-dir", type=Path, default=ROOT_DIR / "gradcam_results" / "face_region_metric")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    rows = run(args)
    summarize(rows, args.output_dir)


if __name__ == "__main__":
    main()