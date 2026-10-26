"""
Verification Grad-CAM: impact of MTCNN alignment on Euclidean (contrastive)
and cosine (ArcFace) face-verification models.

"""
import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.image import scale_cam_image

from train_siamese_methodology import (
    VGG16SiameseEncoder, ResNet50SiameseEncoder, make_transforms,
)
from train_arcface_resnet import ArcFaceEncoder, _TEST_TRANSFORM as ARCFACE_TEST_TRANSFORM
from data.pair_generator import _load_csv_pairs


class SignedGradCAM(GradCAM):
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
            cam = np.abs(cam)

            self.last_raw_stats = {"mean": float(np.mean(cam)), "max": float(np.max(cam))}
            scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])
        return cam_per_target_layer


ROOT_DIR = Path(__file__).parent
LFW_ROOT_DEFAULT         = ROOT_DIR / "datasets" / "lfw" / "lfw"
LFW_ALIGNED_ROOT_DEFAULT = ROOT_DIR / "datasets" / "lfw_aligned"
PAIRS_CSV_DEFAULT        = ROOT_DIR / "datasets" / "lfw" / "pairs.csv"

CONTRASTIVE_THRESHOLD = 0.5   # margin/2, margin=1.0 (all contrastive runs)
ARCFACE_THRESHOLD     = 0.5   # cosine similarity threshold (train_arcface_resnet.py)

MODEL_SPECS = {
    "V4": dict(
        family="vgg", aligned=False, image_size=128, metric="euclidean",
        checkpoint=ROOT_DIR / "checkpoints" / "VGG_Celeba_PaperSGD_DataAugmentation_L4L5_LRSchedule"
                   / "best_V4_vgg_sgd_aug_L45_lrsched.pt",
    ),
    "V4a": dict(
        family="vgg", aligned=True, image_size=128, metric="euclidean",
        checkpoint=ROOT_DIR / "checkpoints" / "VGG_Celeba_PaperSGD_DataAugmentation_L4L5_LRSchedule_FaceAligned_FullAug"
                   / "best_V4b_vgg_sgd_full_L45_lrsched_aligned.pt",
    ),
    "R4": dict(
        family="resnet", aligned=False, image_size=224, metric="euclidean",
        checkpoint=ROOT_DIR / "checkpoints" / "Resnet50_Celeba_PaperSgd_DiffLR_Aug_L45_LRSchedule"
                   / "best_V4_resnet50_paper_sgd_difLR_Aug_L45_LRSchedule.pt",
    ),
    "R4a": dict(
        family="resnet", aligned=True, image_size=224, metric="euclidean",
        checkpoint=ROOT_DIR / "checkpoints" / "Resnet50_Celeba_PaperSgd_DiffLR_Aug_L45_LRSchedule_FaceAligned_FullAug"
                   / "best_R4a_resnet50_sgd_Aug_L45_lrsched_aligned.pt",
    ),
    "AF": dict(
        family="arcface", aligned=False, image_size=224, metric="cosine",
        checkpoint=ROOT_DIR / "checkpoints" / "Resnet50_Celeba_224x224_layer3_layer4_newMethod"
                   / "best_arcface_resnet.pt",
    ),
    "AFa": dict(
        family="arcface", aligned=True, image_size=224, metric="cosine",
        checkpoint=ROOT_DIR / "checkpoints" / "Resnet50_Celeba_224x224_layer3_layer4_newMethod_withoutTTA_alignedDatasets"
                   / "best_arcface_resnet.pt",
    ),
}

FAMILY_ORDER = ["vgg", "resnet", "arcface"]
FAMILY_LABEL = {"vgg": "VGG16-Siamese (Euclidean)",
                "resnet": "ResNet50-Siamese (Euclidean)",
                "arcface": "ResNet50-ArcFace (Cosine)"}


class EuclideanSimilarityTarget:
    def __init__(self, anchor):
        self.anchor = anchor

    def __call__(self, model_output):
        return -torch.norm(model_output - self.anchor, p=2)


class CosineSimilarityTarget:
    def __init__(self, anchor):
        self.anchor = anchor

    def __call__(self, model_output):
        return torch.dot(model_output, self.anchor)


def build_encoder(spec):
    if spec["family"] == "vgg":
        return VGG16SiameseEncoder(finetune="layer4_5", bn_head=False, image_size=spec["image_size"])
    elif spec["family"] in ("resnet", "arcface"):
        if spec["family"] == "resnet":
            return ResNet50SiameseEncoder(finetune="layer4_5", bn_head=False, image_size=spec["image_size"])
        return ArcFaceEncoder()
    raise ValueError(spec["family"])


def load_encoder(spec, device):
    encoder = build_encoder(spec)
    ckpt = torch.load(spec["checkpoint"], map_location=device, weights_only=True)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    if spec["family"] == "arcface":
        state = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
    encoder.load_state_dict(state)
    encoder.to(device)
    encoder.eval()
    return encoder


def target_layers_for(encoder, family):
    if family == "vgg":
        return [encoder.features[28]]   # last Conv2d before the final ReLU+MaxPool
    return [encoder.features[-1][-1]]   # last Bottleneck of layer4 (resnet + arcface)


def get_test_transform(spec):
    if spec["family"] == "arcface":
        return ARCFACE_TEST_TRANSFORM
    _, test_tf = make_transforms(spec["image_size"], "none")
    return test_tf


def load_pil(path):
    return Image.open(str(path)).convert("RGB")


def display_array(pil_img, image_size):
    resized = pil_img.resize((image_size, image_size), Image.BILINEAR)
    return np.asarray(resized).astype(np.float32) / 255.0


def build_pairs(n_pairs, seed, lfw_root, lfw_aligned_root, pairs_csv):
    raw_pairs, _ = _load_csv_pairs(pairs_csv, lfw_root, skip_missing=True)

    candidates = []
    for path_a, path_b, label in raw_pairs:
        aligned_a = lfw_aligned_root / path_a.relative_to(lfw_root)
        aligned_b = lfw_aligned_root / path_b.relative_to(lfw_root)
        if aligned_a.exists() and aligned_b.exists():
            candidates.append((path_a, path_b, aligned_a, aligned_b, label))

    if not candidates:
        raise RuntimeError("No LFW pairs found with both raw and aligned images present.")

    genuine  = [c for c in candidates if c[4] == 0]
    impostor = [c for c in candidates if c[4] == 1]
    rng = random.Random(seed)
    rng.shuffle(genuine)
    rng.shuffle(impostor)

    n_genuine  = (n_pairs + 1) // 2
    n_impostor = n_pairs - n_genuine
    selected = genuine[:n_genuine] + impostor[:n_impostor]
    rng.shuffle(selected)
    print(f"[gradcam] Selected {len(selected)} pairs "
          f"({sum(1 for s in selected if s[4]==0)} genuine, "
          f"{sum(1 for s in selected if s[4]==1)} impostor)")
    return selected


def run_pair_for_model(key, spec, encoder, cam, pairs, device, output_dir, csv_writer):
    test_tf = get_test_transform(spec)
    image_size = spec["image_size"]
    family_dir = output_dir / key
    family_dir.mkdir(parents=True, exist_ok=True)

    imgB_cache = []  # per-pair {"plain": uint8 HxWx3, "overlay": uint8 HxWx3} for the summary figure
    raw_mean_list = []  # flat list of raw CAM magnitudes (imgA + imgB, all pairs) for this model

    for i, (raw_a, raw_b, aligned_a, aligned_b, label) in enumerate(pairs):
        path_a, path_b = (aligned_a, aligned_b) if spec["aligned"] else (raw_a, raw_b)
        pil_a, pil_b = load_pil(path_a), load_pil(path_b)
        tensor_a = test_tf(pil_a).unsqueeze(0).to(device)
        tensor_b = test_tf(pil_b).unsqueeze(0).to(device)

        with torch.no_grad():
            emb_a = encoder(tensor_a).squeeze(0)
            emb_b = encoder(tensor_b).squeeze(0)

        if spec["metric"] == "euclidean":
            score = float(torch.norm(emb_a - emb_b, p=2).item())
            same_pred = score < CONTRASTIVE_THRESHOLD
            target_cls = EuclideanSimilarityTarget
        else:
            score = float(torch.dot(emb_a, emb_b).item())
            same_pred = score >= ARCFACE_THRESHOLD
            target_cls = CosineSimilarityTarget

        correct = same_pred == (label == 0)
        kind = "genuine" if label == 0 else "impostor"

        pair_imgB = None
        raw_means = {}
        for tag, vis_tensor, vis_pil, anchor in (
            ("imgA", tensor_a, pil_a, emb_b),
            ("imgB", tensor_b, pil_b, emb_a),
        ):
            grayscale_cam = cam(input_tensor=vis_tensor, targets=[target_cls(anchor)])[0]
            raw_means[tag] = cam.last_raw_stats["mean"]
            raw_mean_list.append(cam.last_raw_stats["mean"])
            display_img = display_array(vis_pil, image_size)
            overlay = show_cam_on_image(display_img, grayscale_cam, use_rgb=True)
            out_path = family_dir / f"pair{i:02d}_{kind}_{tag}.png"
            Image.fromarray(overlay).save(out_path)
            if tag == "imgB":
                pair_imgB = {"plain": (display_img * 255).astype(np.uint8), "overlay": overlay}
        imgB_cache.append(pair_imgB)

        csv_writer.writerow([i, kind, key, spec["aligned"], f"{score:.6f}", same_pred, correct,
                              f"{raw_means['imgA']:.8f}", f"{raw_means['imgB']:.8f}"])

    print(f"[gradcam] {key}: wrote {len(pairs) * 2} heatmaps -> {family_dir}")
    return imgB_cache, raw_mean_list


def make_summary_figure(family, pairs, summary_cache, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    unaligned_key = next(k for k, s in MODEL_SPECS.items() if s["family"] == family and not s["aligned"])
    aligned_key   = next(k for k, s in MODEL_SPECS.items() if s["family"] == family and s["aligned"])
    unaligned_cache = summary_cache[unaligned_key]
    aligned_cache   = summary_cache[aligned_key]

    n = len(pairs)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.2 * n))
    if n == 1:
        axes = axes[None, :]
    col_titles = ["Unaligned - image", "Unaligned - Grad-CAM", "Aligned - image", "Aligned - Grad-CAM"]

    for i, (_, _, _, _, label) in enumerate(pairs):
        kind = "genuine" if label == 0 else "impostor"
        imgs = (unaligned_cache[i]["plain"], unaligned_cache[i]["overlay"],
                aligned_cache[i]["plain"], aligned_cache[i]["overlay"])
        for j, img in enumerate(imgs):
            axes[i, j].imshow(img)
            axes[i, j].axis("off")
        axes[i, 0].text(-0.15, 0.5, kind, transform=axes[i, 0].transAxes,
                         rotation=90, va="center", ha="center", fontsize=10)

    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=11)

    fig.suptitle(f"Alignment impact — {FAMILY_LABEL[family]}", fontsize=13)
    fig.tight_layout()
    out_path = output_dir / f"summary_{family}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[gradcam] Summary figure -> {out_path}")


def make_magnitude_figure(raw_stats, output_dir):

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(FAMILY_ORDER), figsize=(5 * len(FAMILY_ORDER), 4.5))
    for ax, family in zip(axes, FAMILY_ORDER):
        unaligned_key = next(k for k, s in MODEL_SPECS.items() if s["family"] == family and not s["aligned"])
        aligned_key   = next(k for k, s in MODEL_SPECS.items() if s["family"] == family and s["aligned"])
        u_vals = np.array(raw_stats[unaligned_key])
        a_vals = np.array(raw_stats[aligned_key])
        means = [u_vals.mean(), a_vals.mean()]
        stds  = [u_vals.std(), a_vals.std()]
        ax.bar(["Unaligned", "Aligned"], means, yerr=stds, capsize=6,
               color=["#6f9fd8", "#d87f6f"])
        ax.set_title(FAMILY_LABEL[family], fontsize=10)
        ax.set_ylabel("Mean |Grad-CAM| (raw units)")

    fig.suptitle("Raw Grad-CAM signal strength — aligned vs unaligned", fontsize=13)
    fig.tight_layout()
    out_path = output_dir / "magnitude_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[gradcam] Magnitude comparison -> {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-pairs", type=int, default=4)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--lfw-root", type=Path, default=LFW_ROOT_DEFAULT)
    p.add_argument("--lfw-aligned-root", type=Path, default=LFW_ALIGNED_ROOT_DEFAULT)
    p.add_argument("--pairs-csv", type=Path, default=PAIRS_CSV_DEFAULT)
    p.add_argument("--output-dir", type=Path, default=ROOT_DIR / "gradcam_results")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gradcam] Device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs = build_pairs(args.n_pairs, args.seed, args.lfw_root, args.lfw_aligned_root, args.pairs_csv)

    summary_cache = {}
    raw_stats = {}
    csv_path = args.output_dir / "scores.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_idx", "kind", "model", "aligned", "score", "predicted_same", "correct",
                          "raw_cam_mean_imgA", "raw_cam_mean_imgB"])

        for key, spec in MODEL_SPECS.items():
            print(f"\n[gradcam] === {key} ({spec['family']}, aligned={spec['aligned']}) ===")
            encoder = load_encoder(spec, device)
            target_layers = target_layers_for(encoder, spec["family"])
            with SignedGradCAM(model=encoder, target_layers=target_layers) as cam:
                summary_cache[key], raw_stats[key] = run_pair_for_model(
                    key, spec, encoder, cam, pairs, device, args.output_dir, writer)

    for family in FAMILY_ORDER:
        make_summary_figure(family, pairs, summary_cache, args.output_dir)
    make_magnitude_figure(raw_stats, args.output_dir)

    print(f"\n[gradcam] Done. Scores -> {csv_path}")


if __name__ == "__main__":
    main()
