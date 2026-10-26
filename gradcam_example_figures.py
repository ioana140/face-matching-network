
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from facenet_pytorch import MTCNN
from pytorch_grad_cam.utils.image import show_cam_on_image

from gradcam_alignment import (
    MODEL_SPECS, FAMILY_LABEL, ROOT_DIR,
    LFW_ROOT_DEFAULT, LFW_ALIGNED_ROOT_DEFAULT, PAIRS_CSV_DEFAULT,
    load_encoder, target_layers_for, get_test_transform, load_pil, build_pairs,
    EuclideanSimilarityTarget, CosineSimilarityTarget, display_array,
)
from gradcam_face_region_metric import RawCaptureGradCAM, detect_face_box, box_mask, energy_ratio

UNALIGNED_KEYS = ["V4", "R4", "AF"]
ALIGNED_KEYS = ["V4a", "R4a", "AFa"]


def render_panels(keys, pairs, use_aligned_images, device, mtcnn, output_path, condition_label):
    encoders, transforms_, layers = {}, {}, {}
    for key in keys:
        spec = MODEL_SPECS[key]
        enc = load_encoder(spec, device)
        encoders[key] = enc
        transforms_[key] = get_test_transform(spec)
        layers[key] = target_layers_for(enc, spec["family"])

    fig, axes = plt.subplots(len(pairs), len(keys), figsize=(4.2 * len(keys), 4.2 * len(pairs)))
    if len(pairs) == 1:
        axes = axes[None, :]

    for row, (raw_a, raw_b, aligned_a, aligned_b, label) in enumerate(pairs):
        kind = "genuine" if label == 0 else "impostor"
        path_a, path_b = (aligned_a, aligned_b) if use_aligned_images else (raw_a, raw_b)
        pil_a, pil_b = load_pil(path_a), load_pil(path_b)

        for col, key in enumerate(keys):
            spec = MODEL_SPECS[key]
            image_size = spec["image_size"]
            test_tf = transforms_[key]
            tensor_a = test_tf(pil_a).unsqueeze(0).to(device)
            tensor_b = test_tf(pil_b).unsqueeze(0).to(device)

            with torch.no_grad():
                emb_a = encoders[key](tensor_a).squeeze(0)
                emb_b = encoders[key](tensor_b).squeeze(0)

            target_cls = EuclideanSimilarityTarget if spec["metric"] == "euclidean" else CosineSimilarityTarget

            with RawCaptureGradCAM(model=encoders[key], target_layers=layers[key]) as cam:
                grayscale_cam = cam(input_tensor=tensor_b, targets=[target_cls(emb_a)])[0]
                raw_map = cam.last_raw_map[0]

            display_img = display_array(pil_b, image_size)
            overlay = show_cam_on_image(display_img, grayscale_cam, use_rgb=True)

            box = detect_face_box(mtcnn, pil_b, image_size)
            ratio = energy_ratio(raw_map, box_mask(box, image_size)) if box is not None else None

            ax = axes[row, col]
            ax.imshow(overlay)
            if box is not None:
                x1, y1, x2, y2 = box
                ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                                linewidth=2, edgecolor="white", facecolor="none"))
            ratio_txt = f"{ratio*100:.0f}% in face" if ratio is not None else "no face detected"
            ax.text(0.5, -0.08, ratio_txt, transform=ax.transAxes, ha="center", va="top",
                     fontsize=11, fontweight="bold")
            ax.axis("off")
            if row == 0:
                ax.set_title(FAMILY_LABEL[spec["family"]], fontsize=11)
            if col == 0:
                ax.text(-0.08, 0.5, kind, transform=ax.transAxes, rotation=90, va="center", ha="center", fontsize=10)

    fig.suptitle(f"Grad-CAM energy: white box = detected face region, % = energy inside it ({condition_label} inputs)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[example_figures] Saved -> {output_path}")
    svg_path = str(output_path).rsplit(".", 1)[0] + ".svg"
    fig.savefig(svg_path, bbox_inches="tight")
    print(f"[example_figures] Saved -> {svg_path}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-pairs", type=int, default=3)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--output-dir", type=str, default=str(ROOT_DIR / "gradcam_results" / "face_region_metric"))
    p.add_argument("--tag", type=str, default="",
                    help="Suffix for output filenames (e.g. 'set2') so repeated runs don't overwrite earlier ones.")
    p.add_argument("--pair-indices", type=str, default="",
                    help="Comma-separated pair indices to hand-pick from the pool "
                         "(e.g. from face_region_ratios.csv analysis) instead of taking "
                         "the first --n-pairs. Regenerates the pool with --n-pairs/--seed "
                         "(use --n-pairs 200 --seed 12345 to match face_region_ratios.csv) "
                         "and selects only these indices, in order.")
    args = p.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pool = build_pairs(args.n_pairs, args.seed, LFW_ROOT_DEFAULT, LFW_ALIGNED_ROOT_DEFAULT, PAIRS_CSV_DEFAULT)
    if args.pair_indices:
        idx_list = [int(x) for x in args.pair_indices.split(",")]
        pairs = [pool[i] for i in idx_list]
    else:
        pairs = pool
    mtcnn = MTCNN(device=device, select_largest=True, post_process=False, keep_all=False)

    render_panels(UNALIGNED_KEYS, pairs, use_aligned_images=False, device=device, mtcnn=mtcnn,
                  output_path=f"{args.output_dir}/example_panels{suffix}.png", condition_label="unaligned")
    render_panels(ALIGNED_KEYS, pairs, use_aligned_images=True, device=device, mtcnn=mtcnn,
                  output_path=f"{args.output_dir}/example_panels_aligned{suffix}.png", condition_label="aligned")


if __name__ == "__main__":
    main()
