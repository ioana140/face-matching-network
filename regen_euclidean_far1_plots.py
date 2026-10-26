"""
Generate the FAR=1%-threshold distance distribution plot for V4 (VGG16,
unaligned) and R4 (ResNet50, unaligned)
"""
import torch

from gradcam_alignment import (
    MODEL_SPECS, ROOT_DIR, LFW_ROOT_DEFAULT, PAIRS_CSV_DEFAULT,
    load_encoder, get_test_transform,
)
from train_siamese_methodology import compute_pair_distances
from utils.visualization import plot_distance_distribution
from data import verify_lfw_root, build_lfw_test_pairs

FAR1_THRESHOLDS = {
    "V4": 0.1842,
    "R4": 0.1817,
}
LOG_DIRS = {
    "V4": ROOT_DIR / "logs" / "VGG_Celeba_PaperSGD_DataAugmentation_L4L5_LRSchedule",
    "R4": ROOT_DIR / "logs" / "Resnet50_Celeba_PaperSgd_DiffLR_Aug_L45_LRSchedule",
}
RUN_NAMES = {
    "V4": "V4_vgg_sgd_aug_L45_lrsched",
    "R4": "V4_resnet50_paper_sgd_difLR_Aug_L45_LRSchedule",
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[regen_euclid] Device: {device}")

    lfw_root = verify_lfw_root(LFW_ROOT_DEFAULT)
    pairs = build_lfw_test_pairs(csv_path=PAIRS_CSV_DEFAULT, lfw_root=lfw_root, skip_missing=True)
    print(f"[regen_euclid] LFW pairs: {len(pairs)}")

    for key in ("V4", "R4"):
        spec = MODEL_SPECS[key]
        print(f"\n[regen_euclid] === {key} ({spec['family']}) ===")
        encoder = load_encoder(spec, device)
        transform = get_test_transform(spec)

        dists, labs = compute_pair_distances(encoder, pairs, device, transform)

        out_path = LOG_DIRS[key] / f"distance_distribution_{RUN_NAMES[key]}_far1.png"
        plot_distance_distribution(
            dists, labs,
            threshold=FAR1_THRESHOLDS[key],
            save_path=out_path,
            title=f"Distance Distribution -- {'VGG16' if key == 'V4' else 'ResNet50'}-Siamese (unaligned)",
            figsize=(10, 5.5),
            fontsize_title=17, fontsize_label=15, fontsize_tick=13, fontsize_legend=13,
            also_svg=True,
        )
        print(f"[regen_euclid] Saved -> {out_path}")


if __name__ == "__main__":
    main()
