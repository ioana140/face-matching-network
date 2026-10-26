"""
Generates the cosine similarity distribution plots (FAR=1% threshold and
threshold=0.5) for AF (ArcFace-ResNet50, unaligned)
"""
from pathlib import Path

import torch

from train_arcface_resnet import ArcFaceModel, compute_pair_similarities, load_celeba_identity_data
from utils.visualization import plot_distance_distribution
from data import verify_lfw_root, build_image_catalogue, build_lfw_test_pairs

ROOT = Path(__file__).parent
CELEBA_ROOT          = ROOT / "datasets" / "celeba" / "img_align_celeba"
CELEBA_IDENTITY_FILE = ROOT / "datasets" / "celeba" / "identity_CelebA.txt"
LFW_ROOT             = ROOT / "datasets" / "lfw" / "lfw"
PAIRS_CSV            = ROOT / "datasets" / "lfw" / "pairs.csv"

CHECKPOINT = ROOT / "checkpoints" / "Resnet50_Celeba_224x224_layer3_layer4_newMethod" / "best_arcface_resnet.pt"
LOG_DIR    = ROOT / "logs" / "Resnet50_Celeba_224x224_layer3_layer4_newMethod"

FAR1_THRESHOLD = 0.2892   # from master_results.csv: AF, target FAR=1%


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[regen_af] Device: {device}")

    print("[regen_af] Loading CelebA identity data ...")
    (_, _, _, _, num_classes, _, _) = load_celeba_identity_data(CELEBA_ROOT, CELEBA_IDENTITY_FILE)

    lfw_root = verify_lfw_root(LFW_ROOT)
    build_image_catalogue(lfw_root)
    test_pairs = build_lfw_test_pairs(csv_path=PAIRS_CSV, lfw_root=lfw_root, skip_missing=True)
    print(f"[regen_af] LFW pairs: {len(test_pairs)}")

    print(f"[regen_af] Loading checkpoint: {CHECKPOINT}")
    model = ArcFaceModel(num_classes=num_classes)
    ckpt = torch.load(str(CHECKPOINT), map_location=device, weights_only=True)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device).eval()

    print("[regen_af] Running LFW inference ...")
    sims, labs = compute_pair_similarities(model, test_pairs, device)

    out_far1 = LOG_DIR / "distance_distribution_arcface_resnet_far1.png"
    plot_distance_distribution(
        sims, labs, threshold=FAR1_THRESHOLD, save_path=out_far1,
        title="Similarity Distribution -- ArcFace-ResNet50 (unaligned)",
        x_label="Cosine Similarity",
        figsize=(10, 5.5),
        fontsize_title=17, fontsize_label=15, fontsize_tick=13, fontsize_legend=13,
        also_svg=True,
    )
    print(f"[regen_af] Saved -> {out_far1}")

    out_05 = LOG_DIR / "distance_distribution_arcface_resnet_t05.png"
    plot_distance_distribution(
        sims, labs, threshold=0.5, save_path=out_05,
        title="Similarity Distribution -- ArcFace-ResNet50 (unaligned)",
        x_label="Cosine Similarity",
        figsize=(10, 5.5),
        fontsize_title=17, fontsize_label=15, fontsize_tick=13, fontsize_legend=13,
        also_svg=True,
    )
    print(f"[regen_af] Saved -> {out_05}")


if __name__ == "__main__":
    main()
