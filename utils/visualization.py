from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[Path] = None,
) -> None:

    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    if "val_loss" in history:
        axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_title("Contrastive Loss (Eq. 1)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    train_acc_key = "train_acc" if "train_acc" in history else "train_accuracy"
    axes[1].plot(epochs, [a*100 for a in history[train_acc_key]], label="Train Acc")
    if "val_acc" in history:
        axes[1].plot(epochs, [a*100 for a in history["val_acc"]], label="Val Acc")
    axes[1].axhline(95.62, color="r", linestyle="--", label="Paper target (95.62%)")
    axes[1].set_title("Face Recognition Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle("Siamese-VGG Training (Heidari & Fouladi-Ghaleh, 2021)")
    plt.tight_layout()

    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"[visualization] Saved training plot -> {save_path}")
    plt.close(fig)


def plot_pairs(
    pairs:      List[Tuple],
    n_pairs:    int = 8,
    save_path:  Optional[Path] = None,
    image_size: Tuple[int, int] = (128, 128),
) -> None:
    try:
        import cv2
        def _load(p):
            img = cv2.imread(str(p))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return cv2.resize(img, image_size[::-1])
    except ImportError:
        from PIL import Image as PILImage
        def _load(p):
            return np.array(PILImage.open(str(p)).convert("RGB").resize(image_size[::-1]))

    plt = _plt()
    sampled = pairs[:n_pairs]
    n = len(sampled)
    fig, axes = plt.subplots(n, 2, figsize=(5, n * 2.5))
    if n == 1:
        axes = [axes]

    for i, (pa, pb, label) in enumerate(sampled):
        for j, (ax, path) in enumerate(zip(axes[i], [pa, pb])):
            ax.imshow(_load(path))
            ax.axis("off")
        color = "green" if label == 0 else "red"
        lbl   = "Similar (0)" if label == 0 else "Dissimilar (1)"
        axes[i][0].set_title(lbl, color=color, fontsize=9)

    plt.suptitle("Sample Image Pairs (Fig. 4 style)", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"[visualization] Saved pairs plot -> {save_path}")
    plt.close(fig)


def plot_distance_distribution(
    distances:      np.ndarray,
    labels:         np.ndarray,
    threshold:      float = 0.5,
    save_path:      Optional[Path] = None,
    title:          str = "Distance Distribution — Siamese",
    x_label:        str = "Euclidean Distance",
    figsize:        Tuple[float, float] = (10, 5),
    fontsize_title: Optional[float] = None,
    fontsize_label: Optional[float] = None,
    fontsize_tick:  Optional[float] = None,
    fontsize_legend: Optional[float] = None,
    also_svg:       bool = False,
) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(distances[labels == 0], bins=60, alpha=0.6,
            label="Similar (y=0)", color="steelblue", density=True)
    ax.hist(distances[labels == 1], bins=60, alpha=0.6,
            label="Dissimilar (y=1)", color="tomato", density=True)
    ax.axvline(threshold, color="black", linestyle="--",
               label=f"Threshold = {threshold}")
    ax.set_xlabel(x_label, fontsize=fontsize_label)
    ax.set_ylabel("Density", fontsize=fontsize_label)
    ax.set_title(title, fontsize=fontsize_title)
    if fontsize_tick is not None:
        ax.tick_params(axis="both", labelsize=fontsize_tick)
    ax.legend(fontsize=fontsize_legend); ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"[visualization] Saved distance plot -> {save_path}")
        if also_svg:
            svg_path = Path(save_path).with_suffix(".svg")
            plt.savefig(str(svg_path), bbox_inches="tight")
            print(f"[visualization] Saved distance plot -> {svg_path}")
    plt.close(fig)


def plot_confusion_matrix(
    y_true:    np.ndarray,
    y_pred:    np.ndarray,
    save_path: Optional[Path] = None,
    title:     str = "Confusion Matrix — Siamese",
) -> None:
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    plt = _plt()
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Similar (0)", "Dissimilar (1)"],
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"[visualization] Saved confusion matrix -> {save_path}")
    plt.close(fig)
