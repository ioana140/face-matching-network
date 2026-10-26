

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config import DECISION_THRESHOLD
from model.distance_layer import predict_labels


def collect_predictions(
    model:  nn.Module,
    loader: DataLoader,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    all_dist:   List[float] = []
    all_labels: List[int]   = []

    with torch.no_grad():
        for img_a, img_b, labels in loader:
            img_a  = img_a.to(device,  non_blocking=True)
            img_b  = img_b.to(device,  non_blocking=True)
            dist   = model(img_a, img_b).cpu().numpy()
            all_dist.extend(dist.tolist())
            all_labels.extend(labels.numpy().astype(int).tolist())

    return (
        np.array(all_dist,   dtype=np.float32),
        np.array(all_labels, dtype=np.int32),
    )


def compute_eer(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> Tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer, float(thresholds[idx])


def evaluate_model(
    model:     nn.Module,
    loader:    DataLoader,
    threshold: float = DECISION_THRESHOLD,
    verbose:   bool  = True,
    device:    Optional[torch.device] = None,
    wandb_run  = None,
) -> Dict[str, float]:
    distances, labels = collect_predictions(model, loader, device)
    preds = (distances >= threshold).astype(np.int32)

    # Core metrics
    acc         = float(accuracy_score(labels, preds))
    auc         = float(roc_auc_score(labels, distances))
    ap          = float(average_precision_score(labels, distances))

    prec_w  = float(precision_score(labels, preds, average="weighted", zero_division=0))
    rec_w   = float(recall_score(   labels, preds, average="weighted", zero_division=0))
    f1_w    = float(f1_score(       labels, preds, average="weighted", zero_division=0))

    prec_m  = float(precision_score(labels, preds, average="macro",    zero_division=0))
    rec_m   = float(recall_score(   labels, preds, average="macro",    zero_division=0))
    f1_m    = float(f1_score(       labels, preds, average="macro",    zero_division=0))

    report  = classification_report(
        labels, preds,
        target_names=["similar_0", "dissimilar_1"],
        output_dict=True, zero_division=0,
    )

    eer, eer_thresh = compute_eer(labels, distances)

    sim_mask  = labels == 0
    dis_mask  = labels == 1
    mean_sim  = float(distances[sim_mask].mean())  if sim_mask.any()  else float("nan")
    mean_dis  = float(distances[dis_mask].mean())  if dis_mask.any()  else float("nan")

    metrics: Dict[str, float] = {
        "accuracy":                       acc,
        "accuracy_pct":                   acc * 100,

        "precision_weighted":             prec_w,
        "recall_weighted":                rec_w,
        "f1_weighted":                    f1_w,

        "precision_macro":                prec_m,
        "recall_macro":                   rec_m,
        "f1_macro":                       f1_m,

        "precision_similar":              float(report["similar_0"]["precision"]),
        "recall_similar":                 float(report["similar_0"]["recall"]),
        "f1_similar":                     float(report["similar_0"]["f1-score"]),

        "precision_dissimilar":           float(report["dissimilar_1"]["precision"]),
        "recall_dissimilar":              float(report["dissimilar_1"]["recall"]),
        "f1_dissimilar":                  float(report["dissimilar_1"]["f1-score"]),

        "auc_roc":                        auc,
        "average_precision":              ap,

        "eer":                            eer,
        "eer_threshold":                  eer_thresh,

        "mean_distance_similar":          mean_sim,
        "mean_distance_dissimilar":       mean_dis,
        "distance_gap":                   mean_dis - mean_sim,
    }

    if verbose:
        _print_metrics(metrics, labels, preds)

    # W&B logging
    if wandb_run is not None:
        try:
            import wandb
            wandb_run.summary.update({f"test/{k}": v for k, v in metrics.items()})
            wandb_run.log({
                "test/confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None, y_true=labels.tolist(), preds=preds.tolist(),
                    class_names=["Similar", "Dissimilar"],
                )
            })
        except Exception:
            pass

    return metrics


def _print_metrics(
    metrics: Dict[str, float],
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
) -> None:
    w = 64
    print("\n" + "=" * w)
    print("  EVALUATION RESULTS  —  Siamese-VGG")
    print("=" * w)
    print(f"  {'Accuracy':<38} {metrics['accuracy_pct']:.2f} %")
    print()
    print(f"  {'Precision (weighted)':<38} {metrics['precision_weighted']*100:.2f} %")
    print(f"  {'Recall    (weighted)':<38} {metrics['recall_weighted']*100:.2f} %")
    print(f"  {'F1-Score  (weighted)':<38} {metrics['f1_weighted']*100:.2f} %")
    print()
    print(f"  {'Precision (macro)':<38} {metrics['precision_macro']*100:.2f} %")
    print(f"  {'Recall    (macro)':<38} {metrics['recall_macro']*100:.2f} %")
    print(f"  {'F1-Score  (macro)':<38} {metrics['f1_macro']*100:.2f} %")
    print()
    print(f"  {'AUC-ROC':<38} {metrics['auc_roc']:.4f}")
    print(f"  {'Average Precision (PR-AUC)':<38} {metrics['average_precision']:.4f}")
    print(f"  {'Equal Error Rate (EER)':<38} {metrics['eer']*100:.2f} %  "
          f"(@ d={metrics['eer_threshold']:.3f})")
    print()
    print(f"  {'Mean dist — similar pairs':<38} {metrics['mean_distance_similar']:.4f}")
    print(f"  {'Mean dist — dissimilar pairs':<38} {metrics['mean_distance_dissimilar']:.4f}")
    print(f"  {'Distance gap (dissim - sim)':<38} {metrics['distance_gap']:.4f}")
    print("-" * w)

    cm = confusion_matrix(y_true, y_pred)
    print("  Confusion Matrix:")
    print(f"                   Pred Similar  Pred Dissimilar")
    print(f"  True Similar     {cm[0,0]:>12}  {cm[0,1]:>15}")
    print(f"  True Dissimilar  {cm[1,0]:>12}  {cm[1,1]:>15}")
    print("-" * w)
    print("  Per-class:")
    for cls, key in [("Similar (0)", "similar"), ("Dissimilar (1)", "dissimilar")]:
        p = metrics[f"precision_{key}"] * 100
        r = metrics[f"recall_{key}"]    * 100
        f = metrics[f"f1_{key}"]        * 100
        print(f"    {cls:<16}  P={p:.1f}%  R={r:.1f}%  F1={f:.1f}%")
    print("=" * w)


LFW_BASELINE_COMPARISON = [
    ("DLB [10]",                88.50, None),
    ("CFN+APEM [9]",            87.50, 1.57),
    ("L-CSSE+KSRC [8]",         92.02, None),
    ("SiameseFace1 [11]",       94.80, None),
    ("Weighted PCA-EFMNet [7]", 95.00, 0.71),
    ("Siamese-VGG (reference)", 95.62, 0.42),
    ("CosFace [15]",            99.73, None),
]


def print_comparison_table(our_accuracy_pct: float) -> None:
    w = 64
    print("\n" + "=" * w)
    print("  Face Recognition Rate on LFW Dataset")
    print("=" * w)
    print(f"  {'Method':<34}  {'Rate (%)'}")
    print("-" * w)
    for method, rate, std in LFW_BASELINE_COMPARISON:
        rate_str = f"{rate:.2f} ± {std:.2f}" if std else f"{rate:.2f}"
        print(f"  {method:<34}  {rate_str}")
    print("-" * w)
    print(f"  {'Siamese-VGG (this impl.)':<34}  {our_accuracy_pct:.2f}")
    print("=" * w)


def evaluate_with_multiple_thresholds(
    distances:  np.ndarray,
    labels:     np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> Dict[float, float]:
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.2, 0.9, 0.05).tolist()]
    results = {t: float(accuracy_score(labels, (distances >= t).astype(int)))
               for t in thresholds}
    best_t = max(results, key=results.get)
    print(f"\n[evaluator] Threshold sweep → best: {best_t:.2f} "
          f"(acc={results[best_t]*100:.2f}%)")
    return results
