

import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import (
    CHECKPOINT_DIR,
    CONTRASTIVE_MARGIN,
    DECISION_THRESHOLD,
    EPOCHS,
    LEARNING_RATE,
    LOG_DIR,
)
from training.losses import ContrastiveLoss


def _compute_metrics(
    all_distances: List[float],
    all_labels:    List[int],
) -> Dict[str, float]:
    distances = np.array(all_distances, dtype=np.float32)
    labels    = np.array(all_labels,    dtype=np.int32)
    preds     = (distances >= DECISION_THRESHOLD).astype(np.int32)

    return {
        "accuracy":  float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, average="weighted",
                                           zero_division=0)),
        "recall":    float(recall_score(labels, preds, average="weighted",
                                        zero_division=0)),
        "f1":        float(f1_score(labels, preds, average="weighted",
                                    zero_division=0)),
        "auc_roc":   float(roc_auc_score(labels, distances)),
    }


class Trainer:
    def __init__(
        self,
        model:          nn.Module,
        learning_rate:  float = LEARNING_RATE,
        margin:         float = CONTRASTIVE_MARGIN,
        epochs:         int   = EPOCHS,
        checkpoint_dir: Path  = CHECKPOINT_DIR,
        log_dir:        Path  = LOG_DIR,
        device:         Optional[torch.device] = None,
    ):
        self.model          = model
        self.epochs         = epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir        = Path(log_dir)

        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        print(f"[trainer] Device : {self.device}")
        if self.device.type == "cuda":
            print(f"[trainer] GPU    : {torch.cuda.get_device_name(0)}")

        self.model.to(self.device)

        self.optimizer = Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
        )
        self.criterion = ContrastiveLoss(margin=margin)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.history: Dict[str, List[float]] = {
            "train_loss":      [],
            "train_accuracy":  [],
            "train_precision": [],
            "train_recall":    [],
            "train_f1":        [],
            "train_auc_roc":   [],
        }

    def _run_epoch(self, loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        self.model.train(True)

        total_loss:    float       = 0.0
        n_batches:     int         = 0
        all_distances: List[float] = []
        all_labels:    List[int]   = []

        for img_a, img_b, labels in loader:
            img_a  = img_a.to(self.device,  non_blocking=True)
            img_b  = img_b.to(self.device,  non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            distances = self.model(img_a, img_b)        # (B,) raw distances
            loss      = self.criterion(distances, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss    += loss.item()
            n_batches     += 1
            all_distances.extend(distances.detach().cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().astype(int).tolist())

        mean_loss = total_loss / n_batches
        metrics   = _compute_metrics(all_distances, all_labels)
        return mean_loss, metrics

    def fit(self, train_loader: DataLoader) -> Dict[str, List[float]]:
        import wandb

        best_loss       = float("inf")
        best_checkpoint = self.checkpoint_dir / "best_model.pt"

        # CSV log
        csv_path = self.log_dir / "training_log.csv"
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch",
                "train_loss", "train_accuracy", "train_precision",
                "train_recall", "train_f1", "train_auc_roc",
                "time_s",
            ])

        print(f"\n[trainer] Training  epochs={self.epochs}  lr={LEARNING_RATE}\n")

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            train_loss, metrics = self._run_epoch(train_loader)
            elapsed = time.time() - t0

            # Store history
            self.history["train_loss"].append(train_loss)
            for key, val in metrics.items():
                self.history[f"train_{key}"].append(val)

            # Console
            print(
                f"Epoch {epoch:03d}/{self.epochs}  "
                f"loss={train_loss:.4f}  "
                f"acc={metrics['accuracy']*100:.2f}%  "
                f"prec={metrics['precision']*100:.2f}%  "
                f"rec={metrics['recall']*100:.2f}%  "
                f"f1={metrics['f1']*100:.2f}%  "
                f"auc={metrics['auc_roc']:.4f}  "
                f"({elapsed:.1f}s)"
            )

            # CSV
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    epoch,
                    f"{train_loss:.6f}",
                    f"{metrics['accuracy']:.6f}",
                    f"{metrics['precision']:.6f}",
                    f"{metrics['recall']:.6f}",
                    f"{metrics['f1']:.6f}",
                    f"{metrics['auc_roc']:.6f}",
                    f"{elapsed:.1f}",
                ])

            # W&B — always log
            wandb.log({
                "epoch":            epoch,
                "train/loss":       train_loss,
                "train/accuracy":   metrics["accuracy"],
                "train/precision":  metrics["precision"],
                "train/recall":     metrics["recall"],
                "train/f1":         metrics["f1"],
                "train/auc_roc":    metrics["auc_roc"],
            })

            # Save best model (by train loss — no validation set)
            if train_loss < best_loss:
                best_loss = train_loss
                torch.save(
                    {
                        "epoch":           epoch,
                        "model_state":     self.model.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "train_loss":      train_loss,
                        "train_acc":       metrics["accuracy"],
                    },
                    best_checkpoint,
                )
                print(f"  ✓ Saved best model  (train_loss={train_loss:.4f})")

        print(f"\n[trainer] Training complete.")
        print(f"[trainer] Best checkpoint: {best_checkpoint}")

        # Restore best weights
        ckpt = torch.load(best_checkpoint, map_location=self.device,
                          weights_only=True)
        self.model.load_state_dict(ckpt["model_state"])
        print(f"[trainer] Best weights restored — epoch {ckpt['epoch']}  "
              f"train_loss={ckpt['train_loss']:.4f}  "
              f"train_acc={ckpt['train_acc']*100:.2f}%")

        return self.history

    def save(self, path: Path) -> None:
        torch.save(self.model.state_dict(), path)
        print(f"[trainer] Saved → {path}")

    @staticmethod
    def load_checkpoint(
        model:  nn.Module,
        path:   Path,
        device: Optional[torch.device] = None,
    ) -> nn.Module:
        device = device or torch.device("cpu")
        ckpt   = torch.load(path, map_location=device, weights_only=True)
        state  = ckpt["model_state"] if "model_state" in ckpt else ckpt
        model.load_state_dict(state)
        model.to(device)
        print(f"[trainer] Loaded checkpoint from {path}")
        return model
