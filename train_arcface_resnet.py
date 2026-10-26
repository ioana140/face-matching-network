import argparse
import csv
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image

from sklearn.metrics import (
    accuracy_score, confusion_matrix as cm_func,
    f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)


# Hyperparameters
ARCFACE_LR_BACKBONE = 1e-3   # SGD LR backbone
ARCFACE_LR_HEAD     = 1e-2   # SGD LR head
ARCFACE_SCALE   = 64.0    # scale factor (s)
ARCFACE_MARGIN  = 0.5     # angular margin
ARCFACE_BATCH   = 64      # batch size
ACCUM_STEPS     = 8       # gradient accumulation
ARCFACE_EPOCHS  = 100
EMBED_DIM       = 512     # embedding dimension
WEIGHT_DECAY    = 5e-4    # L2 regularization
PATIENCE        = 10
THRESHOLD       = 0.5     # cosine similarity threshold for verification


# Held-out set
HOLDOUT_IDENTITIES   = 500
HOLDOUT_PAIRS_PER_KIND = 1500
HOLDOUT_SEED         = 12345
TARGET_FARS          = (0.001, 0.01, 0.05)

# ResNet-50 settings
RESNET_FREEZE_UP_TO = 6

# Image settings
IMAGE_HEIGHT = 224
IMAGE_WIDTH  = 224

_NORM_MEAN = [0.485, 0.456, 0.406]
_NORM_STD  = [0.229, 0.224, 0.225]

# Test-time augmentation
USE_TTA = False

# MTCNN alignment
USE_ALIGNED = True

# Training augmentation -- the only variable changed vs. the other cases
# (everything else stays the same).
TRAIN_AUGMENTATION = "full"

_TRAIN_TRANSFORM_LIGHT = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=8, translate=(0.03, 0.03), scale=(0.97, 1.03)),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(_NORM_MEAN, _NORM_STD),
])

_TRAIN_TRANSFORM_FULL = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.RandomGrayscale(p=0.1),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    transforms.ToTensor(),
    transforms.Normalize(_NORM_MEAN, _NORM_STD),
])

_TRAIN_TRANSFORM = _TRAIN_TRANSFORM_FULL if TRAIN_AUGMENTATION == "full" else _TRAIN_TRANSFORM_LIGHT

_TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(_NORM_MEAN, _NORM_STD),
])


# Dataset
class CelebAIdentityDataset(Dataset):
    def __init__(self, image_paths, identity_indices, augment=True):
        self.image_paths = image_paths
        self.identity_indices = identity_indices
        self.transform = _TRAIN_TRANSFORM if augment else _TEST_TRANSFORM

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(str(self.image_paths[idx])).convert("RGB")
        img = self.transform(img)
        label = self.identity_indices[idx]
        return img, label

# ResNet-50 Encoder
class ArcFaceEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        self.features = nn.Sequential(
            resnet.conv1,      # 0
            resnet.bn1,        # 1
            resnet.relu,       # 2
            resnet.maxpool,    # 3
            resnet.layer1,     # 4
            resnet.layer2,     # 5
            resnet.layer3,     # 6
            resnet.layer4,     # 7
        )
        self.avgpool = resnet.avgpool

        for i, layer in enumerate(self.features):
            for name, param in layer.named_parameters():
                if i < RESNET_FREEZE_UP_TO:
                    param.requires_grad = False #straturi inghetate
                elif "bn" in name or "downsample.1" in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True #straturi antrenabile

        n_frozen = sum(1 for p in self.features.parameters() if not p.requires_grad)
        n_train  = sum(1 for p in self.features.parameters() if p.requires_grad)
        print(f"[arcface_encoder] Frozen: {n_frozen}  Trainable: {n_train}")
        print(f"[arcface_encoder] Stages 1-3 frozen, Stages 4-5 (layer3+layer4) fine-tuned")
        print(f"[arcface_encoder] Backbone BN frozen (eval mode)")

        self.fc = nn.Linear(2048, EMBED_DIM)
        self.bn = nn.BatchNorm1d(EMBED_DIM)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            for module in self.features.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    module.eval()
        return self

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.bn(x)
        x = F.normalize(x, p=2, dim=1)
        return x

# ArcFace Head
class ArcFaceHead(nn.Module):
    def __init__(self, embed_dim, num_classes, scale=ARCFACE_SCALE, margin=ARCFACE_MARGIN):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.num_classes = num_classes

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings, labels):
        W = F.normalize(self.weight, p=2, dim=1)

        cosine = F.linear(embeddings, W)  # (B, num_classes)
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        sine = torch.sqrt(1.0 - cosine.pow(2))

        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # Numerical stability when cos(theta) < cos(pi - m)
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Target class gets cos(theta + m), the rest keep cos(theta)
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        logits = logits * self.scale

        return logits

# Full ArcFace Model
class ArcFaceModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = ArcFaceEncoder()
        self.arcface_head = ArcFaceHead(EMBED_DIM, num_classes)

    def forward(self, images, labels=None):
        embeddings = self.encoder(images)
        if labels is not None:
            logits = self.arcface_head(embeddings, labels)
            return logits, embeddings
        return embeddings

    def get_embedding(self, images):
        return self.encoder(images)



def load_celeba_identity_data(
    celeba_root,
    identity_file,
    train_ratio=0.8,
    holdout_identities=HOLDOUT_IDENTITIES,
    holdout_seed=HOLDOUT_SEED,
):
    import random
    rng = random.Random(42)

    img_identity = {}
    with open(identity_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                img_path = celeba_root / parts[0]
                if img_path.exists():
                    img_identity[img_path] = parts[1]

    id_to_images = defaultdict(list)
    for img_path, ident in img_identity.items():
        id_to_images[ident].append(img_path)
    id_to_images = dict(id_to_images)

    holdout_rng = random.Random(holdout_seed)
    eligible_for_holdout = sorted([k for k, v in id_to_images.items() if len(v) >= 2])
    holdout_rng.shuffle(eligible_for_holdout)
    holdout_ids = set(eligible_for_holdout[:holdout_identities])

    holdout_id_to_images = {k: id_to_images[k] for k in holdout_ids}
    train_id_to_images   = {k: v for k, v in id_to_images.items() if k not in holdout_ids}

    unique_ids = sorted(train_id_to_images.keys())
    id_to_idx = {ident: idx for idx, ident in enumerate(unique_ids)}
    num_classes = len(unique_ids)

    n_total_imgs   = sum(len(v) for v in id_to_images.values())
    n_train_imgs   = sum(len(v) for v in train_id_to_images.values())
    n_holdout_imgs = sum(len(v) for v in holdout_id_to_images.values())

    print(f"[data] CelebA: {n_total_imgs} images, {len(id_to_images)} identities")
    print(f"[data] Held out for verification: "
          f"{len(holdout_id_to_images)} identities, {n_holdout_imgs} images "
          f"(NOT used in classification)")
    print(f"[data] For classification: {num_classes} identities, {n_train_imgs} images")

    train_paths, train_labels = [], []
    val_paths, val_labels = [], []

    for ident, paths in train_id_to_images.items():
        paths = list(paths)
        rng.shuffle(paths)
        idx = id_to_idx[ident]
        split = max(1, int(len(paths) * train_ratio))

        for p in paths[:split]:
            train_paths.append(p)
            train_labels.append(idx)
        for p in paths[split:]:
            val_paths.append(p)
            val_labels.append(idx)

    print(f"[data] Train: {len(train_paths)} images, Val: {len(val_paths)} images")

    return (
        train_paths, train_labels,
        val_paths, val_labels,
        num_classes, id_to_idx,
        holdout_id_to_images,
    )


def build_celeba_verification_pairs(
    holdout_id_to_images,
    n_per_kind=HOLDOUT_PAIRS_PER_KIND,
    seed=HOLDOUT_SEED,
):
    import random
    rng = random.Random(seed + 1)

    eligible = [k for k, v in holdout_id_to_images.items() if len(v) >= 2]
    if len(eligible) < 2:
        raise ValueError("Not enough held-out identities to build verification pairs")

    same_pairs = []
    while len(same_pairs) < n_per_kind:
        ident = rng.choice(eligible)
        imgs = holdout_id_to_images[ident]
        a, b = rng.sample(imgs, 2)
        same_pairs.append((a, b, 0))

    all_ids = list(holdout_id_to_images.keys())
    diff_pairs = []
    while len(diff_pairs) < n_per_kind:
        id_a, id_b = rng.sample(all_ids, 2)
        a = rng.choice(holdout_id_to_images[id_a])
        b = rng.choice(holdout_id_to_images[id_b])
        diff_pairs.append((a, b, 1))

    pairs = same_pairs + diff_pairs
    rng.shuffle(pairs)
    print(f"[data] Verification pairs: {len(same_pairs)} same + {len(diff_pairs)} diff "
          f"= {len(pairs)} total (from {len(holdout_id_to_images)} held-out identities)")
    return pairs

def _embed_with_tta(model, batch, tta=USE_TTA):
    emb = model.get_embedding(batch)
    if tta:
        emb_flip = model.get_embedding(torch.flip(batch, dims=[-1]))
        emb = F.normalize(emb + emb_flip, p=2, dim=1)
    return emb


def evaluate_lfw_cosine(model, test_pairs, device, threshold=THRESHOLD, tta=USE_TTA):
    model.eval()

    all_similarities = []
    all_labels = []

    batch_size = 64
    for i in range(0, len(test_pairs), batch_size):
        batch = test_pairs[i:i+batch_size]
        imgs_a, imgs_b, labels = [], [], []

        for path_a, path_b, label in batch:
            img_a = Image.open(str(path_a)).convert("RGB")
            img_b = Image.open(str(path_b)).convert("RGB")
            imgs_a.append(_TEST_TRANSFORM(img_a))
            imgs_b.append(_TEST_TRANSFORM(img_b))
            labels.append(label)

        batch_a = torch.stack(imgs_a).to(device)
        batch_b = torch.stack(imgs_b).to(device)

        with torch.no_grad():
            emb_a = _embed_with_tta(model, batch_a, tta=tta)
            emb_b = _embed_with_tta(model, batch_b, tta=tta)
            cos_sim = F.cosine_similarity(emb_a, emb_b, dim=1)

        all_similarities.extend(cos_sim.cpu().numpy().tolist())
        all_labels.extend(labels)

    similarities = np.array(all_similarities)
    labels = np.array(all_labels)

    preds = (similarities < threshold).astype(int)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    prec = precision_score(labels, preds, average="weighted", zero_division=0)
    rec = recall_score(labels, preds, average="weighted", zero_division=0)
    auc = roc_auc_score(labels, 1.0 - similarities)

    conf = cm_func(labels, preds)
    tp_sim = conf[0, 0]
    fn = conf[0, 1]
    fp = conf[1, 0]
    tp_dis = conf[1, 1]

    sim_mask = labels == 0
    dis_mask = labels == 1
    mean_sim = float(similarities[sim_mask].mean()) if sim_mask.any() else 0
    mean_dis = float(similarities[dis_mask].mean()) if dis_mask.any() else 0

    # EER
    fpr, tpr, _ = roc_curve(labels, 1.0 - similarities)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float(fpr[eer_idx])

    metrics = {
        "accuracy": acc,
        "accuracy_pct": acc * 100,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc_roc": auc,
        "eer": eer,
        "mean_sim_similar": mean_sim,
        "mean_sim_dissimilar": mean_dis,
        "sim_gap": mean_sim - mean_dis,
        "fp": int(fp),
        "fn": int(fn),
    }

    return metrics, similarities, labels


def threshold_sweep_cosine(similarities, labels):
    best_t, best_acc = 0.5, 0.0
    for t in np.arange(0.0, 1.0, 0.005):
        preds = (similarities < t).astype(int)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_t = t
    return best_t, best_acc


def compute_pair_similarities(model, pairs, device, batch_size=64, tta=USE_TTA):
    model.eval()
    sims, labs = [], []
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            imgs_a, imgs_b, labels = [], [], []
            for path_a, path_b, label in batch:
                img_a = Image.open(str(path_a)).convert("RGB")
                img_b = Image.open(str(path_b)).convert("RGB")
                imgs_a.append(_TEST_TRANSFORM(img_a))
                imgs_b.append(_TEST_TRANSFORM(img_b))
                labels.append(label)
            ba = torch.stack(imgs_a).to(device, non_blocking=True)
            bb = torch.stack(imgs_b).to(device, non_blocking=True)
            ea = _embed_with_tta(model, ba, tta=tta)
            eb = _embed_with_tta(model, bb, tta=tta)
            cs = F.cosine_similarity(ea, eb, dim=1)
            sims.extend(cs.cpu().numpy().tolist())
            labs.extend(labels)
    return np.asarray(sims), np.asarray(labs)


def verification_metrics(similarities, labels):
    if len(np.unique(labels)) < 2:
        return {"auc": 0.5, "eer": 0.5}
    auc = float(roc_auc_score(labels, 1.0 - similarities))
    fpr, tpr, _ = roc_curve(labels, 1.0 - similarities)
    fnr = 1 - tpr
    eer_idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float(fpr[eer_idx])
    return {"auc": auc, "eer": eer}


def threshold_at_far(similarities, labels, target_far):
    sims = np.asarray(similarities, dtype=np.float64)
    labs = np.asarray(labels)

    impostor_sims = sims[labs == 1]   # different-identity pairs
    genuine_sims  = sims[labs == 0]   # same-identity pairs

    if len(impostor_sims) == 0 or len(genuine_sims) == 0:
        return 0.5, 0.0, 0.0

    impostor_sorted = np.sort(impostor_sims)[::-1]   # descending
    n_imp = len(impostor_sorted)
    k = int(np.floor(target_far * n_imp))

    if k <= 0:
        threshold = float(impostor_sorted[0]) + 1e-9
    elif k >= n_imp:
        threshold = float(impostor_sorted[-1])
    else:
        threshold = float((impostor_sorted[k - 1] + impostor_sorted[k]) / 2.0)

    achieved_far = float(np.mean(impostor_sims >= threshold))
    achieved_tar = float(np.mean(genuine_sims  >= threshold))
    return threshold, achieved_far, achieved_tar


def apply_threshold(similarities, labels, threshold):
    sims = np.asarray(similarities, dtype=np.float64)
    labs = np.asarray(labels)
    preds = (sims < threshold).astype(int)   # 0=same, 1=diff (matches label coding)
    fn = int(np.sum((labs == 0) & (preds == 1)))   # genuine rejected
    fp = int(np.sum((labs == 1) & (preds == 0)))   # impostor accepted
    n_same = int(np.sum(labs == 0))
    n_diff = int(np.sum(labs == 1))
    tar = (n_same - fn) / n_same if n_same else 0.0
    far = fp / n_diff if n_diff else 0.0
    acc = float(np.mean(labs == preds))
    return {
        "threshold": float(threshold),
        "fp": fp, "fn": fn,
        "n_same": n_same, "n_diff": n_diff,
        "tar": tar, "far": far,
        "accuracy": acc,
    }


# Training
def train_arcface(
    model, train_loader, val_loader, epochs, device,
    checkpoint_dir, log_dir,
    verification_pairs=None,
):

    import wandb

    use_verification = verification_pairs is not None and len(verification_pairs) > 0

    # Differential LR: pretrained backbone vs. randomly-initialized head
    backbone_params = [p for p in model.encoder.features.parameters() if p.requires_grad]
    head_params = (
        list(model.encoder.fc.parameters()) +
        list(model.encoder.bn.parameters()) +
        list(model.arcface_head.parameters())
    )
    optimizer = SGD([
        {"params": backbone_params, "lr": ARCFACE_LR_BACKBONE},
        {"params": head_params,     "lr": ARCFACE_LR_HEAD},
    ], momentum=0.9, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()


    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max" if use_verification else "min",
        factor=0.5, patience=5,
    )

    best_score = -float("inf") if use_verification else float("inf")
    patience_counter = 0
    best_checkpoint = checkpoint_dir / "best_arcface_resnet.pt"

    history = {
        "train_loss": [], "train_accuracy": [],
        "val_loss": [], "val_accuracy": [],
        "ver_auc": [], "ver_eer": [],
    }

    csv_path = log_dir / "training_log_arcface_resnet.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "train_acc",
            "val_loss", "val_acc",
            "ver_auc", "ver_eer",
            "time_s",
        ])

    print(f"\n[arcface] Training  epochs={epochs}  "
          f"lr_backbone={ARCFACE_LR_BACKBONE}  lr_head={ARCFACE_LR_HEAD}  "
          f"scale={ARCFACE_SCALE}  margin={ARCFACE_MARGIN}  "
          f"batch={ARCFACE_BATCH}x{ACCUM_STEPS}={ARCFACE_BATCH*ACCUM_STEPS}  "
          f"weight_decay={WEIGHT_DECAY}  patience={PATIENCE}\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        optimizer.zero_grad()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits, _ = model(images, labels)
            loss = criterion(logits, labels) / ACCUM_STEPS
            loss.backward()

            if (batch_idx + 1) % ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * ACCUM_STEPS * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # Val
        model.eval()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                logits, _ = model(images, labels)
                loss = criterion(logits, labels)

                total_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += images.size(0)

        val_loss = total_loss / total
        val_acc = correct / total

        # Verification on the held-out set
        if use_verification:
            ver_sims, ver_labs = compute_pair_similarities(
                model, verification_pairs, device,
            )
            ver = verification_metrics(ver_sims, ver_labs)
            ver_auc, ver_eer = ver["auc"], ver["eer"]
        else:
            ver_auc, ver_eer = float("nan"), float("nan")

        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["ver_auc"].append(ver_auc)
        history["ver_eer"].append(ver_eer)

        msg = (f"Epoch {epoch:03d}/{epochs}  "
               f"train_loss={train_loss:.4f}  train_acc={train_acc*100:.2f}%  |  "
               f"val_loss={val_loss:.4f}  val_acc={val_acc*100:.2f}%")
        if use_verification:
            msg += f"  |  ver_auc={ver_auc:.4f}  ver_eer={ver_eer*100:.2f}%"
        msg += f"  ({elapsed:.1f}s)"
        print(msg)

        # CSV
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                f"{train_loss:.6f}", f"{train_acc:.6f}",
                f"{val_loss:.6f}", f"{val_acc:.6f}",
                f"{ver_auc:.6f}",  f"{ver_eer:.6f}",
                f"{elapsed:.1f}",
            ])

        # W&B
        wandb_log = {
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
        }
        if use_verification:
            wandb_log.update({
                "ver/auc": ver_auc,
                "ver/eer": ver_eer,
            })
        wandb.log(wandb_log)

        # LR scheduler
        scheduler.step(ver_auc if use_verification else val_loss)
        lr_bb = optimizer.param_groups[0]["lr"]
        lr_hd = optimizer.param_groups[1]["lr"]
        if lr_bb < ARCFACE_LR_BACKBONE or lr_hd < ARCFACE_LR_HEAD:
            print(f"  LR reduced -> backbone={lr_bb:.2e}  head={lr_hd:.2e}")

        # Best model + early stopping
        improved = (
            (use_verification and ver_auc > best_score) or
            (not use_verification and val_loss < best_score)
        )
        if improved:
            best_score = ver_auc if use_verification else val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "ver_auc": ver_auc,
                "ver_eer": ver_eer,
                "num_classes": model.arcface_head.num_classes,
                "selection_metric": "ver_auc" if use_verification else "val_loss",
            }, best_checkpoint)
            tag = (f"ver_auc={ver_auc:.4f}" if use_verification
                   else f"val_loss={val_loss:.4f}")
            print(f"  -> Saved best model  ({tag})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                metric_name = "ver_auc" if use_verification else "val_loss"
                print(f"\n[arcface] Early stopping at epoch {epoch} "
                      f"(no {metric_name} improvement for {PATIENCE} epochs)")
                break

    # Restore best
    print(f"\n[arcface] Training complete.")
    ckpt = torch.load(best_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    sel = ckpt.get("selection_metric", "val_loss")
    extra = (f"  ver_auc={ckpt.get('ver_auc', float('nan')):.4f}"
             if sel == "ver_auc" else "")
    print(f"[arcface] Best weights restored -- epoch {ckpt['epoch']} "
          f"(selected on {sel}){extra}\n"
          f"           train_loss={ckpt['train_loss']:.4f}  "
          f"train_acc={ckpt['train_acc']*100:.2f}%  "
          f"val_loss={ckpt['val_loss']:.4f}  "
          f"val_acc={ckpt['val_acc']*100:.2f}%")

    return history


# Main
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--epochs", type=int, default=ARCFACE_EPOCHS)
    p.add_argument("--batch-size", type=int, default=ARCFACE_BATCH)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()

    import wandb

    from config import WANDB_PROJECT, WANDB_RUN_NAME
    ROOT_DIR_LOCAL = Path(__file__).parent
    CELEBA_ROOT           = ROOT_DIR_LOCAL / "datasets" / "celeba" / "img_align_celeba"
    CELEBA_IDENTITY_FILE  = ROOT_DIR_LOCAL / "datasets" / "celeba" / "identity_CelebA.txt"
    LFW_ROOT              = ROOT_DIR_LOCAL / "datasets" / "lfw" / "lfw"
    PAIRS_CSV             = ROOT_DIR_LOCAL / "datasets" / "lfw" / "pairs.csv"
    CHECKPOINT_DIR        = ROOT_DIR_LOCAL / "checkpoints"
    LOG_DIR               = ROOT_DIR_LOCAL / "logs"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if USE_ALIGNED:
        CELEBA_ALIGNED_ROOT = ROOT_DIR_LOCAL / "datasets" / "celeba_aligned"
        LFW_ALIGNED_ROOT    = ROOT_DIR_LOCAL / "datasets" / "lfw_aligned"
        if not CELEBA_ALIGNED_ROOT.exists():
            raise FileNotFoundError(
                f"USE_ALIGNED=True but {CELEBA_ALIGNED_ROOT} does not exist. "
                f"Run align_dataset.py first."
            )
        if not LFW_ALIGNED_ROOT.exists():
            raise FileNotFoundError(
                f"USE_ALIGNED=True but {LFW_ALIGNED_ROOT} does not exist. "
                f"Run align_dataset.py first."
            )
        CELEBA_ROOT = CELEBA_ALIGNED_ROOT
        LFW_ROOT    = LFW_ALIGNED_ROOT
        print(f"[arcface] Cazul 17: using ALIGNED datasets")
        print(f"[arcface]   CelebA aligned: {CELEBA_ROOT}")
        print(f"[arcface]   LFW    aligned: {LFW_ROOT}")
    from data import (
        verify_lfw_root, build_image_catalogue,
        build_lfw_test_pairs,
    )
    from utils import plot_training_history

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[arcface] Device : {device}")
    if device.type == "cuda":
        print(f"[arcface] GPU    : {torch.cuda.get_device_name(0)}")


    (train_paths, train_labels,
     val_paths, val_labels,
     num_classes, id_to_idx,
     holdout_id_to_images) = load_celeba_identity_data(
        CELEBA_ROOT, CELEBA_IDENTITY_FILE,
    )

    verification_pairs = build_celeba_verification_pairs(holdout_id_to_images)

    train_ds = CelebAIdentityDataset(train_paths, train_labels, augment=True)
    val_ds = CelebAIdentityDataset(val_paths, val_labels, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        generator=torch.Generator().manual_seed(42),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # LFW test
    lfw_root = verify_lfw_root(LFW_ROOT)
    build_image_catalogue(lfw_root)
    test_pairs = build_lfw_test_pairs(
        csv_path=PAIRS_CSV, lfw_root=lfw_root, skip_missing=True,
    )

    print(f"[arcface] Train: {len(train_ds)} images")
    print(f"[arcface] Val:   {len(val_ds)} images")
    print(f"[arcface] Verification (held-out CelebA): {len(verification_pairs)} pairs")
    print(f"[arcface] Test:  {len(test_pairs)} LFW pairs")
    print(f"[arcface] Classes: {num_classes} identities (held-out IDs excluded)")

    # W&B
    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"{WANDB_RUN_NAME}-arcface-resnet50",
        config={
            "optimizer":        "SGD (momentum=0.9)",
            "lr_backbone":      ARCFACE_LR_BACKBONE,
            "lr_head":          ARCFACE_LR_HEAD,
            "weight_decay":     WEIGHT_DECAY,
            "batch_size":       f"{args.batch_size}x{ACCUM_STEPS}={args.batch_size*ACCUM_STEPS}",
            "epochs":           args.epochs,
            "arcface_scale":    ARCFACE_SCALE,
            "arcface_margin":   ARCFACE_MARGIN,
            "embed_dim":        EMBED_DIM,
            "num_classes":      num_classes,
            "backbone":         "ResNet-50",
            "frozen_stages":    "1-3 (conv1, layer1, layer2)",
            "fine_tune_stages": "4-5 (layer3, layer4)",
            "backbone_bn":      "frozen (eval mode)",
            "fc_head":          f"Linear(2048->{EMBED_DIM}) + BN + L2norm",
            "loss":             "ArcFace (angular margin)",
            "similarity":       "cosine",
            "input_size":       f"{IMAGE_HEIGHT}x{IMAGE_WIDTH}x3",
            "normalization":    f"mean={_NORM_MEAN}, std={_NORM_STD}",
            "tta":              "hflip-embed-avg" if USE_TTA else "off",
            "alignment":        "MTCNN 5-landmark (224x224, margin=20)" if USE_ALIGNED else "none",
            "augmentation":     "flip+affine+jitter (light, post-alignment)" if USE_ALIGNED
                                else "flip+affine+grayscale+jitter+blur",
            "dataset":          "CelebA (classification) + LFW 6000 (test)"
                                + (" [aligned]" if USE_ALIGNED else ""),
            "train_images":     len(train_ds),
            "val_images":       len(val_ds),
            "experiment":       "ArcFace + ResNet-50 (from scratch)",
            "device":           str(device),
        },
    )

    # Model
    model = ArcFaceModel(num_classes=num_classes)

    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        state = ckpt["model_state"] if "model_state" in ckpt else ckpt
        model.load_state_dict(state)
        print(f"[arcface] Loaded checkpoint: {args.checkpoint}")

    model.to(device)

    total = sum(p.numel() for p in model.parameters())
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"[arcface] Params: {total:,} total  {frozen:,} frozen  "
          f"{total-frozen:,} trainable")

    # Train
    if not args.eval_only:
        history = train_arcface(
            model, train_loader, val_loader,
            epochs=args.epochs, device=device,
            checkpoint_dir=CHECKPOINT_DIR, log_dir=LOG_DIR,
            verification_pairs=verification_pairs,
        )
        plot_training_history(history, save_path=LOG_DIR / "training_curves_arcface_resnet.png")
        run.log({"training_curves": wandb.Image(
            str(LOG_DIR / "training_curves_arcface_resnet.png"))})

    print(f"\n[arcface] Evaluating on LFW (cosine similarity, threshold={THRESHOLD}) ...")

    metrics, similarities, labels = evaluate_lfw_cosine(
        model, test_pairs, device, threshold=THRESHOLD,
    )

    print(f"\n================================================================")
    print(f"  EVALUATION RESULTS -- ArcFace + ResNet-50")
    print(f"================================================================")
    print(f"  Accuracy                               {metrics['accuracy_pct']:.2f} %")
    print(f"  Precision (weighted)                   {metrics['precision']*100:.2f} %")
    print(f"  Recall    (weighted)                   {metrics['recall']*100:.2f} %")
    print(f"  F1-Score  (weighted)                   {metrics['f1']*100:.2f} %")
    print(f"  AUC-ROC                                {metrics['auc_roc']:.4f}")
    print(f"  EER                                    {metrics['eer']*100:.2f} %")
    print(f"  Mean cosine sim -- similar pairs       {metrics['mean_sim_similar']:.4f}")
    print(f"  Mean cosine sim -- dissimilar pairs    {metrics['mean_sim_dissimilar']:.4f}")
    print(f"  Similarity gap (sim - dissim)          {metrics['sim_gap']:.4f}")
    print(f"----------------------------------------------------------------")
    print(f"  FP (dissimilar predicted similar):     {metrics['fp']}")
    print(f"  FN (similar predicted dissimilar):     {metrics['fn']}")
    print(f"================================================================")

    print(f"\n[arcface] Calibrating thresholds on held-out CelebA pairs ...")
    cal_sims, cal_labs = compute_pair_similarities(model, verification_pairs, device)
    cal_ver = verification_metrics(cal_sims, cal_labs)
    print(f"[arcface] Calibration set:  AUC={cal_ver['auc']:.4f}  "
          f"EER={cal_ver['eer']*100:.2f}%  "
          f"({int((cal_labs==0).sum())} same / {int((cal_labs==1).sum())} diff)")

    print(f"\n================================================================")
    print(f"  TABLE II -- FAR-targeted operating points")
    print(f"  Thresholds calibrated on held-out CelebA, applied to LFW")
    print(f"================================================================")
    print(f"  {'Target FAR':>10} {'Thresh':>7} {'Calib FAR':>9} {'Calib TAR':>9}"
          f"   |  {'LFW Acc':>7} {'LFW TAR':>7} {'LFW FAR':>7} {'LFW FP':>6} {'LFW FN':>6}")
    print(f"  ----------------------------------------------------------------"
          f"------------------")

    far_table = {}
    for target_far in TARGET_FARS:
        thr, cal_far, cal_tar = threshold_at_far(cal_sims, cal_labs, target_far)
        lfw = apply_threshold(similarities, labels, thr)
        far_table[target_far] = {
            "threshold": thr,
            "calib_far": cal_far, "calib_tar": cal_tar,
            "lfw_acc": lfw["accuracy"], "lfw_tar": lfw["tar"],
            "lfw_far": lfw["far"], "lfw_fp": lfw["fp"], "lfw_fn": lfw["fn"],
        }
        print(f"  {target_far*100:>9.2f}% {thr:>7.4f} "
              f"{cal_far*100:>8.2f}% {cal_tar*100:>8.2f}%   |  "
              f"{lfw['accuracy']*100:>6.2f}% "
              f"{lfw['tar']*100:>6.2f}% {lfw['far']*100:>6.2f}% "
              f"{lfw['fp']:>6} {lfw['fn']:>6}")
    print(f"================================================================")
    print(f"  Read this table as: 'at FAR={TARGET_FARS[0]*100:.1f}% (calibrated),")
    print(f"  the model accepts LFW_TAR% of legitimate users on LFW.'")
    print(f"  Pick the row that matches your security policy.")
    print(f"================================================================")

    best_t, best_acc = threshold_sweep_cosine(similarities, labels)
    print(f"\n[arcface] (reference) LFW oracle threshold sweep "
          f"-> {best_t:.3f} acc={best_acc*100:.2f}% "
          f"-- tuned ON the test set, do not cite as the operating point")

    # W&B log
    run.summary.update({f"test/{k}": v for k, v in metrics.items()})
    run.summary.update({
        "calib/auc": cal_ver["auc"],
        "calib/eer": cal_ver["eer"],
        "test/oracle_threshold": best_t,
        "test/oracle_accuracy": best_acc,
    })
    for target_far, row in far_table.items():
        tag = f"far_{int(target_far*1000):04d}"  # e.g. far_0010 = FAR 1.0%
        for k, v in row.items():
            run.summary[f"{tag}/{k}"] = v

    # Plots
    from utils import plot_confusion_matrix
    preds = (similarities < THRESHOLD).astype(int)
    plot_confusion_matrix(labels, preds,
                          save_path=LOG_DIR / "confusion_matrix_arcface_resnet.png")
    run.log({"test/confusion_matrix": wandb.Image(
        str(LOG_DIR / "confusion_matrix_arcface_resnet.png"))})

    # Table
    print(f"\n[arcface] Result: {metrics['accuracy_pct']:.2f}% "
          f"(fixed threshold={THRESHOLD})")

    print(f"\n================================================================")
    print(f"  TABLE I -- Face Recognition Rate on LFW Dataset")
    print(f"================================================================")
    print(f"  Method                              Rate (%)")
    print(f"----------------------------------------------------------------")
    print(f"  Siamese-VGG (paper)                 95.62 +/- 0.42")
    print(f"  Siamese-VGG + hard mining (ours)    89.18")
    print(f"  Siamese-ResNet50 (ours)             85.70")
    print(f"  ArcFace-ResNet50 (ours)             {metrics['accuracy_pct']:.2f}")
    print(f"================================================================")

    run.finish()
    print(f"\n[arcface] Done.")
    print(f"  Logs       -> {LOG_DIR}")
    print(f"  Checkpoint -> {CHECKPOINT_DIR / 'best_arcface_resnet.pt'}")


if __name__ == "__main__":
    main()
