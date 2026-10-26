import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import VGG16_Weights, ResNet50_Weights
from PIL import Image

from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)


# Default
EMBED_DIM            = 512
WEIGHT_DECAY_DEFAULT = 5e-4
PATIENCE             = 10
ACCUM_STEPS_DEFAULT  = 1

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

HOLDOUT_IDENTITIES     = 500
HOLDOUT_PAIRS_PER_KIND = 1500
HOLDOUT_SEED           = 12345
TARGET_FARS            = (0.001, 0.01, 0.05)



def make_transforms(image_size, augmentation):
    test = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])

    if augmentation == 'none':
        train = test
    elif augmentation == 'full':
        train = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05),
                                    scale=(0.95, 1.05)),
            transforms.RandomGrayscale(p=0.1),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.1, hue=0.05),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])
    elif augmentation == 'light':
        train = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=8, translate=(0.03, 0.03),
                                    scale=(0.97, 1.03)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15,
                                   saturation=0.05, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])
    else:
        raise ValueError(f"Unknown augmentation: {augmentation}")
    return train, test


class VGG16SiameseEncoder(nn.Module):
    _FREEZE_BOUND = {'layer5': 24, 'layer4_5': 17}

    def __init__(self, finetune='layer5', bn_head=False,
                 image_size=128, embed_dim=EMBED_DIM):
        super().__init__()
        if finetune not in self._FREEZE_BOUND:
            raise ValueError(f"finetune must be one of {list(self._FREEZE_BOUND)}")
        self.finetune = finetune

        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg.features

        bound = self._FREEZE_BOUND[finetune]
        for i, layer in enumerate(self.features):
            for p in layer.parameters():
                p.requires_grad = (i >= bound)

        spatial = image_size // 32
        in_dim  = 512 * spatial * spatial

        head_layers = []
        for _ in range(3):
            head_layers.append(nn.Linear(in_dim, embed_dim))
            if bn_head:
                head_layers.append(nn.BatchNorm1d(embed_dim))
            head_layers.append(nn.ReLU(inplace=True))
            in_dim = embed_dim
        self.fc_head = nn.Sequential(*head_layers)

        n_frozen = sum(1 for p in self.features.parameters() if not p.requires_grad)
        n_train  = sum(1 for p in self.features.parameters() if p.requires_grad)
        print(f"[vgg16] finetune={finetune}  bn_head={bn_head}  "
              f"frozen_params={n_frozen}  trainable_params={n_train}  "
              f"flatten_dim={512*spatial*spatial}")

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc_head(x)
        return x


class ResNet50SiameseEncoder(nn.Module):

    _FREEZE_INDEX = {'layer5': 7, 'layer4_5': 6}

    def __init__(self, finetune='layer5', bn_head=False,
                 image_size=128, embed_dim=EMBED_DIM):
        super().__init__()
        if finetune not in self._FREEZE_INDEX:
            raise ValueError(f"finetune must be one of {list(self._FREEZE_INDEX)}")
        self.finetune = finetune

        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        self.avgpool = resnet.avgpool

        bound = self._FREEZE_INDEX[finetune]
        for i, layer in enumerate(self.features):
            for name, p in layer.named_parameters():
                if i < bound:
                    p.requires_grad = False
                elif "bn" in name or "downsample.1" in name:
                    p.requires_grad = False  # backbone BN frozen
                else:
                    p.requires_grad = True

        in_dim = 2048
        head_layers = []
        head_layers.append(nn.Linear(in_dim, embed_dim))
        if bn_head:
            head_layers.append(nn.BatchNorm1d(embed_dim))
        head_layers.append(nn.ReLU(inplace=True))
        self.fc_head = nn.Sequential(*head_layers)

        n_frozen = sum(1 for p in self.features.parameters() if not p.requires_grad)
        n_train  = sum(1 for p in self.features.parameters() if p.requires_grad)
        print(f"[resnet50] finetune={finetune}  bn_head={bn_head}  "
              f"frozen_params={n_frozen}  trainable_params={n_train}")

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
        x = self.fc_head(x)
        return x


def build_encoder(backbone, finetune, bn_head, image_size):
    if backbone == 'vgg16':
        return VGG16SiameseEncoder(finetune=finetune, bn_head=bn_head,
                                   image_size=image_size)
    elif backbone == 'resnet50':
        return ResNet50SiameseEncoder(finetune=finetune, bn_head=bn_head,
                                      image_size=image_size)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")


# Contrastive loss
def contrastive_loss(emb_a, emb_b, labels, margin=1.0):

    d = (emb_a - emb_b).pow(2).sum(dim=1).clamp(min=1e-12).sqrt()
    y = labels.float()
    loss_same = (1.0 - y) * d.pow(2)
    loss_diff = y * F.relu(margin - d).pow(2)
    return (loss_same + loss_diff).mean()



# Datasets
class CelebaPairDataset(Dataset):

    def __init__(self, pairs, transform):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path_a, path_b, label = self.pairs[idx]
        img_a = Image.open(str(path_a)).convert("RGB")
        img_b = Image.open(str(path_b)).convert("RGB")
        return self.transform(img_a), self.transform(img_b), int(label)



def load_celeba_with_holdout(
    celeba_root,
    identity_file,
    holdout_identities=HOLDOUT_IDENTITIES,
    holdout_seed=HOLDOUT_SEED,
):

    import random
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
    eligible = sorted([k for k, v in id_to_images.items() if len(v) >= 2])
    holdout_rng.shuffle(eligible)
    holdout_ids = set(eligible[:holdout_identities])

    holdout = {k: id_to_images[k] for k in holdout_ids}
    train   = {k: v for k, v in id_to_images.items() if k not in holdout_ids}

    print(f"[data] CelebA: {sum(len(v) for v in id_to_images.values())} images, "
          f"{len(id_to_images)} identities")
    print(f"[data] Held out for verification: {len(holdout)} ids, "
          f"{sum(len(v) for v in holdout.values())} images")
    print(f"[data] For training:               {len(train)} ids, "
          f"{sum(len(v) for v in train.values())} images")
    return train, holdout


def build_train_val_pairs(
    train_id_to_images,
    n_train_pairs=200_000,
    n_val_pairs=20_000,
    train_ratio=0.8,
    seed=42,
):
    import random
    rng = random.Random(seed)

    train_imgs_per_id = {}
    val_imgs_per_id   = {}
    for ident, paths in train_id_to_images.items():
        paths = list(paths)
        rng.shuffle(paths)
        split = max(1, int(len(paths) * train_ratio))
        if len(paths) > split:
            train_imgs_per_id[ident] = paths[:split]
            val_imgs_per_id[ident]   = paths[split:]
        else:
            train_imgs_per_id[ident] = paths
            val_imgs_per_id[ident]   = paths

    def gen_pairs(per_id, n_pairs, rng):
        pairs = []
        all_ids = list(per_id.keys())
        eligible = [i for i in all_ids if len(per_id[i]) >= 2]
        n_same = n_pairs // 2
        n_diff = n_pairs - n_same
        while len(pairs) < n_same:
            ident = rng.choice(eligible)
            a, b = rng.sample(per_id[ident], 2)
            pairs.append((a, b, 0))
        while len(pairs) < n_pairs:
            id_a, id_b = rng.sample(all_ids, 2)
            a = rng.choice(per_id[id_a])
            b = rng.choice(per_id[id_b])
            pairs.append((a, b, 1))
        rng.shuffle(pairs)
        return pairs

    train_pairs = gen_pairs(train_imgs_per_id, n_train_pairs, rng)
    val_pairs   = gen_pairs(val_imgs_per_id, n_val_pairs, rng)
    print(f"[data] Train pairs: {len(train_pairs)}  Val pairs: {len(val_pairs)}")
    return train_pairs, val_pairs


def build_celeba_verification_pairs(
    holdout_id_to_images,
    n_per_kind=HOLDOUT_PAIRS_PER_KIND,
    seed=HOLDOUT_SEED,
):
    import random
    rng = random.Random(seed + 1)
    eligible = [k for k, v in holdout_id_to_images.items() if len(v) >= 2]
    same_pairs = []
    while len(same_pairs) < n_per_kind:
        ident = rng.choice(eligible)
        a, b = rng.sample(holdout_id_to_images[ident], 2)
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
    print(f"[data] Verification pairs: {len(same_pairs)} same + "
          f"{len(diff_pairs)} diff = {len(pairs)} total")
    return pairs



def compute_pair_distances(model, pairs, device, transform, batch_size=64):
    model.eval()
    dists, labs = [], []
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            imgs_a, imgs_b, labels = [], [], []
            for path_a, path_b, label in batch:
                img_a = Image.open(str(path_a)).convert("RGB")
                img_b = Image.open(str(path_b)).convert("RGB")
                imgs_a.append(transform(img_a))
                imgs_b.append(transform(img_b))
                labels.append(label)
            ba = torch.stack(imgs_a).to(device, non_blocking=True)
            bb = torch.stack(imgs_b).to(device, non_blocking=True)
            ea = model(ba)
            eb = model(bb)
            d = (ea - eb).pow(2).sum(dim=1).clamp(min=1e-12).sqrt()
            dists.extend(d.cpu().numpy().tolist())
            labs.extend(labels)
    return np.asarray(dists), np.asarray(labs)


def verification_metrics_dist(distances, labels):
    if len(np.unique(labels)) < 2:
        return {"auc": 0.5, "eer": 0.5}
    auc = float(roc_auc_score(labels, distances))
    fpr, tpr, _ = roc_curve(labels, distances)
    fnr = 1 - tpr
    eer_idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float(fpr[eer_idx])
    return {"auc": auc, "eer": eer}


def threshold_at_far_dist(distances, labels, target_far):
    dists = np.asarray(distances, dtype=np.float64)
    labs  = np.asarray(labels)
    impostor = dists[labs == 1]
    genuine  = dists[labs == 0]
    if len(impostor) == 0 or len(genuine) == 0:
        return 0.5, 0.0, 0.0
    impostor_sorted = np.sort(impostor)  # ascending
    n_imp = len(impostor_sorted)
    k = int(np.floor(target_far * n_imp))
    if k <= 0:
        threshold = float(impostor_sorted[0]) - 1e-9
    elif k >= n_imp:
        threshold = float(impostor_sorted[-1])
    else:
        threshold = float((impostor_sorted[k - 1] + impostor_sorted[k]) / 2.0)
    achieved_far = float(np.mean(impostor < threshold))
    achieved_tar = float(np.mean(genuine  < threshold))
    return threshold, achieved_far, achieved_tar


def apply_threshold_dist(distances, labels, threshold):
    dists = np.asarray(distances, dtype=np.float64)
    labs  = np.asarray(labels)
    preds = (dists >= threshold).astype(int)  # >= threshold -> diff
    fn = int(np.sum((labs == 0) & (preds == 1)))
    fp = int(np.sum((labs == 1) & (preds == 0)))
    n_same = int(np.sum(labs == 0))
    n_diff = int(np.sum(labs == 1))
    tar = (n_same - fn) / n_same if n_same else 0.0
    far = fp / n_diff if n_diff else 0.0
    acc = float(np.mean(labs == preds))
    prec = float(precision_score(labs, preds, average="weighted", zero_division=0))
    rec  = float(recall_score   (labs, preds, average="weighted", zero_division=0))
    f1v  = float(f1_score       (labs, preds, average="weighted", zero_division=0))
    return {
        "threshold": float(threshold),
        "fp": fp, "fn": fn,
        "n_same": n_same, "n_diff": n_diff,
        "tar": tar, "far": far,
        "accuracy": acc,
        "precision": prec, "recall": rec, "f1": f1v,
    }


def compute_pair_metrics(distances, labels, threshold):
    th_metrics = apply_threshold_dist(distances, labels, threshold)
    auc_metrics = verification_metrics_dist(distances, labels)
    return {
        **th_metrics,
        "auc": auc_metrics["auc"],
        "eer": auc_metrics["eer"],
    }


def mine_hard_negatives(
    model, train_id_to_images, n_pairs, device, transform,
    embed_sample_per_id=10, hard_fraction=0.3, batch_size=64,
):
    model.eval()
    sampled_ids = []
    sampled_paths = []
    for ident, paths in train_id_to_images.items():
        if len(paths) < 2:
            continue
        chosen = paths[:embed_sample_per_id]
        for p in chosen:
            sampled_ids.append(ident)
            sampled_paths.append(p)

    embs = []
    with torch.no_grad():
        for i in range(0, len(sampled_paths), batch_size):
            batch = sampled_paths[i:i+batch_size]
            imgs = [transform(Image.open(str(p)).convert("RGB")) for p in batch]
            x = torch.stack(imgs).to(device, non_blocking=True)
            embs.append(model(x).cpu())
    embs = torch.cat(embs, dim=0).numpy()
    sampled_ids = np.asarray(sampled_ids)
    sampled_paths = np.asarray([str(p) for p in sampled_paths])

    n_hard = int(n_pairs * 0.5 * hard_fraction)
    n_easy_neg = (n_pairs // 2) - n_hard
    n_pos = n_pairs - (n_hard + n_easy_neg)

    print(f"[mining] anchors={len(sampled_paths)} ids_sampled={len(set(sampled_ids))}")
    print(f"[mining] target: n_hard_neg={n_hard} n_easy_neg={n_easy_neg} n_pos={n_pos}")

    import random
    rng = random.Random(HOLDOUT_SEED + 2)

    pairs = []

    chunk = 256
    for i in range(0, len(embs), chunk):
        anchors = embs[i:i+chunk]
        anchor_ids = sampled_ids[i:i+chunk]
        a_sq = np.sum(anchors ** 2, axis=1)          # (chunk,)
        b_sq = np.sum(embs    ** 2, axis=1)          # (N,)
        dot  = anchors @ embs.T                      # (chunk, N)
        dists_sq = a_sq[:, None] + b_sq[None, :] - 2.0 * dot
        dists = np.sqrt(np.maximum(dists_sq, 0.0))
        for j, aid in enumerate(anchor_ids):
            mask_diff = (sampled_ids != aid)
            cand_idx = np.where(mask_diff)[0]
            if len(cand_idx) == 0:
                continue
            d_diff = dists[j, cand_idx]
            order = np.argsort(d_diff)[:5]
            for k in order:
                neg_idx = cand_idx[k]
                pairs.append((Path(sampled_paths[i+j]),
                              Path(sampled_paths[neg_idx]), 1))
                if len([p for p in pairs if p[2] == 1]) >= n_hard:
                    break
            if len([p for p in pairs if p[2] == 1]) >= n_hard:
                break
        if len([p for p in pairs if p[2] == 1]) >= n_hard:
            break

    all_ids = list(train_id_to_images.keys())
    while len([p for p in pairs if p[2] == 1]) < (n_hard + n_easy_neg):
        id_a, id_b = rng.sample(all_ids, 2)
        a = rng.choice(train_id_to_images[id_a])
        b = rng.choice(train_id_to_images[id_b])
        pairs.append((a, b, 1))

    eligible = [k for k, v in train_id_to_images.items() if len(v) >= 2]
    while len([p for p in pairs if p[2] == 0]) < n_pos:
        ident = rng.choice(eligible)
        a, b = rng.sample(train_id_to_images[ident], 2)
        pairs.append((a, b, 0))

    rng.shuffle(pairs)
    n_actual_hard = sum(1 for p in pairs if p[2] == 1)
    n_actual_pos = sum(1 for p in pairs if p[2] == 0)
    print(f"[mining] mined {n_actual_hard} negatives + {n_actual_pos} positives "
          f"= {len(pairs)} total")
    return pairs



def train_siamese(
    model, train_loader, val_loader, verification_pairs,
    optimizer, scheduler, criterion_kwargs,
    epochs, accum_steps, device,
    test_transform, checkpoint_dir, log_dir, run_name,
    remine_every=None, train_id_to_images=None, n_train_pairs=200_000, train_transform=None,
):
    import wandb
    margin = criterion_kwargs.get('margin', 1.0)

    best_auc = -float("inf")
    patience_counter = 0
    best_checkpoint = checkpoint_dir / f"best_{run_name}.pt"

    history = {
        "train_loss": [], "val_loss": [],
        "ver_auc": [], "ver_eer": [],
    }

    csv_path = log_dir / f"training_log_{run_name}.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch",
            "train_loss", "train_acc", "train_f1", "train_prec", "train_rec", "train_auc",
            "val_loss",   "val_acc",   "val_f1",   "val_prec",   "val_rec",   "val_auc",
            "ver_auc", "ver_eer", "ver_acc",
            "time_s",
        ])

    pair_threshold = margin / 2.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        model.train()
        total_loss, n_samples = 0.0, 0
        train_dists, train_labs = [], []
        optimizer.zero_grad()
        for batch_idx, (img_a, img_b, labels) in enumerate(train_loader):
            img_a = img_a.to(device, non_blocking=True)
            img_b = img_b.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            emb_a = model(img_a)
            emb_b = model(img_b)
            d = (emb_a - emb_b).pow(2).sum(dim=1).clamp(min=1e-12).sqrt()
            loss = contrastive_loss(emb_a, emb_b, labels, margin=margin) / accum_steps
            loss.backward()
            if (batch_idx + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            total_loss += loss.item() * accum_steps * img_a.size(0)
            n_samples += img_a.size(0)
            train_dists.append(d.detach().cpu().numpy())
            train_labs.append(labels.detach().cpu().numpy())
        train_loss = total_loss / max(n_samples, 1)
        train_dists = np.concatenate(train_dists)
        train_labs  = np.concatenate(train_labs)
        train_m = compute_pair_metrics(train_dists, train_labs, pair_threshold)

        model.eval()
        total_loss, n_samples = 0.0, 0
        val_dists, val_labs = [], []
        with torch.no_grad():
            for img_a, img_b, labels in val_loader:
                img_a = img_a.to(device, non_blocking=True)
                img_b = img_b.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                emb_a = model(img_a)
                emb_b = model(img_b)
                d = (emb_a - emb_b).pow(2).sum(dim=1).clamp(min=1e-12).sqrt()
                loss = contrastive_loss(emb_a, emb_b, labels, margin=margin)
                total_loss += loss.item() * img_a.size(0)
                n_samples += img_a.size(0)
                val_dists.append(d.cpu().numpy())
                val_labs.append(labels.cpu().numpy())
        val_loss = total_loss / max(n_samples, 1)
        val_dists = np.concatenate(val_dists)
        val_labs  = np.concatenate(val_labs)
        val_m = compute_pair_metrics(val_dists, val_labs, pair_threshold)

        ver_dists, ver_labs = compute_pair_distances(
            model, verification_pairs, device, test_transform,
        )
        ver_m = compute_pair_metrics(ver_dists, ver_labs, pair_threshold)
        ver_auc, ver_eer = ver_m["auc"], ver_m["eer"]
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["ver_auc"].append(ver_auc)
        history["ver_eer"].append(ver_eer)

        print(f"Epoch {epoch:03d}/{epochs}  "
              f"train: loss={train_loss:.4f} acc={train_m['accuracy']*100:.2f}% "
              f"f1={train_m['f1']*100:.2f}%  |  "
              f"val: loss={val_loss:.4f} acc={val_m['accuracy']*100:.2f}% "
              f"f1={val_m['f1']*100:.2f}% auc={val_m['auc']:.4f}  |  "
              f"ver_auc={ver_auc:.4f} ver_eer={ver_eer*100:.2f}%  ({elapsed:.1f}s)")

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                f"{train_loss:.6f}", f"{train_m['accuracy']:.6f}",
                f"{train_m['f1']:.6f}", f"{train_m['precision']:.6f}",
                f"{train_m['recall']:.6f}", f"{train_m['auc']:.6f}",
                f"{val_loss:.6f}", f"{val_m['accuracy']:.6f}",
                f"{val_m['f1']:.6f}", f"{val_m['precision']:.6f}",
                f"{val_m['recall']:.6f}", f"{val_m['auc']:.6f}",
                f"{ver_auc:.6f}", f"{ver_eer:.6f}", f"{ver_m['accuracy']:.6f}",
                f"{elapsed:.1f}",
            ])

        wandb.log({
            "epoch": epoch,
            "train/loss":      train_loss,
            "train/accuracy":  train_m["accuracy"],
            "train/precision": train_m["precision"],
            "train/recall":    train_m["recall"],
            "train/f1":        train_m["f1"],
            "train/auc":       train_m["auc"],
            "val/loss":        val_loss,
            "val/accuracy":    val_m["accuracy"],
            "val/precision":   val_m["precision"],
            "val/recall":      val_m["recall"],
            "val/f1":          val_m["f1"],
            "val/auc":         val_m["auc"],
            "val/eer":         val_m["eer"],
            "ver/auc":         ver_auc,
            "ver/eer":         ver_eer,
            "ver/accuracy":    ver_m["accuracy"],
            "ver/f1":          ver_m["f1"],
        })

        if scheduler is not None:
            scheduler.step(ver_auc)
            lrs = [pg['lr'] for pg in optimizer.param_groups]
            print(f"  LR groups: {[f'{lr:.2e}' for lr in lrs]}")

        if (remine_every is not None and train_id_to_images is not None
                and train_transform is not None
                and epoch % remine_every == 0):
            print(f"\n[mining] Re-mining hard negatives after epoch {epoch}...")
            new_pairs = mine_hard_negatives(
                model, train_id_to_images,
                n_pairs=n_train_pairs,
                device=device, transform=test_transform,
            )
            new_ds = CelebaPairDataset(new_pairs, train_transform)
            train_loader = DataLoader(
                new_ds,
                batch_size=train_loader.batch_size,
                shuffle=True,
                num_workers=train_loader.num_workers,
                pin_memory=True,
                generator=torch.Generator().manual_seed(42 + epoch),
            )
            print(f"[mining] DataLoader rebuilt with {len(new_ds)} pairs.")

        # Best model
        if ver_auc > best_auc:
            best_auc = ver_auc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "ver_auc": ver_auc,
                "ver_eer": ver_eer,
                "selection_metric": "ver_auc",
            }, best_checkpoint)
            print(f"  -> Saved best model (ver_auc={ver_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n[siamese] Early stopping at epoch {epoch} "
                      f"(no ver_auc improvement for {PATIENCE} epochs)")
                break

    print(f"\n[siamese] Training complete. Best ver_auc={best_auc:.4f}")
    ckpt = torch.load(best_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    print(f"[siamese] Restored epoch {ckpt['epoch']}  ver_auc={ckpt['ver_auc']:.4f}")
    return history, best_checkpoint



# CLI
def parse_args():
    p = argparse.ArgumentParser()
    # Architecture
    p.add_argument("--backbone",      choices=["vgg16", "resnet50"], required=True)
    p.add_argument("--finetune",      choices=["layer5", "layer4_5"], default="layer5")
    p.add_argument("--bn-head",       action="store_true",
                   help="Insert BatchNorm1d after each FC layer (VGG only).")
    p.add_argument("--image-size",    type=int, default=128)
    # Optimization
    p.add_argument("--optimizer",     choices=["adam", "sgd"], default="adam")
    p.add_argument("--lr-backbone",   type=float, default=5e-6)
    p.add_argument("--lr-head",       type=float, default=5e-6)
    p.add_argument("--weight-decay",  type=float, default=0.0,
                   help="0 for paper, 5e-4 for ArcFace-style.")
    p.add_argument("--batch-size",    type=int, default=32)
    p.add_argument("--accum-steps",   type=int, default=ACCUM_STEPS_DEFAULT)
    p.add_argument("--lr-schedule",   choices=["none", "plateau"], default="none")
    p.add_argument("--margin",        type=float, default=1.0)
    # Augmentation
    p.add_argument("--augmentation",  choices=["none", "full", "light"], default="none")
    # Hard mining
    p.add_argument("--hard-mining",   choices=["none", "negatives"], default="none")
    p.add_argument("--remine-every",  type=int, default=5,
                   help="Re-mine hard negatives every N epochs (only when --hard-mining negatives).")
    p.add_argument("--from-checkpoint", type=str, default=None,
                   help="Load weights from this checkpoint (for hard-mining stages).")
    # Data
    p.add_argument("--n-train-pairs", type=int, default=200_000)
    p.add_argument("--n-val-pairs",   type=int, default=20_000)
    p.add_argument("--aligned",       action="store_true",
                   help="Use aligned datasets (CELEBA_ALIGNED_ROOT, LFW_ALIGNED_ROOT).")
    # Training
    p.add_argument("--epochs",        type=int, default=100)
    p.add_argument("--num-workers",   type=int, default=4)
    p.add_argument("--run-name",      type=str, required=True,
                   help="Used in checkpoint filename, log filename, W&B run name.")
    p.add_argument("--eval-only",     action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    import wandb

    from config import (
        CELEBA_ROOT, CELEBA_IDENTITY_FILE,
        CHECKPOINT_DIR, LOG_DIR,
        LFW_ROOT, PAIRS_CSV,
        WANDB_PROJECT, WANDB_RUN_NAME,
    )
    if args.aligned:
        from config import CELEBA_ALIGNED_ROOT, LFW_ALIGNED_ROOT
        if not CELEBA_ALIGNED_ROOT.exists() or not LFW_ALIGNED_ROOT.exists():
            raise FileNotFoundError("Aligned datasets not found; run align_dataset.py first.")
        CELEBA_ROOT = CELEBA_ALIGNED_ROOT
        LFW_ROOT    = LFW_ALIGNED_ROOT
        print(f"[siamese] Using ALIGNED datasets")
    from data import verify_lfw_root, build_image_catalogue, build_lfw_test_pairs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[siamese] Device : {device}")
    if device.type == "cuda":
        print(f"[siamese] GPU    : {torch.cuda.get_device_name(0)}")


    train_transform, test_transform = make_transforms(args.image_size, args.augmentation)
    print(f"[siamese] image_size={args.image_size}  augmentation={args.augmentation}")

    train_id_to_images, holdout_id_to_images = load_celeba_with_holdout(
        CELEBA_ROOT, CELEBA_IDENTITY_FILE,
    )

    if args.hard_mining == 'none':
        train_pairs, val_pairs = build_train_val_pairs(
            train_id_to_images,
            n_train_pairs=args.n_train_pairs,
            n_val_pairs=args.n_val_pairs,
        )
    else:
        _, val_pairs = build_train_val_pairs(
            train_id_to_images,
            n_train_pairs=args.n_train_pairs,
            n_val_pairs=args.n_val_pairs,
        )
        train_pairs = None

    verification_pairs = build_celeba_verification_pairs(holdout_id_to_images)

    # LFW test
    lfw_root = verify_lfw_root(LFW_ROOT)
    build_image_catalogue(lfw_root)
    test_pairs = build_lfw_test_pairs(
        csv_path=PAIRS_CSV, lfw_root=lfw_root, skip_missing=True,
    )

    # Model
    model = build_encoder(args.backbone, args.finetune, args.bn_head, args.image_size)
    if args.from_checkpoint:
        ckpt = torch.load(args.from_checkpoint, map_location=device, weights_only=True)
        state = ckpt["model_state"] if "model_state" in ckpt else ckpt
        model.load_state_dict(state, strict=False)
        print(f"[siamese] Loaded weights from {args.from_checkpoint}")
    model.to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[siamese] Params: total={n_total:,}  trainable={n_train:,}")

    # Hard mining
    if args.hard_mining == 'negatives' and not args.eval_only:
        train_pairs = mine_hard_negatives(
            model, train_id_to_images,
            n_pairs=args.n_train_pairs,
            device=device, transform=test_transform,  # use test transform for stable embeddings
        )

    # DataLoader
    if not args.eval_only:
        train_ds = CelebaPairDataset(train_pairs, train_transform)
        val_ds   = CelebaPairDataset(val_pairs, test_transform)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
            generator=torch.Generator().manual_seed(42),
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )

    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and "fc_head" not in n]
    head_params     = [p for n, p in model.named_parameters()
                       if p.requires_grad and "fc_head" in n]
    if args.optimizer == 'adam':
        optimizer = Adam([
            {"params": backbone_params, "lr": args.lr_backbone},
            {"params": head_params,     "lr": args.lr_head},
        ], weight_decay=args.weight_decay)
    else:
        optimizer = SGD([
            {"params": backbone_params, "lr": args.lr_backbone},
            {"params": head_params,     "lr": args.lr_head},
        ], momentum=0.9, weight_decay=args.weight_decay)

    scheduler = None
    if args.lr_schedule == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    # W&B
    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"{WANDB_RUN_NAME}-{args.run_name}",
        config=vars(args),
    )

    # Train
    if not args.eval_only:
        remine = args.remine_every if args.hard_mining == 'negatives' else None
        history, best_ckpt = train_siamese(
            model, train_loader, val_loader, verification_pairs,
            optimizer, scheduler,
            criterion_kwargs={"margin": args.margin},
            epochs=args.epochs, accum_steps=args.accum_steps, device=device,
            test_transform=test_transform,
            checkpoint_dir=CHECKPOINT_DIR, log_dir=LOG_DIR,
            run_name=args.run_name,
            remine_every=remine,
            train_id_to_images=train_id_to_images if remine else None,
            n_train_pairs=args.n_train_pairs,
            train_transform=train_transform if remine else None,
        )


    print(f"\n[siamese] Calibrating thresholds on held-out CelebA pairs ...")
    cal_dists, cal_labs = compute_pair_distances(
        model, verification_pairs, device, test_transform,
    )
    cal_ver = verification_metrics_dist(cal_dists, cal_labs)
    print(f"[siamese] Calibration:  AUC={cal_ver['auc']:.4f}  "
          f"EER={cal_ver['eer']*100:.2f}%")

    print(f"\n[siamese] LFW eval ...")
    lfw_dists, lfw_labs = compute_pair_distances(
        model, test_pairs, device, test_transform,
    )
    lfw_ver = verification_metrics_dist(lfw_dists, lfw_labs)

    default_t = args.margin / 2.0
    lfw_default = apply_threshold_dist(lfw_dists, lfw_labs, default_t)
    print(f"\n================================================================")
    print(f"  EVALUATION RESULTS -- Siamese-{args.backbone.upper()} (run={args.run_name})")
    print(f"================================================================")
    print(f"  LFW @ threshold={default_t:.3f} (margin/2):")
    print(f"    Accuracy              {lfw_default['accuracy']*100:>7.2f} %")
    print(f"    Precision (weighted)  {lfw_default['precision']*100:>7.2f} %")
    print(f"    Recall    (weighted)  {lfw_default['recall']*100:>7.2f} %")
    print(f"    F1-Score  (weighted)  {lfw_default['f1']*100:>7.2f} %")
    print(f"    AUC-ROC               {lfw_ver['auc']:>7.4f}")
    print(f"    EER                   {lfw_ver['eer']*100:>7.2f} %")
    print(f"    Mean dist — same      {lfw_dists[lfw_labs==0].mean():>7.4f}")
    print(f"    Mean dist — diff      {lfw_dists[lfw_labs==1].mean():>7.4f}")
    print(f"    Distance gap          {lfw_dists[lfw_labs==1].mean() - lfw_dists[lfw_labs==0].mean():>7.4f}")
    print(f"    FP = {lfw_default['fp']}     FN = {lfw_default['fn']}")
    print(f"================================================================")

    run.summary["test/accuracy"]  = lfw_default["accuracy"]
    run.summary["test/precision"] = lfw_default["precision"]
    run.summary["test/recall"]    = lfw_default["recall"]
    run.summary["test/f1"]        = lfw_default["f1"]
    run.summary["test/auc"]       = lfw_ver["auc"]
    run.summary["test/eer"]       = lfw_ver["eer"]
    run.summary["test/fp"]        = lfw_default["fp"]
    run.summary["test/fn"]        = lfw_default["fn"]


    print(f"\n================================================================")
    print(f"  TABLE II -- FAR-targeted operating points")
    print(f"  Thresholds calibrated on held-out CelebA, applied to LFW")
    print(f"================================================================")
    print(f"  {'Target FAR':>10} {'Thresh':>7} {'Calib FAR':>9} {'Calib TAR':>9}"
          f"   |  {'LFW Acc':>7} {'LFW TAR':>7} {'LFW FAR':>7} {'LFW FP':>6} {'LFW FN':>6}")
    print(f"  ----------------------------------------------------------------"
          f"------------------")
    for tf in TARGET_FARS:
        thr, cf, ct = threshold_at_far_dist(cal_dists, cal_labs, tf)
        lfw = apply_threshold_dist(lfw_dists, lfw_labs, thr)
        print(f"  {tf*100:>9.2f}% {thr:>7.4f} "
              f"{cf*100:>8.2f}% {ct*100:>8.2f}%   |  "
              f"{lfw['accuracy']*100:>6.2f}% "
              f"{lfw['tar']*100:>6.2f}% {lfw['far']*100:>6.2f}% "
              f"{lfw['fp']:>6} {lfw['fn']:>6}")
        run.summary[f"far_{int(tf*1000):04d}/threshold"] = thr
        run.summary[f"far_{int(tf*1000):04d}/lfw_acc"]   = lfw['accuracy']
        run.summary[f"far_{int(tf*1000):04d}/lfw_tar"]   = lfw['tar']
        run.summary[f"far_{int(tf*1000):04d}/lfw_far"]   = lfw['far']
        run.summary[f"far_{int(tf*1000):04d}/lfw_fp"]    = lfw['fp']
        run.summary[f"far_{int(tf*1000):04d}/lfw_fn"]    = lfw['fn']
    print(f"================================================================")

    run.summary["calib/auc"]    = cal_ver["auc"]
    run.summary["calib/eer"]    = cal_ver["eer"]
    run.summary["lfw/auc"]      = lfw_ver["auc"]
    run.summary["lfw/eer"]      = lfw_ver["eer"]
    run.summary["lfw/acc_default_t"] = lfw_default["accuracy"]


    if not args.eval_only:
        try:
            _make_training_curves_plot(
                csv_path=LOG_DIR / f"training_log_{args.run_name}.csv",
                run_name=args.run_name,
                save_path=LOG_DIR / f"training_curves_{args.run_name}.png",
            )
            run.log({"training_curves": wandb.Image(
                str(LOG_DIR / f"training_curves_{args.run_name}.png"))})
        except Exception as e:
            print(f"[plot] training_curves error: {e}")

    try:
        from utils.visualization import plot_confusion_matrix
        preds = (lfw_dists >= default_t).astype(int)
        cm_path = LOG_DIR / f"confusion_matrix_{args.run_name}.png"
        backbone_label = args.backbone.upper().replace("RESNET", "ResNet").replace("VGG", "VGG")
        plot_confusion_matrix(lfw_labs, preds, save_path=cm_path,
                              title=f"Confusion Matrix — Siamese-{backbone_label}")
        run.log({"confusion_matrix": wandb.Image(str(cm_path))})
    except Exception as e:
        print(f"[plot] confusion_matrix error: {e}")

    try:
        from utils.visualization import plot_distance_distribution
        dist_path = LOG_DIR / f"distance_distribution_{args.run_name}.png"
        backbone_label = args.backbone.upper().replace("RESNET", "ResNet").replace("VGG", "VGG")
        plot_distance_distribution(lfw_dists, lfw_labs, threshold=default_t,
                                   save_path=dist_path,
                                   title=f"Distance Distribution — Siamese-{backbone_label}")
        run.log({"distance_distribution": wandb.Image(str(dist_path))})
    except Exception as e:
        print(f"[plot] distance_distribution error: {e}")

    run.finish()
    print(f"\n[siamese] Done. Run: {args.run_name}")


def _make_training_curves_plot(csv_path, run_name, save_path):
    import csv as _csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(csv_path) as f:
        rows = list(_csv.DictReader(f))
    if not rows:
        return
    ep   = [int(r["epoch"]) for r in rows]
    trL  = [float(r["train_loss"])      for r in rows]
    trA  = [float(r["train_acc"])*100   for r in rows]
    trF  = [float(r["train_f1"])*100    for r in rows]
    trU  = [float(r["train_auc"])       for r in rows]
    vlL  = [float(r["val_loss"])        for r in rows]
    vlA  = [float(r["val_acc"])*100     for r in rows]
    vlF  = [float(r["val_f1"])*100      for r in rows]
    vlU  = [float(r["val_auc"])         for r in rows]
    veU  = [float(r["ver_auc"])         for r in rows]
    veE  = [float(r["ver_eer"])*100     for r in rows]
    veA  = [float(r["ver_acc"])*100     for r in rows]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes[0,0].plot(ep, trL, label="Train", color="steelblue")
    axes[0,0].plot(ep, vlL, label="Val",   color="tomato")
    axes[0,0].set_title("Contrastive Loss"); axes[0,0].set_xlabel("Epoch"); axes[0,0].set_ylabel("Loss")
    axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

    axes[0,1].plot(ep, trA, label="Train", color="steelblue")
    axes[0,1].plot(ep, vlA, label="Val",   color="tomato")
    axes[0,1].plot(ep, veA, label="Held-out CelebA", color="green")
    axes[0,1].set_title("Accuracy at threshold=0.5"); axes[0,1].set_xlabel("Epoch"); axes[0,1].set_ylabel("Accuracy (%)")
    axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

    axes[0,2].plot(ep, trF, label="Train", color="steelblue")
    axes[0,2].plot(ep, vlF, label="Val",   color="tomato")
    axes[0,2].set_title("F1-Score (weighted) at threshold=0.5")
    axes[0,2].set_xlabel("Epoch"); axes[0,2].set_ylabel("F1 (%)")
    axes[0,2].legend(); axes[0,2].grid(True, alpha=0.3)

    axes[1,0].plot(ep, trU, label="Train",           color="steelblue")
    axes[1,0].plot(ep, vlU, label="Val",             color="tomato")
    axes[1,0].plot(ep, veU, label="Held-out CelebA", color="green", linewidth=2)
    axes[1,0].set_title("AUC-ROC (threshold-free)"); axes[1,0].set_xlabel("Epoch"); axes[1,0].set_ylabel("AUC")
    axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

    axes[1,1].plot(ep, veE, color="green", linewidth=2)
    axes[1,1].set_title("EER on Held-out CelebA (threshold-free)")
    axes[1,1].set_xlabel("Epoch"); axes[1,1].set_ylabel("EER (%)")
    axes[1,1].grid(True, alpha=0.3)

    best_ep = max(range(len(veU)), key=lambda i: veU[i])
    axes[1,2].axis("off")
    axes[1,2].text(0.05, 0.85, f"Run: {run_name}", fontsize=12, fontweight="bold")
    axes[1,2].text(0.05, 0.70, f"Total epochs: {len(ep)}", fontsize=11)
    axes[1,2].text(0.05, 0.60, f"Best epoch: {ep[best_ep]} (selected by ver_auc)", fontsize=11)
    axes[1,2].text(0.05, 0.50, f"Best ver_auc: {veU[best_ep]:.4f}", fontsize=11)
    axes[1,2].text(0.05, 0.40, f"Best ver_eer: {veE[best_ep]:.2f}%", fontsize=11)

    plt.suptitle(f"Training curves - {run_name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualization] Saved training curves -> {save_path}")


if __name__ == "__main__":
    main()
