import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  type=str, default=None)
    p.add_argument("--eval-only",   action="store_true")
    p.add_argument("--epochs",      type=int, default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-csv",      action="store_true",
                   help="Generate all pairs from scratch (paper mode, no official CSV)")
    return p.parse_args()


def main():
    args = parse_args()

    import wandb
    import torch

    from config import (
        BATCH_SIZE, CHECKPOINT_DIR, CONTRASTIVE_MARGIN,
        DECISION_THRESHOLD, EPOCHS, LEARNING_RATE, LOG_DIR,
        LFW_ROOT, PAIRS_CSV, TARGET_PAIRS, SIMILAR_RATIO,
        WANDB_PROJECT, WANDB_RUN_NAME,
    )
    from data import (
        verify_lfw_root, build_image_catalogue,
        build_pairs, build_train_loader, build_test_loader,
    )
    from evaluation import (
        evaluate_model, evaluate_with_multiple_thresholds,
        print_comparison_table,
    )
    from evaluation.evaluator import collect_predictions
    from model import SiameseNetwork
    from training import Trainer
    from utils import (
        plot_confusion_matrix, plot_distance_distribution,
        plot_pairs, plot_training_history,
    )

    epochs = args.epochs or EPOCHS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # W&B init
    run = wandb.init(
        project = WANDB_PROJECT,
        name    = WANDB_RUN_NAME,
        config  = {
            "learning_rate":      LEARNING_RATE,
            "batch_size":         BATCH_SIZE,
            "epochs":             epochs,
            "contrastive_margin": CONTRASTIVE_MARGIN,
            "decision_threshold": DECISION_THRESHOLD,
            "dropout":            0.0,
            "l2_normalize":       False,
            "pair_mode":          "scratch" if args.no_csv else "csv+generated",
            "backbone":           "VGG-16",
            "frozen_blocks":      "1-4",
            "fine_tune_blocks":   "5",
            "fc_layers":          3,
            "fc_units":           512,
            "input_size":         "128x128x3",
            "dataset":            "LFW",
            "target_pairs":       TARGET_PAIRS,
            "train_split":        0.60,
            "test_split":         0.40,
            "paper":              "Heidari & Fouladi-Ghaleh 2021",
            "device":             str(device),
        },
    )
    print(f"[train.py] W&B run: {run.url}")

    # Verify local dataset
    lfw_root = verify_lfw_root(LFW_ROOT)
    build_image_catalogue(lfw_root)

    # Build pairs — 60% train / 40% test
    train_pairs, test_pairs = build_pairs(
        csv_path      = PAIRS_CSV,
        lfw_root      = lfw_root,
        target_pairs  = TARGET_PAIRS,
        similar_ratio = SIMILAR_RATIO,
        skip_missing  = True,
        use_csv       = not args.no_csv,
    )

    plot_pairs(train_pairs, n_pairs=8, save_path=LOG_DIR / "sample_pairs.png")
    run.log({"sample_pairs": wandb.Image(str(LOG_DIR / "sample_pairs.png"))})

    # DataLoaders
    train_loader = build_train_loader(
        train_pairs, batch_size=BATCH_SIZE, num_workers=args.num_workers
    )
    test_loader = build_test_loader(
        test_pairs, batch_size=BATCH_SIZE, num_workers=args.num_workers
    )

    # Model
    model = SiameseNetwork()

    if args.checkpoint and Path(args.checkpoint).exists():
        model = Trainer.load_checkpoint(model, Path(args.checkpoint), device)
    else:
        model.to(device)

    total  = sum(p.numel() for p in model.parameters())
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"[train.py] Params: {total:,} total  {frozen:,} frozen  "
          f"{total-frozen:,} trainable")


    # Train
    if not args.eval_only:
        trainer = Trainer(
            model          = model,
            learning_rate  = LEARNING_RATE,
            margin         = CONTRASTIVE_MARGIN,
            epochs         = epochs,
            checkpoint_dir = CHECKPOINT_DIR,
            log_dir        = LOG_DIR,
            device         = device,
        )
        history = trainer.fit(train_loader)

        plot_training_history(history, save_path=LOG_DIR / "training_curves.png")
        run.log({"training_curves": wandb.Image(str(LOG_DIR / "training_curves.png"))})

    # Final evaluation on the 40% test set
    print("\n[train.py] Final evaluation on the 40% test set …")
    metrics = evaluate_model(
        model, test_loader,
        threshold = DECISION_THRESHOLD,
        verbose   = True,
        device    = device,
    )

    run.summary.update({f"test/{k}": v for k, v in metrics.items()})
    run.log({f"test/{k}": v for k, v in metrics.items()})

    distances, labels = collect_predictions(model, test_loader, device)
    threshold_results = evaluate_with_multiple_thresholds(distances, labels)

    sweep_table = wandb.Table(
        columns=["threshold", "accuracy_pct"],
        data=[[t, acc * 100] for t, acc in sorted(threshold_results.items())],
    )
    run.log({"test/threshold_sweep": sweep_table})

    # Plots
    preds     = (distances >= DECISION_THRESHOLD).astype(int)
    dist_plot = LOG_DIR / "distance_distribution.png"
    cm_plot   = LOG_DIR / "confusion_matrix.png"

    plot_distance_distribution(distances, labels, save_path=dist_plot)
    plot_confusion_matrix(labels, preds,          save_path=cm_plot)

    run.log({
        "test/distance_distribution": wandb.Image(str(dist_plot)),
        "test/confusion_matrix":      wandb.Image(str(cm_plot)),
        "test/wandb_confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=labels.tolist(),
            preds=preds.tolist(),
            class_names=["Similar", "Dissimilar"],
        ),
    })


    print_comparison_table(metrics["accuracy_pct"])

    run.finish()

    print(f"\n[train.py] Done.")
    print(f"  Logs       → {LOG_DIR}")
    print(f"  Checkpoint → {CHECKPOINT_DIR / 'best_model.pt'}")


if __name__ == "__main__":
    main()
