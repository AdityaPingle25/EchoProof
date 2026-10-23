"""EchoProof Research — Model Training Script.

Paper: "EchoProof: Robust Audio Deepfake Detection Under Unseen Voice Generators and Real-World Distortions"

Trains Wav2Vec2 with Attentive Statistical Pooling and real-world acoustic distortion
data augmentation. Tracks validation EER and AUC-ROC, and checkpoints best weights.
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, f1_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.augmentations import RealWorldDistortionPipeline
from research.dataset import DeepfakeAudioDataset
from research.model import EchoProofClassifier, FocalLoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("echoproof_train")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Computes Equal Error Rate (EER)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    # Find point where FPR == FNR
    try:
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    except Exception:
        # Fallback to absolute difference minimum
        diffs = np.abs(fpr - fnr)
        min_idx = np.argmin(diffs)
        eer = float((fpr[min_idx] + fnr[min_idx]) / 2.0)
    return float(eer)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Runs evaluation on a validation or test DataLoader."""
    model.eval()
    total_loss = 0.0
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            waveforms = batch["waveform"].to(device)
            labels = batch["label"].to(device)

            logits = model(waveforms)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)

            probs = torch.softmax(logits, dim=-1)
            spoof_scores = probs[:, 1].cpu().numpy()

            all_scores.extend(spoof_scores)
            all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_scores = np.array(all_scores)

    avg_loss = total_loss / max(1, len(y_true))
    y_pred = (y_scores >= 0.5).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        auc = float(roc_auc_score(y_true, y_scores))
    except Exception:
        auc = 0.5

    try:
        eer = compute_eer(y_true, y_scores)
    except Exception:
        eer = 0.5

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "f1": f1,
        "auc_roc": auc,
        "eer": eer,
    }


def calibrate_thresholds(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_error_rate: float = 0.05,
) -> dict[str, float | bool]:
    """Set a conservative abstention band from held-out validation scores."""
    model.eval()
    all_scores: list[float] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["waveform"].to(device))
            spoof_scores = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_scores.extend(float(value) for value in spoof_scores)
            all_labels.extend(int(value) for value in batch["label"].cpu().numpy())

    scores = np.asarray(all_scores, dtype=np.float64)
    labels = np.asarray(all_labels, dtype=np.int64)
    genuine_scores = scores[labels == 0]
    spoof_scores = scores[labels == 1]
    if genuine_scores.size == 0 or spoof_scores.size == 0:
        raise ValueError("Threshold calibration requires genuine and spoof validation examples.")

    threshold_genuine = float(np.quantile(spoof_scores, target_error_rate))
    threshold_synthetic = float(np.quantile(genuine_scores, 1.0 - target_error_rate))
    if threshold_genuine > threshold_synthetic:
        # The class distributions overlap too much for a confident abstention band.
        fpr, tpr, candidates = roc_curve(labels, scores, pos_label=1)
        finite = np.isfinite(candidates)
        eer_index = np.flatnonzero(finite)[np.argmin(np.abs(fpr[finite] - (1.0 - tpr[finite])))]
        threshold_genuine = threshold_synthetic = float(candidates[eer_index])

    return {
        "threshold_genuine": threshold_genuine,
        "threshold_synthetic": threshold_synthetic,
        "threshold_is_tuned": True,
        "target_error_rate": target_error_rate,
    }


def train(args):
    set_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Choose --device cpu or enable a CUDA GPU.")
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save run arguments
    with open(output_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    # Load dataset manifests
    df_train = pd.read_csv(args.train_csv)
    df_val = pd.read_csv(args.val_csv)

    logger.info(f"Loaded train set: {len(df_train)} samples, val set: {len(df_val)} samples")

    distortion_pipeline = RealWorldDistortionPipeline() if args.augment_prob > 0.0 else None

    train_dataset = DeepfakeAudioDataset(
        df=df_train,
        base_dir=args.audio_base_dir,
        filename_col=args.filename_col,
        label_col=args.label_col,
        generator_col=args.generator_col,
        target_sr=16000,
        max_duration_sec=args.max_duration_sec,
        distortion_pipeline=distortion_pipeline,
        augmentation_prob=args.augment_prob,
    )

    val_dataset = DeepfakeAudioDataset(
        df=df_val,
        base_dir=args.audio_base_dir,
        filename_col=args.filename_col,
        label_col=args.label_col,
        generator_col=args.generator_col,
        target_sr=16000,
        max_duration_sec=args.max_duration_sec,
        distortion_pipeline=None,
        augmentation_prob=0.0,
    )

    if args.domain_balance_power > 0:
        domain_counts = Counter(train_dataset.domains)
        sample_weights = [
            1.0 / (domain_counts[domain] ** args.domain_balance_power)
            for domain in train_dataset.domains
        ]
        train_sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_dataset),
            replacement=True,
        )
        logger.info(
            "Balancing training examples across %d source/generator domains (power=%.2f)",
            len(domain_counts), args.domain_balance_power,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Initialize model
    logger.info(f"Initializing model with backbone: {args.backbone_id}")
    model = EchoProofClassifier(
        backbone_id=args.backbone_id,
        freeze_backbone_layers=args.freeze_layers,
        pooling=args.pooling,
        dropout=args.dropout,
    ).to(device)

    # Loss and optimizer
    if args.loss == "focal":
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_eer = float("inf")
    history = []

    logger.info("Starting training loop...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        batches = 0

        for batch_idx, batch in enumerate(train_loader):
            waveforms = batch["waveform"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(waveforms)
            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            batches += 1

        avg_train_loss = train_loss / max(1, batches)
        val_metrics = evaluate(model, val_loader, criterion, device)

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_auc_roc": val_metrics["auc_roc"],
            "val_eer": val_metrics["eer"],
        }
        history.append(epoch_record)

        logger.info(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val EER: {val_metrics['eer'] * 100:.2f}% | "
            f"Val AUC: {val_metrics['auc_roc'] * 100:.2f}%"
        )

        # Checkpoint if best EER
        if val_metrics["eer"] < best_val_eer:
            best_val_eer = val_metrics["eer"]
            checkpoint_path = output_dir / "best_model_eer.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "backbone_config": model.backbone.config.to_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_eer": best_val_eer,
                    "config": vars(args),
                    "label_mapping": {0: "bonafide", 1: "spoof"},
                },
                checkpoint_path,
            )
            logger.info(f" Saved new best model to {checkpoint_path} (EER: {best_val_eer * 100:.2f}%)")

    # Save final model and history
    torch.save(model.state_dict(), output_dir / "final_model.pt")
    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time / 60:.2f} minutes. Best Val EER: {best_val_eer * 100:.2f}%")

    best_checkpoint_path = output_dir / "best_model_eer.pt"
    if best_checkpoint_path.is_file():
        best_checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
        calibration = calibrate_thresholds(model, val_loader, device)
        best_checkpoint.update(calibration)
        torch.save(best_checkpoint, best_checkpoint_path)
        with open(output_dir / "decision_thresholds.json", "w", encoding="utf-8") as f:
            json.dump(calibration, f, indent=2)
        logger.info(
            "Validation-calibrated verdict thresholds: genuine<=%.4f, synthetic>=%.4f",
            calibration["threshold_genuine"], calibration["threshold_synthetic"],
        )


def parse_args():
    parser = argparse.ArgumentParser(description="EchoProof AI Model Training")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to training manifest CSV")
    parser.add_argument("--val_csv", type=str, required=True, help="Path to validation manifest CSV")
    parser.add_argument("--audio_base_dir", type=str, required=True, help="Base path for audio files")
    parser.add_argument("--output_dir", type=str, default="./results/checkpoints", help="Output directory")

    parser.add_argument("--backbone_id", type=str, default="facebook/wav2vec2-base", help="Pretrained HF model")
    parser.add_argument("--freeze_layers", type=int, default=8, help="Number of encoder layers to freeze")
    parser.add_argument("--pooling", type=str, default="asp", choices=["asp", "mean"], help="Pooling method")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability")

    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--loss", type=str, default="focal", choices=["ce", "focal"], help="Loss function")
    parser.add_argument("--focal_alpha", type=float, default=0.5, help="Focal loss alpha")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma")

    parser.add_argument("--augment_prob", type=float, default=0.5, help="Distortion augmentation probability")
    parser.add_argument(
        "--domain_balance_power", type=float, default=0.5,
        help="Inverse-frequency sampling exponent by source/generator (0 disables balancing).",
    )
    parser.add_argument("--max_duration_sec", type=float, default=4.0, help="Audio input length in seconds")

    parser.add_argument("--filename_col", type=str, default="filename", help="CSV filename column")
    parser.add_argument("--label_col", type=str, default="label", help="CSV label column")
    parser.add_argument("--generator_col", type=str, default="generator", help="CSV generator column")

    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Compute device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
