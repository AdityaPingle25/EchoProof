"""EchoProof AI - Research evaluation pipeline.

Implements reproducible evaluation on labeled datasets with proper
train/validation/test splitting, threshold tuning on validation data only,
and comprehensive metrics with bootstrap confidence intervals.
"""
from __future__ import annotations
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
)

from app.config import RESULTS_DIR, MODEL_SAMPLE_RATE
from app.schemas import (
    DatasetManifest, EvaluationRequest, MetricsResult,
    EvaluationResult, JobStatus, Verdict,
)
from app.audio.processor import AudioProcessor, AudioProcessingError
from app.models.detector import BaseDetector

logger = logging.getLogger(__name__)


def validate_manifest(manifest: DatasetManifest) -> tuple[pd.DataFrame, list[str]]:
    """
    Validate a dataset manifest and return the loaded metadata and warnings.
    """
    warnings = []
    base_path = Path(manifest.base_path)
    csv_path = base_path / manifest.metadata_csv

    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Check required columns
    for col in [manifest.filename_column, manifest.label_column]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in CSV. Columns: {list(df.columns)}")

    # Map labels to binary
    all_genuine = set(manifest.genuine_labels)
    all_synthetic = set(manifest.synthetic_labels)

    df["_binary_label"] = df[manifest.label_column].apply(
        lambda x: "genuine" if str(x).lower().strip() in {l.lower() for l in all_genuine}
        else ("synthetic" if str(x).lower().strip() in {l.lower() for l in all_synthetic}
              else "unknown")
    )

    unknown_count = (df["_binary_label"] == "unknown").sum()
    if unknown_count > 0:
        unknown_labels = df[df["_binary_label"] == "unknown"][manifest.label_column].unique()
        warnings.append(
            f"{unknown_count} samples with unmapped labels: {list(unknown_labels)[:10]}. "
            f"These will be excluded."
        )
        df = df[df["_binary_label"] != "unknown"]

    if len(df) == 0:
        raise ValueError("No valid samples after label mapping.")

    # Check files exist (sample check)
    sample_files = df[manifest.filename_column].head(10)
    missing = 0
    for fname in sample_files:
        fpath = base_path / fname
        if not fpath.exists():
            missing += 1
    if missing > 0:
        warnings.append(f"{missing}/{len(sample_files)} sampled files not found at {base_path}.")

    # Check for duplicates
    dupes = df[manifest.filename_column].duplicated().sum()
    if dupes > 0:
        warnings.append(f"{dupes} duplicate filenames found.")

    n_genuine = (df["_binary_label"] == "genuine").sum()
    n_synthetic = (df["_binary_label"] == "synthetic").sum()
    logger.info(f"Dataset: {len(df)} samples ({n_genuine} genuine, {n_synthetic} synthetic)")

    return df, warnings


def split_dataset(
    df: pd.DataFrame,
    manifest: DatasetManifest,
    validation_split: float = 0.2,
    speaker_disjoint: bool = True,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Split into validation and test sets."""
    warnings = []
    rng = np.random.RandomState(random_seed)

    if speaker_disjoint and manifest.speaker_column and manifest.speaker_column in df.columns:
        speakers = df[manifest.speaker_column].unique()
        rng.shuffle(speakers)
        n_val = max(1, int(len(speakers) * validation_split))
        val_speakers = set(speakers[:n_val])
        val_df = df[df[manifest.speaker_column].isin(val_speakers)]
        test_df = df[~df[manifest.speaker_column].isin(val_speakers)]
        warnings.append(f"Speaker-disjoint split: {n_val} val speakers, {len(speakers) - n_val} test speakers.")
    else:
        if speaker_disjoint:
            warnings.append(
                "Speaker-disjoint split requested but speaker column not available. "
                "Falling back to random split."
            )
        indices = rng.permutation(len(df))
        n_val = max(1, int(len(df) * validation_split))
        val_df = df.iloc[indices[:n_val]]
        test_df = df.iloc[indices[n_val:]]

    # Verify both splits have both classes
    for split_name, split_df in [("validation", val_df), ("test", test_df)]:
        classes = split_df["_binary_label"].unique()
        if len(classes) < 2:
            warnings.append(f"{split_name} split has only class(es): {list(classes)}. Metrics may be undefined.")

    return val_df, test_df, warnings


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Optional[float]:
    """Compute Equal Error Rate (EER) safely without duplicate-FPR interpolation crashes."""
    y_true_arr = np.asarray(y_true)
    y_scores_arr = np.asarray(y_scores)

    if len(y_true_arr) < 2 or len(y_scores_arr) < 2:
        logger.warning("EER computation failed: fewer than 2 samples provided.")
        return None

    unique_classes = np.unique(y_true_arr)
    if len(unique_classes) < 2:
        logger.warning(f"EER computation failed: only one class ({unique_classes[0]}) present in ground truth.")
        return None

    try:
        fpr, tpr, thresholds = roc_curve(y_true_arr, y_scores_arr, pos_label=1)
        fnr = 1.0 - tpr

        # Check for perfect separation
        if np.any((fpr == 0.0) & (fnr == 0.0)):
            return 0.0

        diffs = fpr - fnr
        exact_match = np.where(diffs == 0.0)[0]
        if len(exact_match) > 0:
            return float(fpr[exact_match[0]])

        crossing_idx = np.where(diffs > 0.0)[0]
        if len(crossing_idx) == 0:
            min_idx = np.argmin(np.abs(diffs))
            return float((fpr[min_idx] + fnr[min_idx]) / 2.0)

        idx = int(crossing_idx[0])
        idx_prev = idx - 1

        x1, y1 = float(fpr[idx_prev]), float(fnr[idx_prev])
        x2, y2 = float(fpr[idx]), float(fnr[idx])

        if x1 == x2:
            return float(x1)

        m = (y2 - y1) / (x2 - x1)
        if m == 1.0:
            return float((x1 + y1) / 2.0)

        eer = (y1 - m * x1) / (1.0 - m)
        return float(np.clip(eer, 0.0, 1.0))
    except Exception as e:
        logger.warning(f"EER computation failed: {e}")
        return None


def compute_risk_coverage(
    y_true: np.ndarray, y_pred: np.ndarray, confidence: np.ndarray
) -> list[dict[str, float]]:
    """
    Compute risk-coverage curve.
    Risk = error rate; Coverage = fraction of samples not abstained.
    Samples are sorted by confidence (descending), and we compute
    cumulative risk as we include more samples.
    """
    sorted_indices = np.argsort(-confidence)
    y_true_sorted = y_true[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]

    n = len(y_true)
    points = []
    errors = 0
    for i in range(n):
        if y_true_sorted[i] != y_pred_sorted[i]:
            errors += 1
        coverage = (i + 1) / n
        risk = errors / (i + 1)
        if (i + 1) % max(1, n // 100) == 0 or i == n - 1:
            points.append({"coverage": round(coverage, 4), "risk": round(risk, 4)})

    return points


def compute_aurc(risk_coverage: list[dict[str, float]]) -> float:
    """Compute Area Under Risk-Coverage Curve using trapezoidal rule."""
    coverages = [p["coverage"] for p in risk_coverage]
    risks = [p["risk"] for p in risk_coverage]
    trapz_func = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    return float(trapz_func(risks, coverages))


def bootstrap_ci(
    y_true: np.ndarray, y_scores: np.ndarray,
    metric_fn, n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 42
) -> Optional[list[float]]:
    """Compute bootstrap confidence interval for a metric."""
    if len(y_true) < 10:
        return None
    rng = np.random.RandomState(seed)
    values = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, len(y_true), size=len(y_true))
        try:
            val = metric_fn(y_true[idx], y_scores[idx])
            if val is not None and np.isfinite(val):
                values.append(val)
        except Exception:
            continue

    if len(values) < 100:
        return None
    alpha = (1 - ci) / 2
    lower = float(np.percentile(values, alpha * 100))
    upper = float(np.percentile(values, (1 - alpha) * 100))
    return [lower, upper]


def compute_metrics(
    y_true: np.ndarray, y_scores: np.ndarray,
    threshold: float, threshold_source: str = "untuned_default",
    compute_bootstrap: bool = True,
) -> MetricsResult:
    """Compute all metrics for a set of predictions."""
    n_samples = len(y_true)
    n_genuine = int((y_true == 0).sum())
    n_synthetic = int((y_true == 1).sum())

    y_pred = (y_scores >= threshold).astype(int)

    result = MetricsResult(
        n_samples=n_samples,
        n_genuine=n_genuine,
        n_synthetic=n_synthetic,
        threshold_used=threshold,
        threshold_source=threshold_source,
    )

    if n_samples < 2 or n_genuine == 0 or n_synthetic == 0:
        result.accuracy = float(accuracy_score(y_true, y_pred)) if n_samples > 0 else None
        return result

    try:
        result.accuracy = float(accuracy_score(y_true, y_pred))
        result.precision = float(precision_score(y_true, y_pred, zero_division=0))
        result.recall = float(recall_score(y_true, y_pred, zero_division=0))
        result.f1 = float(f1_score(y_true, y_pred, zero_division=0))
        result.roc_auc = float(roc_auc_score(y_true, y_scores))
        result.eer = compute_eer(y_true, y_scores)
        cm = confusion_matrix(y_true, y_pred)
        result.confusion_matrix = cm.tolist()
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        result.fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else None
        result.fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else None
    except Exception as e:
        logger.warning(f"Metric computation error: {e}")

    # Risk-coverage
    try:
        confidence = np.abs(y_scores - 0.5) * 2  # Simple confidence proxy
        result.risk_coverage = compute_risk_coverage(y_true, y_pred, confidence)
        result.aurc = compute_aurc(result.risk_coverage)
    except Exception as e:
        logger.warning(f"Risk-coverage computation error: {e}")

    # Bootstrap CIs
    if compute_bootstrap and n_samples >= 30:
        try:
            ci = {}
            if result.roc_auc is not None:
                ci["roc_auc"] = bootstrap_ci(y_true, y_scores, roc_auc_score)
            if result.accuracy is not None:
                ci["accuracy"] = bootstrap_ci(
                    y_true, y_scores,
                    lambda yt, ys: accuracy_score(yt, (ys >= threshold).astype(int))
                )
            result.bootstrap_ci_95 = {k: v for k, v in ci.items() if v is not None}
        except Exception as e:
            logger.warning(f"Bootstrap CI computation error: {e}")

    return result


async def run_evaluation(
    job_id: str,
    request: EvaluationRequest,
    detector: BaseDetector,
    progress_callback=None,
) -> EvaluationResult:
    """Run a complete evaluation experiment."""
    start_time = time.time()
    warnings = []
    processor = AudioProcessor()

    # Validate manifest
    try:
        df, manifest_warnings = validate_manifest(request.manifest)
        warnings.extend(manifest_warnings)
    except Exception as e:
        return EvaluationResult(
            job_id=job_id, status=JobStatus.FAILED,
            error=f"Manifest validation failed: {e}",
            model_id=detector.get_model_info().get("model_id", ""),
        )

    # Limit samples if requested
    if request.max_samples and len(df) > request.max_samples:
        df = df.sample(n=request.max_samples, random_state=request.random_seed)
        warnings.append(f"Limited to {request.max_samples} samples.")

    # Split dataset
    val_df, test_df, split_warnings = split_dataset(
        df, request.manifest, request.validation_split,
        request.speaker_disjoint, request.random_seed,
    )
    warnings.extend(split_warnings)

    base_path = Path(request.manifest.base_path)

    async def score_dataset(
        split_df: pd.DataFrame, split_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Score all samples in a split."""
        y_true = []
        y_scores = []
        total = len(split_df)

        for i, (_, row) in enumerate(split_df.iterrows()):
            file_path = base_path / row[request.manifest.filename_column]
            label = 1 if row["_binary_label"] == "synthetic" else 0

            try:
                waveform, sr = processor.load_audio(file_path)
                waveform = processor.preprocess_for_model(waveform, sr)
                result = detector.score(waveform, MODEL_SAMPLE_RATE)
                y_true.append(label)
                y_scores.append(result.primary_score)
            except Exception as e:
                logger.warning(f"Skipping {file_path}: {e}")
                continue

            if progress_callback and (i + 1) % 10 == 0:
                progress_callback(
                    progress=(i + 1) / total,
                    message=f"Scoring {split_name}: {i + 1}/{total}"
                )

        return np.array(y_true), np.array(y_scores)

    # Score validation set
    val_y_true, val_y_scores = await score_dataset(val_df, "validation")

    if len(val_y_true) < 2:
        return EvaluationResult(
            job_id=job_id, status=JobStatus.FAILED,
            error="Too few valid validation samples.",
            warnings=warnings,
            model_id=detector.get_model_info().get("model_id", ""),
        )

    # Tune threshold on validation set
    try:
        fpr, tpr, thresholds = roc_curve(val_y_true, val_y_scores)
        # Find threshold that minimizes |FPR - FNR| (EER point)
        fnr = 1 - tpr
        idx = np.nanargmin(np.abs(fpr - fnr))
        tuned_threshold = float(thresholds[idx]) if idx < len(thresholds) else 0.5
        threshold_source = "validation_eer"
    except Exception:
        tuned_threshold = 0.5
        threshold_source = "fallback_0.5"
        warnings.append("Threshold tuning failed. Using 0.5.")

    # Compute validation metrics
    val_metrics = compute_metrics(
        val_y_true, val_y_scores, tuned_threshold, threshold_source
    )

    # Score test set
    test_y_true, test_y_scores = await score_dataset(test_df, "test")

    test_metrics = None
    if len(test_y_true) >= 2:
        test_metrics = compute_metrics(
            test_y_true, test_y_scores, tuned_threshold,
            f"tuned_on_validation ({threshold_source})"
        )
    else:
        warnings.append("Too few valid test samples for metrics.")

    # Save results
    experiment_config = {
        "model_id": detector.get_model_info().get("model_id", ""),
        "dataset": request.manifest.name,
        "random_seed": request.random_seed,
        "validation_split": request.validation_split,
        "speaker_disjoint": request.speaker_disjoint,
        "max_samples": request.max_samples,
        "tuned_threshold": tuned_threshold,
        "threshold_source": threshold_source,
        "n_validation": len(val_y_true),
        "n_test": len(test_y_true),
        "timestamp": time.time(),
    }

    # Save to files
    try:
        results_dir = RESULTS_DIR / job_id
        results_dir.mkdir(parents=True, exist_ok=True)

        with open(results_dir / "experiment_config.json", "w") as f:
            json.dump(experiment_config, f, indent=2)

        if val_metrics:
            with open(results_dir / "validation_metrics.json", "w") as f:
                json.dump(val_metrics.model_dump(), f, indent=2, default=str)

        if test_metrics:
            with open(results_dir / "test_metrics.json", "w") as f:
                json.dump(test_metrics.model_dump(), f, indent=2, default=str)

        # Save predictions
        predictions = []
        for y_t, y_s, split_name, split_df_iter in [
            (val_y_true, val_y_scores, "validation", val_df),
            (test_y_true, test_y_scores, "test", test_df),
        ]:
            for (_, row), yt, ys in zip(split_df_iter.iterrows(), y_t, y_s):
                predictions.append({
                    "filename": row[request.manifest.filename_column],
                    "true_label": int(yt),
                    "score": float(ys),
                    "predicted_label": int(ys >= tuned_threshold),
                    "split": split_name,
                })
        pd.DataFrame(predictions).to_csv(results_dir / "predictions.csv", index=False)

    except Exception as e:
        warnings.append(f"Failed to save some result files: {e}")

    elapsed = time.time() - start_time

    return EvaluationResult(
        job_id=job_id,
        status=JobStatus.COMPLETED,
        dataset_name=request.manifest.name,
        validation_metrics=val_metrics,
        test_metrics=test_metrics,
        experiment_config=experiment_config,
        model_id=detector.get_model_info().get("model_id", ""),
        warnings=warnings,
        processing_time_sec=round(elapsed, 2),
    )
