"""EchoProof AI - FastAPI routes."""
from __future__ import annotations
import asyncio
import io
import logging
import time
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse

from app.config import (
    MODEL_SAMPLE_RATE, MODEL_CHUNK_DURATION_SEC,
    RESULTS_DIR,
)
from app.schemas import (
    HealthResponse, ModelInfoResponse, AnalysisResult,
    AudioMetadata, ChunkResult, RobustnessResult,
    TransformationResult, TransformationConfig,
    RobustnessRequest, EvaluationRequest, EvaluationResult,
    JobStatus, JobInfo, Verdict,
)
from app.audio.processor import AudioProcessor, AudioProcessingError, save_temp_audio, cleanup_temp_file
from app.audio.transforms import AudioTransformer, TransformationError
from app.models.detector import get_detector, DetectorResult
from app.jobs.manager import get_job_manager
from app.evaluation.pipeline import run_evaluation

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")
audio_processor = AudioProcessor()
audio_transformer = AudioTransformer()


# ── Health & Model ──────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health():
    detector = get_detector()
    info = detector.get_model_info()
    return HealthResponse(
        model_loaded=detector.is_loaded() and info["model_type"] != "Detector unavailable (neutral abstention)",
        timestamp=time.time(),
    )


@router.get("/model", response_model=ModelInfoResponse)
async def model_info():
    detector = get_detector()
    info = detector.get_model_info()
    return ModelInfoResponse(
        model_id=info["model_id"],
        model_type=info["model_type"],
        device=info["device"],
        sample_rate=info["sample_rate"],
        max_duration_sec=300,
        chunk_duration_sec=MODEL_CHUNK_DURATION_SEC,
        threshold_genuine=info["threshold_genuine"],
        threshold_synthetic=info["threshold_synthetic"],
        threshold_is_tuned=info["threshold_is_tuned"],
        label_mapping=info["label_mapping"],
        notes=info["notes"],
    )


# ── Audio Analysis ──────────────────────────────────────────────────────────

@router.post("/analyze", response_model=AnalysisResult)
async def analyze_audio(file: UploadFile = File(...)):
    """Analyze an uploaded audio file for speech deepfake detection."""
    detector = get_detector()
    if not detector.is_loaded():
        raise HTTPException(503, "Model not loaded. Backend is starting up.")
    detector_info = detector.get_model_info()

    start_time = time.time()
    job_manager = get_job_manager()
    job_id = job_manager.create_job("analyze")
    temp_path = None
    warnings = []

    try:
        # Read file
        content = await file.read()
        filename = file.filename or "audio.wav"

        # Validate
        audio_processor.validate_file(filename, len(content))

        # Save temp file
        temp_path = save_temp_audio(content, filename)

        # Load and process audio
        waveform, sr = audio_processor.load_audio(temp_path)
        metadata = audio_processor.get_metadata(filename, waveform, sr, len(content))

        # Generate visualizations
        waveform_data = audio_processor.generate_waveform_data(waveform)
        spectrogram_data = audio_processor.generate_spectrogram_data(waveform, sr)

        # Preprocess for model
        model_waveform = audio_processor.preprocess_for_model(waveform, sr)

        # Chunk if needed
        chunks = audio_processor.chunk_audio(
            model_waveform, MODEL_SAMPLE_RATE, MODEL_CHUNK_DURATION_SEC
        )

        # Run inference
        chunk_results = []
        all_scores = []

        for i, (chunk_wav, start_sec, end_sec) in enumerate(chunks):
            result = detector.score(chunk_wav, MODEL_SAMPLE_RATE)
            chunk_results.append(ChunkResult(
                chunk_index=i,
                start_sec=round(start_sec, 3),
                end_sec=round(end_sec, 3),
                raw_scores=result.raw_scores,
                verdict=result.verdict,
                primary_score=result.primary_score,
            ))
            all_scores.append(result.primary_score)
            warnings.extend(result.warnings)

        # Aggregate scores
        if len(all_scores) == 1:
            final_score = all_scores[0]
            aggregation = "single_chunk"
        else:
            final_score = float(sum(all_scores) / len(all_scores))
            aggregation = f"mean_of_{len(all_scores)}_chunks"
            warnings.append(
                f"Audio was split into {len(all_scores)} chunks. "
                f"Final score is the mean. Individual chunk scores may vary."
            )

        # Apply decision thresholds
        threshold_synthetic = float(detector_info["threshold_synthetic"])
        threshold_genuine = float(detector_info["threshold_genuine"])
        if final_score >= threshold_synthetic:
            verdict = Verdict.LIKELY_SYNTHETIC
        elif final_score <= threshold_genuine:
            verdict = Verdict.LIKELY_GENUINE
        else:
            verdict = Verdict.UNCERTAIN

        elapsed = time.time() - start_time

        # Deduplicate warnings
        warnings = list(dict.fromkeys(warnings))

        analysis = AnalysisResult(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            verdict=verdict,
            primary_score=round(final_score, 6),
            score_label="Model score (softmax-normalized, uncalibrated)",
            raw_scores=chunk_results[0].raw_scores if len(chunk_results) == 1 else None,
            chunk_results=chunk_results if len(chunk_results) > 1 else None,
            aggregation_method=aggregation,
            model_id=detector_info["model_id"],
            model_device=detector_info["device"],
            threshold_genuine=threshold_genuine,
            threshold_synthetic=threshold_synthetic,
            threshold_is_tuned=detector_info["threshold_is_tuned"],
            audio_metadata=metadata,
            waveform_data=waveform_data,
            spectrogram_data=spectrogram_data,
            processing_time_sec=round(elapsed, 3),
            warnings=warnings,
        )

        job_manager.update_job(job_id, status=JobStatus.COMPLETED, result=analysis)
        return analysis

    except AudioProcessingError as e:
        raise HTTPException(422, detail=str(e))
    except Exception as e:
        logger.exception("Analysis failed")
        raise HTTPException(500, detail=f"Analysis failed: {str(e)}")
    finally:
        if temp_path:
            cleanup_temp_file(temp_path)


# ── Robustness Lab ──────────────────────────────────────────────────────────

@router.post("/robustness", response_model=RobustnessResult)
async def robustness_test(
    file: UploadFile = File(...),
    noise_snr_db: float | None = None,
    resample_rate: int | None = None,
    lowpass_cutoff: float | None = None,
    gain_db: float | None = None,
    mp3_bitrate: int | None = None,
):
    """Run robustness transformations on uploaded audio."""
    detector = get_detector()
    if not detector.is_loaded():
        raise HTTPException(503, "Model not loaded.")

    start_time = time.time()
    job_manager = get_job_manager()
    job_id = job_manager.create_job("robustness")
    temp_path = None

    # Build transformation list from query params
    transformations = []
    if noise_snr_db is not None:
        transformations.append(TransformationConfig(
            type="noise", params={"snr_db": noise_snr_db}
        ))
    if resample_rate is not None:
        transformations.append(TransformationConfig(
            type="resample", params={"target_sr": resample_rate}
        ))
    if lowpass_cutoff is not None:
        transformations.append(TransformationConfig(
            type="lowpass", params={"cutoff_hz": lowpass_cutoff}
        ))
    if gain_db is not None:
        transformations.append(TransformationConfig(
            type="gain", params={"gain_db": gain_db}
        ))
    if mp3_bitrate is not None:
        transformations.append(TransformationConfig(
            type="mp3_compress", params={"bitrate": mp3_bitrate}
        ))

    # Default transformations if none specified
    if not transformations:
        transformations = [
            TransformationConfig(type="noise", params={"snr_db": 20.0}),
            TransformationConfig(type="resample", params={"target_sr": 8000}),
            TransformationConfig(type="lowpass", params={"cutoff_hz": 4000.0}),
            TransformationConfig(type="gain", params={"gain_db": -6.0}),
        ]

    try:
        content = await file.read()
        filename = file.filename or "audio.wav"
        audio_processor.validate_file(filename, len(content))
        temp_path = save_temp_audio(content, filename)
        waveform, sr = audio_processor.load_audio(temp_path)
        model_waveform = audio_processor.preprocess_for_model(waveform, sr)

        # Score original
        original_result = detector.score(model_waveform, MODEL_SAMPLE_RATE)
        original_analysis = AnalysisResult(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            verdict=original_result.verdict,
            primary_score=original_result.primary_score,
            raw_scores=original_result.raw_scores,
            model_id=original_result.model_id,
        )

        # Run transformations
        transformation_results = []
        all_scores = [original_result.primary_score]

        for config in transformations:
            try:
                transformed = audio_transformer.apply(
                    model_waveform, MODEL_SAMPLE_RATE, config.type, config.params
                )
                t_result = detector.score(transformed, MODEL_SAMPLE_RATE)
                score_change = t_result.primary_score - original_result.primary_score
                transformation_results.append(TransformationResult(
                    transformation=config,
                    original_score=original_result.primary_score,
                    transformed_score=t_result.primary_score,
                    original_verdict=original_result.verdict,
                    transformed_verdict=t_result.verdict,
                    score_change=round(score_change, 6),
                    verdict_changed=t_result.verdict != original_result.verdict,
                    transformation_succeeded=True,
                ))
                all_scores.append(t_result.primary_score)
            except Exception as e:
                transformation_results.append(TransformationResult(
                    transformation=config,
                    original_score=original_result.primary_score,
                    transformed_score=original_result.primary_score,
                    original_verdict=original_result.verdict,
                    transformed_verdict=original_result.verdict,
                    score_change=0.0,
                    verdict_changed=False,
                    transformation_succeeded=False,
                    error=str(e),
                ))

        import numpy as np
        scores_arr = np.array(all_scores)
        elapsed = time.time() - start_time

        result = RobustnessResult(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            original_analysis=original_analysis,
            transformation_results=transformation_results,
            score_std=round(float(np.std(scores_arr)), 6),
            max_score_change=round(float(np.max(np.abs(scores_arr[1:] - scores_arr[0]))), 6) if len(scores_arr) > 1 else 0.0,
            threshold_crossings=sum(1 for tr in transformation_results if tr.verdict_changed),
            processing_time_sec=round(elapsed, 3),
        )

        job_manager.update_job(job_id, status=JobStatus.COMPLETED, result=result)
        return result

    except AudioProcessingError as e:
        raise HTTPException(422, detail=str(e))
    except Exception as e:
        logger.exception("Robustness test failed")
        raise HTTPException(500, detail=f"Robustness test failed: {str(e)}")
    finally:
        if temp_path:
            cleanup_temp_file(temp_path)


# ── Evaluation ──────────────────────────────────────────────────────────────

@router.post("/evaluation/run")
async def start_evaluation(request: EvaluationRequest, background_tasks: BackgroundTasks):
    """Start a dataset evaluation job."""
    detector = get_detector()
    if not detector.is_loaded():
        raise HTTPException(503, "Model not loaded.")

    job_manager = get_job_manager()
    job_id = job_manager.create_job("evaluation")

    async def _run():
        def progress_cb(progress, message):
            job_manager.update_job(job_id, progress=progress, message=message)

        result = await run_evaluation(job_id, request, detector, progress_cb)
        job_manager.update_job(
            job_id,
            status=result.status,
            progress=1.0,
            result=result,
        )

    background_tasks.add_task(_run)
    return {"job_id": job_id, "status": "pending"}


@router.get("/evaluation/results")
async def list_evaluation_results():
    """List available evaluation results."""
    results = []
    if RESULTS_DIR.exists():
        for d in sorted(RESULTS_DIR.iterdir()):
            if d.is_dir():
                config_file = d / "experiment_config.json"
                if config_file.exists():
                    import json
                    with open(config_file) as f:
                        config = json.load(f)
                    results.append({
                        "job_id": d.name,
                        "dataset": config.get("dataset", ""),
                        "timestamp": config.get("timestamp", 0),
                        "n_validation": config.get("n_validation", 0),
                        "n_test": config.get("n_test", 0),
                    })
    return {"results": results}


@router.get("/evaluation/results/{job_id}")
async def get_evaluation_result(job_id: str):
    """Get evaluation results for a specific job."""
    # Check job manager first
    job_manager = get_job_manager()
    job = job_manager.get_job(job_id)
    if job and job.result:
        return job.result

    # Check saved results
    result_dir = RESULTS_DIR / job_id
    if not result_dir.exists():
        raise HTTPException(404, "Evaluation result not found.")

    import json
    response = {"job_id": job_id}
    for fname in ["experiment_config.json", "validation_metrics.json", "test_metrics.json"]:
        fpath = result_dir / fname
        if fpath.exists():
            with open(fpath) as f:
                key = fname.replace(".json", "")
                response[key] = json.load(f)

    return response


@router.get("/evaluation/export/{job_id}")
async def export_evaluation(job_id: str):
    """Export evaluation predictions as CSV."""
    csv_path = RESULTS_DIR / job_id / "predictions.csv"
    if not csv_path.exists():
        raise HTTPException(404, "Predictions file not found.")

    return StreamingResponse(
        open(csv_path, "rb"),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=predictions_{job_id}.csv"},
    )


# ── Jobs ────────────────────────────────────────────────────────────────────

@router.get("/jobs/{job_id}", response_model=JobInfo)
async def get_job(job_id: str):
    job_manager = get_job_manager()
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    return job


@router.get("/jobs/{job_id}/events")
async def get_job_events(job_id: str):
    job_manager = get_job_manager()
    events = job_manager.get_events(job_id)
    return {"job_id": job_id, "events": events}
