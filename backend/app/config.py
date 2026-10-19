"""EchoProof AI - Configuration"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
TEMP_DIR = BASE_DIR / "temp_audio"
TEMP_DIR.mkdir(exist_ok=True)
RESULTS_DIR = BASE_DIR / ".." / "research" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = BASE_DIR / ".." / "research" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Model
MODEL_ID = os.getenv("ECHOPROOF_MODEL_ID", "Sara1708/deepfake-audio-wav2vec2")
MODEL_DEVICE = os.getenv("ECHOPROOF_DEVICE", "cpu")
MODEL_SAMPLE_RATE = 16000
MODEL_MAX_DURATION_SEC = 30  # Max chunk duration for model input
MODEL_CHUNK_DURATION_SEC = 4  # Chunk size for long audio

# Audio
MAX_UPLOAD_SIZE_MB = 25
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
MIN_AUDIO_DURATION_SEC = 0.5
MAX_AUDIO_DURATION_SEC = 300  # 5 minutes
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus"}

# Decision thresholds - UNTUNED defaults
# These are placeholder thresholds and have NOT been validated on a benchmark.
# They should be tuned on a validation set before any research claims.
THRESHOLD_GENUINE = float(os.getenv("ECHOPROOF_THRESHOLD_GENUINE", "0.35"))
THRESHOLD_SYNTHETIC = float(os.getenv("ECHOPROOF_THRESHOLD_SYNTHETIC", "0.65"))
THRESHOLD_IS_TUNED = os.getenv("ECHOPROOF_THRESHOLD_TUNED", "false").lower() == "true"

# Server
CORS_ORIGINS = os.getenv("ECHOPROOF_CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
MAX_CONCURRENT_JOBS = int(os.getenv("ECHOPROOF_MAX_JOBS", "3"))

# Robustness lab defaults
DEFAULT_NOISE_SNR_DB = 20
DEFAULT_RESAMPLE_RATE = 8000
DEFAULT_LOWPASS_CUTOFF = 4000
DEFAULT_GAIN_DB = -6
DEFAULT_MP3_BITRATE = 64
