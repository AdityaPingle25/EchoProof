"""Tests for FastAPI endpoints."""
import pytest
import io
import soundfile as sf
import numpy as np
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def create_dummy_wav():
    sr = 16000
    data = (np.sin(np.linspace(0, 50, sr)) * 0.5).astype(np.float32)
    bio = io.BytesIO()
    sf.write(bio, data, sr, format='WAV')
    bio.seek(0)
    return bio


def test_root_endpoint(client):
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["name"] == "EchoProof AI"


def test_health_endpoint(client):
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert "model_loaded" in data


def test_model_endpoint(client):
    res = client.get("/api/v1/model")
    assert res.status_code == 200
    data = res.json()
    assert "model_id" in data
    assert "threshold_genuine" in data


def test_analyze_endpoint(client):
    wav_bytes = create_dummy_wav()
    files = {"file": ("test_voice.wav", wav_bytes, "audio/wav")}
    res = client.post("/api/v1/analyze", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "completed"
    assert "verdict" in data
    assert "primary_score" in data
