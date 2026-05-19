"""FastAPI 앱 — WAV 업로드 → 4-모델 진위 판별 결과 반환."""

from __future__ import annotations

import io
import os
import time
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import load_all, predict_all_models

MAX_FILE_BYTES = 25 * 1024 * 1024  # 25MB
MODEL_KEYS = ("gru", "lcnn", "crnn", "xlsr_aasist")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 부팅 시 4개 모델 + XLS-R 베이스를 RAM에 상주시킨다.
    load_all()
    yield


app = FastAPI(
    title="Voice Authenticity Detector",
    description="실제 vs 합성 음성 판별 데모 API (4-모델 앙상블)",
    version="0.2.0",
    lifespan=lifespan,
)

_default_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
_extra_origins = [
    o.strip()
    for o in os.environ.get("ALLOWED_ORIGINS", "").split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_default_origins + _extra_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class ModelResult(BaseModel):
    real_prob: float
    fake_prob: float
    prediction: str  # "real" | "fake"
    inference_ms: float


class Consensus(BaseModel):
    prediction: str  # "real" | "fake"
    agreement: float  # 0.0 ~ 1.0


class PredictionResponse(BaseModel):
    filename: str
    sample_rate: int
    duration_sec: float
    total_inference_ms: float
    models: dict[str, ModelResult]
    consensus: Consensus


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "service": "voice-authenticity-detector"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


def _compute_consensus(models: dict[str, dict]) -> Consensus:
    votes = ["real" if m["real_prob"] >= m["fake_prob"] else "fake" for m in models.values()]
    real_count = votes.count("real")
    fake_count = votes.count("fake")
    if real_count >= fake_count:
        prediction = "real"
        agreement = real_count / len(votes)
    else:
        prediction = "fake"
        agreement = fake_count / len(votes)
    return Consensus(prediction=prediction, agreement=agreement)


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="WAV 파일만 지원합니다.")

    raw = await file.read()
    if len(raw) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"파일이 너무 큽니다 (최대 {MAX_FILE_BYTES // (1024 * 1024)}MB)",
        )

    try:
        audio, sample_rate = sf.read(io.BytesIO(raw), dtype="float32")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"WAV 디코딩 실패: {e}") from e

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    duration = float(len(audio) / sample_rate) if sample_rate else 0.0
    if duration < 0.1:
        raise HTTPException(status_code=400, detail="오디오가 너무 짧습니다.")

    start = time.perf_counter()
    raw_results = predict_all_models(audio, sample_rate)
    total_inference_ms = (time.perf_counter() - start) * 1000

    models = {
        k: ModelResult(
            real_prob=raw_results[k]["real_prob"],
            fake_prob=raw_results[k]["fake_prob"],
            prediction="real" if raw_results[k]["real_prob"] >= raw_results[k]["fake_prob"] else "fake",
            inference_ms=raw_results[k]["inference_ms"],
        )
        for k in MODEL_KEYS
    }
    consensus = _compute_consensus(raw_results)

    return PredictionResponse(
        filename=file.filename,
        sample_rate=int(sample_rate),
        duration_sec=duration,
        total_inference_ms=total_inference_ms,
        models=models,
        consensus=consensus,
    )
