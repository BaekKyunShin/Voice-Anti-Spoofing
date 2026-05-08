"""FastAPI 앱 — WAV 업로드 → 진위 판별 결과 반환."""

from __future__ import annotations

import io
import os
import time

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import predict_authenticity

app = FastAPI(
    title="Voice Authenticity Detector",
    description="실제 vs 합성 음성 판별 데모 API",
    version="0.1.0",
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

MAX_FILE_BYTES = 25 * 1024 * 1024  # 25MB


class PredictionResponse(BaseModel):
    real_prob: float
    fake_prob: float
    prediction: str  # "real" | "fake"
    inference_ms: float
    filename: str
    sample_rate: int
    duration_sec: float


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "service": "voice-authenticity-detector"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


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

    # 스테레오면 모노로 변환
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    duration = float(len(audio) / sample_rate) if sample_rate else 0.0
    if duration < 0.1:
        raise HTTPException(status_code=400, detail="오디오가 너무 짧습니다.")

    start = time.perf_counter()
    real_prob, fake_prob = predict_authenticity(audio, sample_rate)
    inference_ms = (time.perf_counter() - start) * 1000

    return PredictionResponse(
        real_prob=real_prob,
        fake_prob=fake_prob,
        prediction="real" if real_prob >= fake_prob else "fake",
        inference_ms=inference_ms,
        filename=file.filename,
        sample_rate=int(sample_rate),
        duration_sec=duration,
    )
