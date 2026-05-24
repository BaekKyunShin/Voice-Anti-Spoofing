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

# ASVspoof 표준 CM score 임계값. bonafide_prob >= 이면 진짜로 판정.
# 추후 EER 기반 튜닝 여지.
BONAFIDE_THRESHOLD = 0.5


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
    bonafide_prob: float  # ASVspoof CM score. 높을수록 진짜.
    prediction: str  # "bonafide" | "spoof"
    inference_ms: float


class Consensus(BaseModel):
    prediction: str  # "bonafide" | "spoof"
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
    votes = [
        "bonafide" if m["bonafide_prob"] >= BONAFIDE_THRESHOLD else "spoof"
        for m in models.values()
    ]
    bonafide_count = votes.count("bonafide")
    spoof_count = votes.count("spoof")
    if bonafide_count >= spoof_count:
        prediction = "bonafide"
        agreement = bonafide_count / len(votes)
    else:
        prediction = "spoof"
        agreement = spoof_count / len(votes)
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
            bonafide_prob=raw_results[k]["bonafide_prob"],
            prediction=(
                "bonafide"
                if raw_results[k]["bonafide_prob"] >= BONAFIDE_THRESHOLD
                else "spoof"
            ),
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
