"""Smoke test: 4개 wav를 predict_all_models에 통과시켜 구조 정합성 확인.

정확성(ipynb와 숫자 일치) 검증은 expected.txt 받은 후 별도 테스트로 추가.
이 테스트는 (1) 코드가 에러 없이 돈다 (2) 응답 스키마가 맞다 만 확인.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import soundfile as sf

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

SAMPLES_DIR = BACKEND_DIR.parent / "docs" / "reference" / "test_samples"

WAV_FILES = [
    "real_01.wav",
    "fake_clova_01.wav",
    "fake_elevenlabs_01.wav",
    "fake_google_01.wav",
]

MODEL_KEYS = ("gru", "lcnn", "crnn", "xlsr_aasist")


@pytest.fixture(scope="module")
def predict():
    from model import predict_all_models, load_all
    load_all()
    return predict_all_models


@pytest.mark.parametrize("wav_name", WAV_FILES)
def test_predict_returns_4_models(predict, wav_name):
    wav_path = SAMPLES_DIR / wav_name
    assert wav_path.exists(), f"테스트 wav 없음: {wav_path}"

    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    result = predict(audio, sr)

    assert set(result.keys()) == set(MODEL_KEYS), f"모델 키 누락: {result.keys()}"

    for key in MODEL_KEYS:
        r = result[key]
        assert {"bonafide_prob", "inference_ms"} <= r.keys()
        assert 0.0 <= r["bonafide_prob"] <= 1.0
        assert r["inference_ms"] > 0
