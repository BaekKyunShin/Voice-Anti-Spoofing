"""4-모델 음성 진위 판별 통합 추론.

서버 부팅 시 _load_all()이 한 번 호출되어 4개 모델 + XLS-R 베이스를 RAM에 상주시킨다.
predict_all_models()는 wav → 4개 모델 각각의 (real_prob, fake_prob, inference_ms)를 반환.

라벨: real=0, fake=1 (ipynb 셀 20 기준).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.architectures import (
    AASISTEmbeddingsClassifier,
    CRNNClassifier,
    GRUClassifier,
    LCNNClassifier,
)
from models.preprocess import (
    XLSR_SAMPLE_RATE,
    make_mel_spectrogram,
    prepare_xlsr_waveform,
)

WEIGHTS_DIR = Path(__file__).parent / "models" / "weights"
XLSR_MODEL_NAME = "facebook/wav2vec2-xls-r-300m"
DEVICE = torch.device("cpu")

_BUNDLE: dict = {}


def _load_state_dict(model: torch.nn.Module, path: Path) -> torch.nn.Module:
    state = torch.load(path, map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model.to(DEVICE)


def load_all() -> None:
    """4개 분류기 + XLS-R 베이스를 RAM에 1회 로드. 부팅 시 호출."""
    if _BUNDLE:
        return

    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

    _BUNDLE["gru"] = _load_state_dict(GRUClassifier(), WEIGHTS_DIR / "GRU_best.pt")
    _BUNDLE["lcnn"] = _load_state_dict(LCNNClassifier(), WEIGHTS_DIR / "LCNN_best.pt")
    _BUNDLE["crnn"] = _load_state_dict(CRNNClassifier(), WEIGHTS_DIR / "CRNN_best.pt")
    _BUNDLE["xlsr_aasist"] = _load_state_dict(
        AASISTEmbeddingsClassifier(), WEIGHTS_DIR / "XLSR_AASIST_best.pt"
    )

    _BUNDLE["xlsr_feature_extractor"] = Wav2Vec2FeatureExtractor.from_pretrained(
        XLSR_MODEL_NAME
    )
    xlsr = Wav2Vec2Model.from_pretrained(XLSR_MODEL_NAME)
    xlsr.eval()
    _BUNDLE["xlsr_base"] = xlsr.to(DEVICE)


def _softmax_probs(logits: torch.Tensor) -> tuple[float, float]:
    """logits (1, 2) → (real_prob, fake_prob). 라벨 0=real, 1=fake."""
    probs = F.softmax(logits, dim=-1).squeeze(0).tolist()
    return float(probs[0]), float(probs[1])


def _run_gru(mel: np.ndarray) -> tuple[float, float]:
    # (80, 400) → (1, 400, 80) — VoiceMelDataset.model_type="rnn"에서 transpose(0,1)와 동일
    x = torch.from_numpy(mel).transpose(0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return _softmax_probs(_BUNDLE["gru"](x))


def _run_lcnn(mel: np.ndarray) -> tuple[float, float]:
    # (80, 400) → (1, 1, 80, 400)
    x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return _softmax_probs(_BUNDLE["lcnn"](x))


def _run_crnn(mel: np.ndarray) -> tuple[float, float]:
    x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return _softmax_probs(_BUNDLE["crnn"](x))


def _run_xlsr_aasist(waveform: np.ndarray) -> tuple[float, float]:
    """raw waveform (80000,) → XLS-R 임베딩 추출 → AASIST 분류."""
    # transformers 5.0+: truncation=True 사용 시 max_length 필수.
    # 어차피 prepare_xlsr_waveform()이 이미 MAX_AUDIO_LENGTH로 잘라/패딩 했으므로 명시.
    from models.preprocess import MAX_AUDIO_LENGTH
    inputs = _BUNDLE["xlsr_feature_extractor"](
        waveform,
        sampling_rate=XLSR_SAMPLE_RATE,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_AUDIO_LENGTH,
    )
    input_values = inputs.input_values.to(DEVICE)
    with torch.no_grad():
        xlsr_out = _BUNDLE["xlsr_base"](input_values, output_hidden_states=False)
        embedding = xlsr_out.last_hidden_state  # (1, seq_len, 1024)
        logits = _BUNDLE["xlsr_aasist"](embedding)
    return _softmax_probs(logits)


def _timed(fn, *args) -> dict:
    start = time.perf_counter()
    real_prob, fake_prob = fn(*args)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return {
        "real_prob": real_prob,
        "fake_prob": fake_prob,
        "inference_ms": elapsed_ms,
    }


def predict_all_models(audio: np.ndarray, sample_rate: int) -> dict[str, dict]:
    """오디오 입력 → 4개 모델 각자의 결과를 dict로 반환.

    Returns:
        {
          "gru":          {"real_prob", "fake_prob", "inference_ms"},
          "lcnn":         {...},
          "crnn":         {...},
          "xlsr_aasist":  {...},
        }
    """
    if not _BUNDLE:
        load_all()

    mel = make_mel_spectrogram(audio, sample_rate)
    waveform = prepare_xlsr_waveform(audio, sample_rate)

    return {
        "gru": _timed(_run_gru, mel),
        "lcnn": _timed(_run_lcnn, mel),
        "crnn": _timed(_run_crnn, mel),
        "xlsr_aasist": _timed(_run_xlsr_aasist, waveform),
    }
