"""음성 진위 판별 모델 인터페이스.

현재는 stub 구현 (랜덤 확률 반환).
실제 모델 학습이 끝나면 `predict_authenticity` 함수만 교체하면 됨.

교체 시 예시 (PyTorch):
    import torch
    import torchaudio

    _model = None

    def _load_model():
        global _model
        if _model is None:
            _model = torch.load("checkpoints/best.pt", map_location="cpu")
            _model.eval()
        return _model

    def predict_authenticity(audio: np.ndarray, sample_rate: int) -> tuple[float, float]:
        model = _load_model()
        # 1) 전처리 (mel spectrogram 등)
        # 2) tensor 변환
        # 3) 추론
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=-1).squeeze().tolist()
        # (real_prob, fake_prob) 순서로 반환
        return probs[0], probs[1]
"""

from __future__ import annotations

import hashlib

import numpy as np


def predict_authenticity(audio: np.ndarray, sample_rate: int) -> tuple[float, float]:
    """오디오 입력 → (real_prob, fake_prob) 반환.

    현재 구현: 입력 해시 기반 결정론적 stub. 같은 파일은 같은 결과를 줘서
    UI 시연 중에도 일관되게 보임. 학습된 모델 준비되면 교체.
    """
    # 신호 통계 일부와 길이를 시드로 사용 → 동일 파일 동일 결과
    digest = hashlib.sha256(audio.tobytes()[:8192]).digest()
    seed = int.from_bytes(digest[:4], "big")
    rng = np.random.default_rng(seed)

    real_prob = float(rng.beta(2.0, 2.0))
    fake_prob = 1.0 - real_prob
    return real_prob, fake_prob
