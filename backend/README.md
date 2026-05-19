---
title: Voice Anti-Spoofing Detector
emoji: 🎙️
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
pinned: false
short_description: 4-모델 음성 진위 판별 데모 (GRU/LCNN/CRNN/XLS-R+AASIST)
---

# Voice Authenticity Detector — Backend

FastAPI 기반 음성 진위 판별 API. WAV 1개를 4개 모델(GRU, LCNN, CRNN, XLS-R+AASIST)로 동시 판별.

> 이 README의 상단 YAML는 Hugging Face Spaces 메타데이터다. 로컬 실행에는 영향 없음.

## 로컬 실행

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

확인: http://localhost:8000/docs

`backend/models/weights/`에 `.pt` 4개(`GRU_best.pt`, `LCNN_best.pt`, `CRNN_best.pt`, `XLSR_AASIST_best.pt`)가 있어야 부팅 성공.

## 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| GET | `/` | 헬스 체크 |
| GET | `/health` | 헬스 체크 |
| POST | `/predict` | WAV 업로드 → 4-모델 결과 + consensus |

응답 스키마:

```json
{
  "filename": "sample.wav",
  "sample_rate": 16000,
  "duration_sec": 4.21,
  "total_inference_ms": 365.5,
  "models": {
    "gru":         {"real_prob": 0.83, "fake_prob": 0.17, "prediction": "real", "inference_ms": 12.3},
    "lcnn":        {"real_prob": 0.91, "fake_prob": 0.09, "prediction": "real", "inference_ms": 18.7},
    "crnn":        {"real_prob": 0.78, "fake_prob": 0.22, "prediction": "real", "inference_ms": 22.1},
    "xlsr_aasist": {"real_prob": 0.95, "fake_prob": 0.05, "prediction": "real", "inference_ms": 312.4}
  },
  "consensus": {
    "prediction": "real",
    "agreement": 1.0
  }
}
```

## 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `ALLOWED_ORIGINS` | (없음) | CORS 허용 오리진을 콤마로 구분. `*.vercel.app`은 항상 허용. |
| `HF_HOME` | `~/.cache/huggingface` | XLS-R 베이스 모델 캐시 경로 |

## 테스트

```bash
.venv/bin/pytest tests/ -v
```
