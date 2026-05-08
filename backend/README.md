# Voice Authenticity Detector — Backend

FastAPI 기반 음성 진위 판별 API.

## 로컬 실행

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

확인: http://localhost:8000/docs

## 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| GET | `/` | 헬스 체크 |
| GET | `/health` | 헬스 체크 |
| POST | `/predict` | WAV 업로드 → `{real_prob, fake_prob, prediction, ...}` |

## 모델 교체 방법

`model.py`의 `predict_authenticity(audio, sample_rate)` 함수만 수정하면 됨.
파일 상단 docstring에 PyTorch 예시 포함.

## 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `ALLOWED_ORIGINS` | (없음) | CORS 허용 오리진을 콤마로 구분. `*.vercel.app`은 항상 허용. |
