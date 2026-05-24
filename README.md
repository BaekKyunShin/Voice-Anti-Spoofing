# Voice Authenticity Detector — Web Demo

실제 음성과 합성(딥페이크) 음성을 **4개 모델로 동시 판별**하는 데모.
WAV를 업로드하면 GRU / LCNN / CRNN / XLS-R+AASIST가 각자 추론하고, 모델별 결과 + 다수결 consensus를 카드 4장으로 보여준다.

## 🌐 라이브 데모

| | URL |
|---|---|
| **프론트엔드 (Vercel)** | https://voice-anti-spoofing.vercel.app |
| **백엔드 API (HF Spaces)** | https://baekkyun-voice-spoofing-detector.hf.space |

> HF Spaces 무료 티어는 48시간 미사용 시 sleep. 시연 10분 전에 위 프론트 URL 한 번 열어두면 health ping이 자동으로 컨테이너를 깨움.

## 구성

```
deeplearning_web_view/
├── frontend/                       # Next.js 16 + React 19 + Tailwind v4
│   └── src/components/upload-form.tsx   # 업로드 + 4-카드 결과 그리드
├── backend/                        # FastAPI + PyTorch
│   ├── main.py                     # /predict 엔드포인트
│   ├── model.py                    # predict_all_models() — 4-모델 통합 추론
│   ├── models/
│   │   ├── architectures.py        # 4개 nn.Module 클래스
│   │   ├── preprocess.py           # mel-spec + XLS-R raw waveform
│   │   └── weights/                # .pt 4개 (gitignored)
│   └── tests/test_predict_smoke.py
└── docs/
    ├── project_outline.md          # 전체 설계 문서
    └── hf_spaces_deploy.md         # HF Spaces 배포 가이드 (7단계)
```

| 모델 | 입력 | 특징 |
|---|---|---|
| GRU | mel (80, 400) | Bi-GRU 시계열 |
| LCNN | mel (80, 400) | Light CNN + MaxFeatureMap |
| CRNN | mel (80, 400) | CNN + Bi-GRU 결합 |
| XLS-R + AASIST | raw 5초 16kHz | facebook/wav2vec2-xls-r-300m 임베딩 → AASIST 분류 |

## 응답 스키마 (`POST /predict`)

```json
{
  "filename": "sample.wav",
  "sample_rate": 16000,
  "duration_sec": 4.21,
  "total_inference_ms": 365.5,
  "models": {
    "gru":         {"bonafide_prob": 0.83, "prediction": "bonafide", "inference_ms": 12.3},
    "lcnn":        {"bonafide_prob": 0.91, "prediction": "bonafide", "inference_ms": 18.7},
    "crnn":        {"bonafide_prob": 0.78, "prediction": "bonafide", "inference_ms": 22.1},
    "xlsr_aasist": {"bonafide_prob": 0.95, "prediction": "bonafide", "inference_ms": 312.4}
  },
  "consensus": {"prediction": "bonafide", "agreement": 1.0}
}
```

`bonafide_prob` — ASVspoof 표준 CM score. **높을수록 진짜(bonafide), 낮을수록 합성(spoof)**. 판정 임계값은 `0.5` 고정 (`backend/main.py:BONAFIDE_THRESHOLD`, 추후 EER 기반 튜닝 여지).

`consensus.agreement` 는 4-모델 다수결 일치 비율 (1.0 = 4/4, 0.75 = 3/4 등).

## 로컬 개발

```bash
# 1번 터미널 — 백엔드 (port 8000)
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# backend/models/weights/ 에 .pt 4개가 있어야 부팅 성공
uvicorn main:app --reload --port 8000
```

```bash
# 2번 터미널 — 프론트엔드 (port 3000)
cd frontend
cp .env.local.example .env.local   # NEXT_PUBLIC_API_URL=http://localhost:8000
npm install
npm run dev
```

http://localhost:3000 접속.

테스트:
```bash
cd backend && .venv/bin/pytest tests/ -v
```

## 배포

자세한 절차는 [`docs/hf_spaces_deploy.md`](docs/hf_spaces_deploy.md) 참고. 요약:

- **백엔드** — HF Spaces (Docker, CPU Basic 무료) — Dockerfile에서 XLS-R 베이스 ~1.2GB를 빌드 시 사전 다운로드
- **프론트엔드** — Vercel — Root Directory `frontend`, 환경변수 `NEXT_PUBLIC_API_URL` 만 설정
- **CORS** — `backend/main.py`가 `*.vercel.app` 도메인은 자동 허용. 커스텀 도메인은 HF Space의 `ALLOWED_ORIGINS` 변수에 추가

## 기술 스택

| 영역 | 기술 |
|------|------|
| 프론트 | Next.js 16 (App Router, Turbopack), React 19, Tailwind v4 |
| 백엔드 | FastAPI, soundfile, librosa, NumPy |
| 모델 | PyTorch 2.10 (CPU), transformers 5.0 (XLS-R) |
| 배포 | Vercel (frontend), Hugging Face Spaces Docker (backend) |
