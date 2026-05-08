# Voice Authenticity Detector — Web Demo

실제 음성과 합성 음성을 판별하는 딥러닝 모델 시연용 웹뷰.

## 구성

```
deeplearning_web_view/
├── frontend/      # Next.js 16 + Tailwind v4 + shadcn (21st.dev 호환)
└── backend/       # FastAPI + (PyTorch 모델 자리)
```

- **프론트엔드**: WAV 업로드 UI, 결과 시각화, Dotted Surface 배경
- **백엔드**: WAV 디코딩 → 모델 추론 → `{real_prob, fake_prob}` 반환
- **현재 모델**: 결정론적 stub (해시 기반). 실제 모델 학습 후 `backend/model.py`의 `predict_authenticity` 함수만 교체.

## 로컬 개발

터미널 2개 띄우고:

```bash
# 1번 터미널 — 백엔드
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

```bash
# 2번 터미널 — 프론트엔드
cd frontend
cp .env.local.example .env.local   # NEXT_PUBLIC_API_URL=http://localhost:8000
npm install
npm run dev
```

http://localhost:3000 접속.

## 모델 교체

`backend/model.py` 상단의 docstring에 PyTorch 교체 예시가 있습니다. 시그니처만 지키면 GRU / LCNN / 파인튜닝 모델 모두 동일 인터페이스로 받습니다:

```python
def predict_authenticity(audio: np.ndarray, sample_rate: int) -> tuple[float, float]:
    # (real_prob, fake_prob) 반환
    ...
```

체크포인트 파일은 `backend/checkpoints/` 폴더(이미 .gitignore 처리)에 두는 걸 권장합니다.

## 배포 — 팀원과 공유하기

### 백엔드: Hugging Face Spaces (Docker)

PyTorch + torchaudio 같은 무거운 ML 의존성을 무료로 받아주는 가장 안정적인 옵션.

1. https://huggingface.co/new-space 에서 Space 생성 (SDK: **Docker**)
2. `backend/` 디렉터리에 `Dockerfile` 추가:
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   RUN apt-get update && apt-get install -y libsndfile1 && rm -rf /var/lib/apt/lists/*
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY . .
   EXPOSE 7860
   CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
   ```
3. HF Space는 7860 포트 강제 → `Dockerfile`에서 노출
4. Space 저장소를 backend 코드로 채우고 push
5. URL: `https://<username>-<space-name>.hf.space`
6. Space의 **Settings → Variables**에서 `ALLOWED_ORIGINS` 에 Vercel 프론트 도메인 추가

### 프론트엔드: Vercel

1. https://vercel.com/new — GitHub 저장소 연결 (또는 `vercel` CLI)
2. **Root Directory**: `frontend`
3. **Environment Variables**:
   - `NEXT_PUBLIC_API_URL` = `https://<username>-<space-name>.hf.space`
4. Deploy

배포 완료되면 `https://<project>.vercel.app` 주소를 팀원에게 공유.

### CORS 주의

- `backend/main.py`는 `*.vercel.app` 도메인은 자동 허용 (preview/production 모두)
- 커스텀 도메인 쓰면 HF Space의 `ALLOWED_ORIGINS` 환경변수에 추가

## 기술 스택

| 영역 | 기술 |
|------|------|
| 프론트 프레임워크 | Next.js 16 (App Router, Turbopack) |
| 스타일링 | Tailwind CSS v4 + shadcn/ui |
| 배경 컴포넌트 | three.js + next-themes (21st.dev Dotted Surface) |
| 백엔드 프레임워크 | FastAPI |
| 오디오 디코딩 | soundfile (libsndfile) |
| 모델 (예정) | PyTorch — GRU / LCNN / SoTA 파인튜닝 |
