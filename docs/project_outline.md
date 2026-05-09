# Voice Anti-Spoofing — 프로젝트 개요

실제 음성과 합성(딥페이크) 음성을 판별하는 딥러닝 데모 시스템.
사용자가 WAV 파일을 업로드하면 백엔드 모델이 추론하고, 결과를 `VOICE AUTH PROTOCOL` 톤의 사이버 보안 UI로 보여준다.

---

## 1. 시스템 구성

```
┌────────────────────────┐         POST /predict          ┌────────────────────────┐
│  Frontend              │  ───────────────────────────▶  │  Backend               │
│  Next.js 16 + React 19 │   multipart (.wav)             │  FastAPI               │
│  Tailwind v4 + Canvas  │  ◀───────────────────────────  │  PyTorch 모델          │
│  Vercel에 배포          │   { real_prob, fake_prob,      │  HF Spaces (Docker)    │
│                        │     prediction, ... }          │     에 배포             │
└────────────────────────┘                                └────────────────────────┘
```

| 영역 | 기술 | 역할 |
|---|---|---|
| Frontend | Next.js 16 (App Router, Turbopack), React 19, Tailwind v4 | UI · 파일 업로드 · 결과 시각화 |
| Backend  | FastAPI, soundfile, NumPy, PyTorch (예정) | WAV 디코딩 · 모델 추론 · JSON 응답 |
| 모델     | 팀원이 학습한 PyTorch 체크포인트 (GRU / LCNN / 파인튜닝 등) | `predict_authenticity()` 함수 뒤에 자유롭게 교체 |

---

## 2. 사용자 흐름 (UI 3단계)

```
   IDLE                 ANALYZING              RESULT
 ┌──────┐  start scan  ┌──────┐  /predict 응답  ┌──────────────┐
 │ 파동  │ ───────────▶ │ 링펄스 │ ─────────────▶ │ FAKE / REAL  │
 │ 대기  │             │ 스캔   │                │ 확률·진단 4줄 │
 └──────┘ ◀─ × reset ──┴──────┘                └──────────────┘
```

- **IDLE**: `VOICE AUTH PROTOCOL` 타이틀 + 흐르는 시안 파동. `[ SELECT .WAV TO SCAN ]` 버튼이 파일 선택 다이얼로그까지 책임진다. 백엔드 오프라인이면 `BACKEND OFFLINE — start uvicorn :8000` 안내가 표시.
- **ANALYZING**: 3겹 링 + 🔍 펄스 + 시안 파형. 백엔드 응답을 기다리는 동안 보임.
- **RESULT**:
  - `prediction: 'fake'` → 빨간 ALERT 화면 + 글리치 + scanline + `FAKE` 배지 + 진단 4줄(FAILED/SYNTHETIC DETECTED 톤)
  - `prediction: 'real'` → 시안 SAFE 화면 + `AUTHENTIC` 배지 + 진단 4줄(PASS/CLEAN 톤)
  - 우측 상단 `×` 버튼으로 IDLE 복귀

---

## 3. 데이터 플로우

1. 브라우저에서 WAV 파일 선택 → `multipart/form-data` 로 `POST {API_URL}/predict`
2. FastAPI가 `soundfile` 로 디코딩 → numpy array + sample_rate 추출
3. `backend/model.py:predict_authenticity(audio, sample_rate) -> (real_prob, fake_prob)` 호출
4. 응답 스키마:

   ```json
   {
     "real_prob": 0.83,
     "fake_prob": 0.17,
     "prediction": "real",
     "inference_ms": 142,
     "filename": "sample.wav",
     "sample_rate": 16000,
     "duration_sec": 4.21
   }
   ```

5. Frontend가 `prediction` 값에 따라 RESULT 화면을 빨강(`fake`) 또는 시안(`real`)으로 분기

별도로 `GET {API_URL}/health` 를 frontend가 15초 주기로 폴링해 IDLE 화면의 우측 하단 상태 점등을 갱신.

---

## 4. 모델 통합 (팀원이 모델 3개를 가져온 후)

핵심 인터페이스는 단 하나의 함수:

```python
# backend/model.py
def predict_authenticity(audio: np.ndarray, sample_rate: int) -> tuple[float, float]:
    """오디오 입력 → (real_prob, fake_prob) 반환."""
```

이 시그니처만 지키면 frontend는 한 줄도 바꿀 필요가 없다. 학습 완료된 모델 3개를 통합하는 일반적인 방법:

### (a) 단일 모델 — 가장 빠른 옵션
```python
_model = torch.load("checkpoints/best.pt", map_location="cpu").eval()

def predict_authenticity(audio, sr):
    x = preprocess(audio, sr)
    with torch.no_grad():
        probs = torch.softmax(_model(x), dim=-1).squeeze().tolist()
    return probs[0], probs[1]
```

### (b) 앙상블 — 3개 모델 평균
```python
_models = [torch.load(p, map_location="cpu").eval()
           for p in ["checkpoints/m1.pt", "checkpoints/m2.pt", "checkpoints/m3.pt"]]

def predict_authenticity(audio, sr):
    x = preprocess(audio, sr)
    probs_sum = torch.zeros(2)
    with torch.no_grad():
        for m in _models:
            probs_sum += torch.softmax(m(x), dim=-1).squeeze()
    p = (probs_sum / len(_models)).tolist()
    return p[0], p[1]
```

### (c) 라우팅 — 입력 특성별로 다른 모델 선택
샘플레이트나 길이에 따라 적합한 모델로 분기. 필요 시 추가.

체크포인트 파일은 `backend/checkpoints/`에 두는 것을 권장 (이미 `.gitignore` 처리됨). PyTorch 의존성은 `backend/requirements.txt`에 추가.

---

## 5. 로컬 개발

```bash
# 터미널 1 — 백엔드 (port 8000)
cd backend
source .venv/bin/activate
uvicorn main:app --reload --port 8000

# 터미널 2 — 프론트엔드 (port 3000)
cd frontend
npm install
npm run dev      # → http://localhost:3000
```

CORS 화이트리스트가 `localhost:3000`만 열려 있으므로 frontend는 반드시 3000 포트로.

---

## 6. 웹 배포

| 영역 | 플랫폼 | 이유 |
|---|---|---|
| Frontend | **Vercel** | Next.js 1-click 배포, CDN 자동 |
| Backend  | **Hugging Face Spaces (Docker)** | PyTorch 모델 호스팅에 적합. Vercel Functions는 50MB·timeout 한계로 부적합 |

### Frontend — Vercel
1. https://vercel.com/new 에서 GitHub 저장소 연결
2. **Root Directory**: `frontend`
3. **Environment Variables**: `NEXT_PUBLIC_API_URL` = HF Space의 backend URL
4. Deploy → `https://<project>.vercel.app`

### Backend — Hugging Face Spaces
1. https://huggingface.co/new-space 에서 SDK: **Docker** 선택
2. `backend/` 디렉터리에 README의 예시 Dockerfile 추가 (libsndfile1 + uvicorn :7860)
3. Space 저장소에 push → `https://<user>-<space>.hf.space`
4. Settings → Variables에서 `ALLOWED_ORIGINS`에 Vercel 도메인 추가

배포 완료 후 흐름:

```
팀원 ─▶ https://<vercel-url>           (frontend)
       └─ 브라우저가 자동으로 호출 ─▶  https://<hf-space>.hf.space/predict (backend)
                                                │
                                                └─ PyTorch 모델 추론
```

본인 컴퓨터는 꺼져있어도 동작. 모델만 새 체크포인트로 교체하면 HF Space에 push 즉시 반영.

---

## 7. 팀 작업 분담 가이드

| 담당 | 작업 | 인터페이스 |
|---|---|---|
| 모델 팀 | PyTorch로 anti-spoofing 모델 학습 → 체크포인트 산출 | `predict_authenticity(audio, sr) -> (real, fake)` 시그니처만 지키면 됨 |
| 백엔드 통합 | `backend/model.py` 교체, `requirements.txt` 갱신, 필요 시 앙상블 코드 작성 | 위 함수 1개 |
| 프론트엔드 | UI 톤·애니메이션·UX | `NEXT_PUBLIC_API_URL` env만 있으면 백엔드 어디 떠있든 동작 |
| 배포 | Vercel(frontend) + HF Spaces(backend) 1회 셋업 후 자동화 | GitHub push만으로 자동 배포 |

각 영역이 함수 시그니처와 API 스키마를 통해 디커플링되어 있으므로, 한 영역의 작업이 다른 영역을 막지 않는다.

---

## 8. 참고 파일

- `README.md` — 빠른 시작 + 배포 가이드 원본
- `backend/main.py` — FastAPI 엔드포인트 (`/health`, `/predict`)
- `backend/model.py` — 모델 인터페이스 (현재 stub)
- `frontend/src/components/upload-form.tsx` — UI 3상태(IDLE/ANALYZING/RESULT) 핵심 컴포넌트
- `frontend/src/app/globals.css` — v3 톤의 keyframes (글리치·scanline·rOut·gPulse 등)
