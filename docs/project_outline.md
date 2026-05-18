# Voice Anti-Spoofing — 프로젝트 개요 (4-모델 확장판)

실제 음성 vs 합성(딥페이크) 음성을 **4개 모델로 동시 판별**하는 데모.
사용자가 WAV를 업로드하면 백엔드가 4개 모델로 추론하고, 모델별 결과를 카드 4장으로 보여준다.

> 이 문서는 미래의 Claude Code가 작업을 이어받기 위한 참고서다. 결정 사항, 인계 자료 명세, 작업 순서, 함정을 모두 담는다.

---

## 1. 최종 시스템 구성

```
┌─────────────────────────┐      POST /predict        ┌──────────────────────────┐
│  Frontend               │  ───────────────────────▶ │  Backend                 │
│  Next.js 16 + React 19  │   multipart (.wav)        │  FastAPI + PyTorch       │
│  Tailwind v4 + Canvas   │  ◀───────────────────────  │  4개 모델 로드           │
│  Vercel 배포            │   { models: {gru, lcnn,   │  HF Spaces (Docker) 배포 │
│                         │      crnn, xlsr_aasist} } │  무료 티어 (CPU)         │
└─────────────────────────┘                           └──────────────────────────┘
```

| 영역 | 기술 | 역할 |
|---|---|---|
| Frontend | Next.js 16 (App Router, Turbopack), React 19, Tailwind v4 | UI · 파일 업로드 · 4개 결과 카드 시각화 |
| Backend  | FastAPI, soundfile, NumPy, PyTorch | WAV 디코딩 · 4-모델 추론 · JSON 응답 |
| 모델     | GRU, LCNN, CRNN, XLS-R + AASIST (팀원 학습) | 각자 독립 추론 후 결과 통합 |

---

## 2. 팀원에게서 받을 인계 자료 (필수)

### 🔴 필수 (이게 있어야 작업 시작 가능)

1. **`.pt` 파일 4개** — 각 모델의 학습된 가중치 (`state_dict` 형태 권장)
   - `gru.pt`, `lcnn.pt`, `crnn.pt`, `xlsr_aasist.pt`
2. **`.ipynb` 원본** — 4개 모델 클래스 정의 + 전처리 + 추론 코드 전부 포함
   - **클린 커널에서 Run All이 통과되는 상태**여야 함
3. **`requirements.txt`** — `pip freeze` 결과 (전체 환경 dump)

### 🟡 있으면 디버깅 시간 크게 단축

- **테스트용 WAV 1~2개** + **그 ipynb에서 나온 모델별 예상 확률값**
  → 백엔드로 이식 후 결과 일치 여부 검증용. 없으면 "내가 옮긴 게 맞나?" 확인 불가.
- **실행 환경 정보**: Colab/로컬/Kaggle, Python 버전, CUDA 여부

### 인계 자료를 받으면 가장 먼저 할 것

```bash
# 1) requirements.txt 받아보고 핵심 패키지 버전 확인
grep -iE "torch|torchaudio|transformers|librosa|s3prl|numpy" requirements.txt

# 2) .pt 파일 크기 확인 (Git LFS 필요 여부 판단)
ls -lh *.pt

# 3) ipynb 한 번 훑기 → 모델 클래스명 / 전처리 함수명 메모
```

---

## 3. 백엔드 이식 절차 (ipynb → backend/)

### 3.1 목표 디렉터리 구조

```
backend/
├── main.py                       # FastAPI 엔드포인트 (응답 스키마 확장)
├── model.py                      # 4-모델 로더 + 통합 추론 함수
├── models/
│   ├── __init__.py
│   ├── architectures.py          # ipynb에서 추출한 nn.Module 클래스 4개
│   ├── preprocess.py             # 모델별 전처리 함수 (mel/LFCC/raw 등)
│   └── weights/                  # ← .pt 파일 4개. .gitignore 대상.
│       ├── gru.pt
│       ├── lcnn.pt
│       ├── crnn.pt
│       └── xlsr_aasist.pt
├── requirements.txt              # 정리된 의존성
└── Dockerfile                    # HF Spaces용 (libsndfile1, uvicorn :7860)
```

### 3.2 ipynb에서 추출할 것 (체크리스트)

- [ ] 4개 모델의 `nn.Module` **클래스 정의** → `models/architectures.py`
- [ ] 4개 모델의 **전처리 함수** (sample_rate, n_mels, hop_length, max_len, 정규화) → `models/preprocess.py`
- [ ] 4개 모델의 **추론 로직** (입력 텐서 shape, eval 모드, softmax/sigmoid, 클래스 매핑) → `model.py`
- [ ] **상수**: `SAMPLE_RATE`, `MAX_LEN`, 라벨 매핑(real=0/fake=1 등) → `models/preprocess.py` 상단

### 3.3 통합 추론 함수 시그니처 (확장)

```python
# backend/model.py
def predict_all_models(audio: np.ndarray, sample_rate: int) -> dict[str, dict]:
    """오디오 입력 → 4개 모델 각자의 결과를 dict로 반환.

    Returns:
        {
          "gru":          {"real_prob": 0.83, "fake_prob": 0.17, "inference_ms": 12.3},
          "lcnn":         {"real_prob": 0.91, "fake_prob": 0.09, "inference_ms": 18.7},
          "crnn":         {"real_prob": 0.78, "fake_prob": 0.22, "inference_ms": 22.1},
          "xlsr_aasist":  {"real_prob": 0.95, "fake_prob": 0.05, "inference_ms": 312.4},
        }
    """
```

### 3.4 모델 로딩 패턴 (RAM에 1회만)

```python
import torch
from models.architectures import GRUModel, LCNNModel, CRNNModel, XLSRAasistModel

_MODELS: dict[str, torch.nn.Module] = {}

def _load_all() -> None:
    """서버 부팅 시 1회만 호출. 4개 모델을 RAM에 상주."""
    global _MODELS
    if _MODELS:
        return
    _MODELS["gru"]         = _init(GRUModel,         "models/weights/gru.pt")
    _MODELS["lcnn"]        = _init(LCNNModel,        "models/weights/lcnn.pt")
    _MODELS["crnn"]        = _init(CRNNModel,        "models/weights/crnn.pt")
    _MODELS["xlsr_aasist"] = _init(XLSRAasistModel,  "models/weights/xlsr_aasist.pt")

def _init(cls, ckpt_path: str) -> torch.nn.Module:
    model = cls()
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model
```

---

## 4. API 응답 스키마 (변경)

### 4.1 새 응답 형식

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

- `consensus.prediction`: 다수결 (4개 중 다수가 real이면 real)
- `consensus.agreement`: 동의 비율 (`1.0` = 4/4 일치, `0.5` = 2/2)

### 4.2 기존 단일 응답 호환

필요 없으면 제거. 프론트는 새 스키마로 일괄 전환.

---

## 5. 프론트엔드 변경 (현재 구조 유지)

### 5.1 변경 포인트만

- **RESULT 화면**: 단일 카드 → **4-모델 카드 그리드** (2x2 또는 4단)
  - 카드별: 모델명, real/fake 확률 바, prediction 배지(REAL/FAKE), inference_ms
- **상단 요약 배너**: `consensus` 결과를 크게 표시 (예: `4/4 MODELS AGREE: AUTHENTIC`)
- **fake/real 분기 색감 유지**: consensus 기준으로 빨강(fake) / 시안(real) 톤 적용
- **각 카드는 개별로도 fake/real 색감 적용** → 모델별 disagreement 시각적 강조

### 5.2 건드릴 파일

- `frontend/src/components/upload-form.tsx` — RESULT 섹션 4-카드 그리드 추가
- `frontend/src/app/globals.css` — 카드용 keyframes/스타일 (기존 톤 재활용)
- API 응답 타입 정의 — 위 4.1 스키마 반영

### 5.3 건드리지 않음

- IDLE / ANALYZING 화면, 톤, Canvas 파동 등 — **현재 그대로 유지**
- `/health` 폴링 — 그대로

---

## 6. 배포

### 6.1 백엔드 — Hugging Face Spaces (무료)

- **SDK**: Docker
- **사양**: Free CPU Basic (2 vCPU, 16GB RAM)
- **URL**: `https://<user>-<space>.hf.space`

#### 모델 파일 크기 처리 (중요)

XLS-R+AASIST는 대개 **300MB+**. HF Spaces git 100MB 제한에 걸림.

**옵션 A — Git LFS** (간단)
```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes models/weights/*.pt
```

**옵션 B — HF Hub 별도 저장소 + 부팅 시 다운로드** (권장, 코드 저장소 가벼움)
```python
# Dockerfile 또는 model.py 초기화에서
from huggingface_hub import snapshot_download
snapshot_download(repo_id="<user>/voice-spoofing-weights", local_dir="models/weights")
```

#### Dockerfile 핵심
```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y libsndfile1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
```

#### Space Settings → Variables
- `ALLOWED_ORIGINS`: Vercel 도메인 추가 (예: `https://your-app.vercel.app`)

### 6.2 프론트엔드 — Vercel (무료)

- **Root Directory**: `frontend`
- **Env**: `NEXT_PUBLIC_API_URL` = `https://<user>-<space>.hf.space`
- GitHub push → 자동 배포

---

## 7. HF Spaces 무료 티어 함정

| 함정 | 대응 |
|---|---|
| **콜드 스타트** (48시간 미사용 시 sleep) | 시연 10분 전 워밍업 요청 1회 전송 |
| **CPU only, XLS-R 느림** (~수 초) | 4개 모델 동시 추론을 `asyncio.to_thread` 등으로 병렬화 |
| **저장소 크기 제한** | 모델은 HF Hub 별도 repo로 분리 (위 옵션 B) |
| **RAM 16GB 한도** | 4개 동시 로드 시 모니터링. XLS-R이 가장 큼. 부족하면 lazy load |
| **무거운 빌드 시간** | Dockerfile 레이어 캐싱 (requirements.txt 먼저 복사) |

---

## 8. 로컬 개발

```bash
# 터미널 1 — 백엔드 (port 8000)
cd backend
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# 터미널 2 — 프론트엔드 (port 3000)
cd frontend
npm install
npm run dev      # → http://localhost:3000
```

CORS 화이트리스트가 `localhost:3000`만 열려 있으므로 frontend는 반드시 3000 포트로.

`backend/models/weights/` 안에 `.pt` 4개 넣어둬야 모델 로드 성공.

---

## 9. 작업 순서 (Claude Code가 이어받을 시점에)

인계 자료를 받은 시점에서 진행할 순서:

1. **인계 자료 수령 확인** — `.pt` 4개, `.ipynb`, `requirements.txt` 모두 있는지
2. **ipynb 분석** — 모델 클래스명, 전처리 함수명, 입력 shape, 라벨 매핑 메모
3. **`backend/models/architectures.py` 작성** — 4개 nn.Module 클래스 이식
4. **`backend/models/preprocess.py` 작성** — 4개 전처리 함수 이식
5. **`backend/model.py` 재작성** — `predict_all_models()` 구현
6. **`backend/main.py` 응답 스키마 변경** — 위 §4.1 형식
7. **TDD로 검증** — 인계받은 테스트 wav로 ipynb 결과와 일치 확인
8. **`backend/requirements.txt` 정리** — pip freeze에서 핵심만 추출
9. **`Dockerfile` 작성/검증** — HF Spaces 배포 가능 상태
10. **프론트엔드 변경** — RESULT 화면 4-카드 그리드 추가 (§5.1)
11. **로컬 통합 테스트** — wav 업로드 → 4-카드 정상 표시
12. **HF Spaces 배포** — Git LFS 또는 HF Hub 방식 결정 후 push
13. **Vercel 환경변수 갱신 + 재배포**
14. **시연 워밍업 스크립트 준비** (§10)

---

## 10. 시연 체크리스트

- [ ] 시연 10분 전 `https://your-app.vercel.app` 접속해서 HF Spaces sleep 깨우기
- [ ] **테스트 wav 준비**: real 1개 + fake 1개 + 모델별 결과 갈리는 케이스 1개 (USB 백업)
- [ ] **모바일 UI 확인**: 발표장 청중이 폰으로 접속 시 4-카드 그리드 깨지지 않는지
- [ ] **HF Spaces 로그 다른 탭에 열어두기**: 실시간 모니터링
- [ ] **백업 플랜**: HF Spaces가 죽으면 로컬 백엔드로 즉시 스위치할 수 있게 `NEXT_PUBLIC_API_URL` override 준비
- [ ] **QR 코드/단축 URL**: 발표 슬라이드에 URL 또는 QR

### 시연 시나리오 추천

1. 진짜 음성 wav → 4개 모델 모두 REAL 높게 → "consensus AUTHENTIC" 그린 톤
2. 합성 음성 wav → 4개 모델 모두 FAKE 높게 → "consensus FAKE" 레드 톤
3. **모델별 결과 갈리는 wav** → "이래서 다중 모델/앙상블이 필요하다" 인사이트 강조

---

## 11. 참고 파일 (인계 후 변경되는 파일)

- `README.md` — 빠른 시작 + 배포 가이드 (응답 스키마 변경 반영 필요)
- `backend/main.py` — FastAPI 엔드포인트 (응답 스키마 확장)
- `backend/model.py` — `predict_all_models()` 함수
- `backend/models/architectures.py` — 4개 nn.Module 클래스 (신규)
- `backend/models/preprocess.py` — 4개 전처리 함수 (신규)
- `backend/Dockerfile` — HF Spaces 배포용 (신규 또는 갱신)
- `frontend/src/components/upload-form.tsx` — RESULT 4-카드 그리드 추가
- `frontend/src/app/globals.css` — 카드용 keyframes 추가

---

## 12. 의사결정 메모 (Claude Code가 헷갈리지 않게)

- **앙상블 안 함**: 4개 모델 결과를 그대로 보여줌 (consensus는 다수결로만 요약). 평균/가중치 합 안 함. "모델별 차이를 보여주는 것"이 데모 목적.
- **단일 모델 fallback 없음**: 4개 다 로드 못 하면 부팅 실패. 시연용이라 graceful degradation 불필요.
- **GPU 안 씀**: 무료 티어가 목표. XLS-R 느려도 감수.
- **WAV만 지원**: mp3 등 다른 포맷 변환 안 함. 시연용 wav만 준비.
- **인증/레이트리밋 없음**: 공개 데모. 트래픽 문제 생기면 그때 대응.
