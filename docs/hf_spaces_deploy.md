# Hugging Face Spaces 배포 가이드

`backend/` 디렉터리를 HF Spaces(Docker SDK)에 배포해서 무료 공개 URL을 얻는 절차다.

---

## 0. 사전 확인

- 가중치 4개 합계 ~13MB → **Git LFS 불필요**, 직접 push 가능
- XLS-R 베이스 모델 ~1.2GB는 Dockerfile에서 빌드 시 다운로드 → repo에 포함 안 함
- HF Spaces 무료 티어: CPU Basic, 16GB RAM, 7860 포트 강제 (Dockerfile에 이미 반영)

---

## 1. HF 계정 & Space 생성 (사용자 직접)

1. https://huggingface.co 가입/로그인
2. https://huggingface.co/new-space 접속
3. 폼 입력:
   - **Space name**: 예 `voice-spoofing-detector` (소문자/하이픈, 결정 후 바꾸기 어려움)
   - **License**: MIT 등 자유
   - **SDK**: `Docker`
   - **Docker template**: `Blank`
   - **Space hardware**: `CPU basic · Free`
   - **Visibility**: `Public` (Vercel에서 호출하려면 public이 편함)
4. Create

생성 후 Space URL: `https://huggingface.co/spaces/<USER>/<SPACE>`
배포된 API URL: `https://<USER>-<SPACE>.hf.space`

---

## 2. Access Token 발급 (사용자 직접)

1. https://huggingface.co/settings/tokens 접속
2. "Create new token" → **Write** 권한 선택
3. 토큰 복사해서 안전한 곳에 보관 (한 번만 표시됨)

토큰을 이 채팅에 붙여넣지 마세요. push 시 사용자 셸에서만 입력.

---

## 3. 푸시 전략 결정

`backend/`만 HF Space repo에 올려야 한다 (HF는 root에 Dockerfile/main.py 기대).

**전략 A: 별도 폴더 clone + 복사 + push (권장, 가장 직관적)**

```bash
# 작업 폴더에서 (예: ~/Desktop)
# HF Space repo를 별도로 clone
git clone https://huggingface.co/spaces/<USER>/<SPACE> hf-space
cd hf-space

# 백엔드 내용 통째로 복사 (가중치 포함)
rsync -av --exclude '.venv' --exclude '__pycache__' --exclude '.pytest_cache' \
  /Users/baekkyunshin/Desktop/deeplearning_web_view/backend/ ./

# .gitignore에 *.pt가 있으면 제거 (HF에는 가중치 같이 올림)
sed -i.bak '/\*\.pt/d' .gitignore && rm .gitignore.bak

# 첫 커밋
git add .
git commit -m "init: 4-모델 음성 진위 판별 백엔드"

# push (token 프롬프트 뜨면 user name + access token 입력)
git push
```

**전략 B: subtree push (한 줄 명령, 약간 어려움)**

```bash
# 메인 repo 루트에서
git remote add hf https://huggingface.co/spaces/<USER>/<SPACE>
git subtree push --prefix=backend hf main
```

단점: 가중치가 `.gitignore`에 잡혀있어서 메인 repo에 안 들어가 있음 → subtree에도 안 들어감.
→ 가중치 별도 처리 필요.

→ **전략 A 추천.**

---

## 4. 빌드 & 첫 동작 확인

push 후 Space 페이지에서:
- "Building" → "Running" (약 5~10분, XLS-R 1.2GB 다운로드 + 의존성 설치 포함)
- 로그 탭에서 진행 상황 확인 가능

확인:
```bash
curl https://<USER>-<SPACE>.hf.space/health
# → {"status":"healthy"}
```

---

## 5. CORS 설정 (Vercel 도메인 허용)

HF Space 페이지 → **Settings → Variables and secrets** → "New variable":
- Name: `ALLOWED_ORIGINS`
- Value: `https://<YOUR-VERCEL-DOMAIN>.vercel.app` (콤마로 여러 개 가능)

저장하면 자동 재시작. `*.vercel.app`은 코드에서 항상 허용하므로 vercel 프리뷰 URL은 따로 추가 안 해도 됨.

---

## 6. Vercel 프론트엔드 환경변수 갱신

Vercel 프로젝트 → Settings → Environment Variables:
- `NEXT_PUBLIC_API_URL` = `https://<USER>-<SPACE>.hf.space`

Redeploy하면 프론트가 HF Spaces 백엔드를 바라봄.

---

## 7. 시연 직전 워밍업

HF Spaces 무료티어는 48시간 미사용 시 sleep. 시연 10분 전:
```bash
curl https://<USER>-<SPACE>.hf.space/health
```
또는 그냥 Vercel 페이지 한 번 열어두면 됨 (프론트가 15초마다 health ping).

첫 추론은 XLS-R 로딩 때문에 30초 ~ 1분 정도 걸릴 수 있음 (콜드 스타트).

---

## 트러블슈팅

| 증상 | 원인/해결 |
|---|---|
| 빌드 중 OOM | 무료 티어 빌드 메모리 한도. Dockerfile에서 `pip install --no-cache-dir` 유지 |
| `xlsr_aasist` 추론 매우 느림 (수 초) | CPU only라 정상. 시연 시나리오에 반영 |
| CORS 차단 | Space `ALLOWED_ORIGINS` 변수 확인 |
| 첫 호출 30초+ | 콜드 스타트. 시연 전 워밍업 ping |
| `.pt 파일이 없습니다` 에러 | `git ls-files | grep .pt` 로 HF push에 가중치 포함됐는지 확인 |
