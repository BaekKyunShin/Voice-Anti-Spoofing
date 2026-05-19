FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CPU wheel을 별도 인덱스에서 받아 이미지 크기 절감.
COPY requirements.txt .
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# XLS-R 베이스 모델(~1.2GB)을 빌드 시 미리 받아 콜드 스타트 단축.
# HF Spaces에서는 HF_HOME=/data/.huggingface 같이 설정해 캐시 디렉토리를 영구 볼륨으로 옮길 수도 있음.
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model; \
    Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-xls-r-300m'); \
    Wav2Vec2Model.from_pretrained('facebook/wav2vec2-xls-r-300m')"

COPY . .

# Hugging Face Spaces는 7860 포트 강제. 로컬에선 -p 8000:7860 으로 매핑하면 됨.
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
