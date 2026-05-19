"""ipynb의 전처리 파이프라인을 백엔드용으로 이식.

ipynb는 librosa.load(wav_path)로 파일에서 직접 읽지만,
백엔드는 FastAPI가 이미 디코딩한 np.ndarray + sample_rate를 받는다.
따라서 (1) 리샘플링을 명시적으로 수행, (2) mel 변환은 동일 파라미터 유지.
"""

from __future__ import annotations

import librosa
import numpy as np

# Mel 입력 모델 (GRU/LCNN/CRNN) 공용 상수 — ipynb 셀 36과 동일.
SAMPLE_RATE = 16000
N_MELS = 80
N_FFT = 1024
HOP_LENGTH = 256
MAX_LENGTH = 400  # time frame 개수 고정

# XLS-R+AASIST 전용 — ipynb 셀 93과 동일.
XLSR_SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 5
MAX_AUDIO_LENGTH = XLSR_SAMPLE_RATE * MAX_AUDIO_SECONDS  # 80,000


def resample_to(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """sr이 target_sr과 다르면 리샘플링."""
    if sr == target_sr:
        return audio
    return librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)


def make_mel_spectrogram(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """wav 신호 → mel-spectrogram (80, 400) float32.

    ipynb 셀 38의 make_mel_spectrogram을 그대로 재현. 단,
    librosa.load 단계를 분리해서 호출 측에서 audio를 넘기도록 변경.
    """
    audio = resample_to(audio, sample_rate, SAMPLE_RATE)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    if mel_db.shape[1] < MAX_LENGTH:
        pad_width = MAX_LENGTH - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mel_db = mel_db[:, :MAX_LENGTH]

    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return mel_db.astype(np.float32)


def prepare_xlsr_waveform(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """raw waveform → XLS-R 입력용 (MAX_AUDIO_LENGTH,) float32.

    ipynb 셀 93의 VoiceXLSRDataset.__getitem__ 로직 재현:
    - 16kHz 리샘플링
    - 5초로 자르거나 zero-padding (뒤쪽)
    """
    audio = resample_to(audio, sample_rate, XLSR_SAMPLE_RATE).astype(np.float32)

    if audio.shape[0] > MAX_AUDIO_LENGTH:
        audio = audio[:MAX_AUDIO_LENGTH]
    else:
        pad = MAX_AUDIO_LENGTH - audio.shape[0]
        audio = np.pad(audio, (0, pad), mode="constant")
    return audio
