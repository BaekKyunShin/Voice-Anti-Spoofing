"""ipynb에서 추출한 4개 음성 진위 판별 모델 아키텍처.

ipynb 셀 58/60/62/95 원본 구조를 그대로 보존한다 (state_dict 로딩 호환성).
AASIST는 forward 시그니처를 정리해서 외부 모듈(xlsr, feature_extractor) 의존성
대신 임베딩 텐서를 받도록 변경 — wrapper(model.py)에서 임베딩 추출 후 호출한다.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GRUClassifier(nn.Module):
    """Bi-GRU 시계열 분류기. 입력 (B, T=400, n_mels=80)."""

    def __init__(
        self,
        input_size: int = 80,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        last_output = output[:, -1, :]
        return self.classifier(last_output)


class MaxFeatureMap2D(nn.Module):
    """LCNN의 핵심 연산: 채널을 절반으로 줄이며 max만 남김."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.chunk(x, 2, dim=1)
        return torch.max(out[0], out[1])


class LCNNClassifier(nn.Module):
    """Light CNN. 입력 (B, 1, 80, 400)."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            MaxFeatureMap2D(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1),
            MaxFeatureMap2D(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1),
            MaxFeatureMap2D(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            MaxFeatureMap2D(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class CRNNClassifier(nn.Module):
    """CNN + Bi-GRU. 입력 (B, 1, 80, 400)."""

    def __init__(
        self,
        num_classes: int = 2,
        gru_hidden: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        # CNN 3번 MaxPool 후: freq 80→10, channel 128 → GRU input = 128*10 = 1280
        self.gru = nn.GRU(
            input_size=128 * 10,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        b, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        output, _ = self.gru(x)
        last_output = output[:, -1, :]
        return self.classifier(last_output)


class AASISTEmbeddingsClassifier(nn.Module):
    """XLS-R 임베딩(1024-dim) → spoofing 분류.

    ipynb 원본 forward는 (raw_audio, xlsr_model, feature_extractor)를 받지만
    여기서는 임베딩 텐서만 받도록 단순화. XLS-R 추론은 wrapper가 처리.

    state_dict 호환: ipynb에서 저장된 conv_block/gru/classifier 파라미터명을 그대로 유지.
    """

    def __init__(
        self,
        xlsr_model_dimension: int = 1024,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(xlsr_model_dimension, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.gru = nn.GRU(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, xlsr_embedding: torch.Tensor) -> torch.Tensor:
        """xlsr_embedding shape: (B, seq_len, 1024)."""
        x = xlsr_embedding.permute(0, 2, 1)  # (B, 1024, seq_len) for Conv1d
        x = self.conv_block(x)
        x = x.permute(0, 2, 1)  # (B, seq_len', 256) for GRU
        output, _ = self.gru(x)
        last_output = output[:, -1, :]
        return self.classifier(last_output)
