from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from config import DEFAULT_CHARSET


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_channels, in_channels, 3, stride, groups=in_channels),
            ConvBNAct(in_channels, out_channels, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DBNet(nn.Module):
    """A compact DBNet-style text segmentation model.

    It predicts a single text probability/logit map. The postprocessor in
    infer.py turns connected text regions into sorted line crops.
    """

    def __init__(self, in_channels: int = 1, inner_channels: int = 64) -> None:
        super().__init__()
        self.stem = ConvBNAct(in_channels, 16, 3, 2)
        self.stage2 = DepthwiseSeparableBlock(16, 24, 2)
        self.stage3 = DepthwiseSeparableBlock(24, 40, 2)
        self.stage4 = DepthwiseSeparableBlock(40, 80, 2)

        self.lat1 = nn.Conv2d(16, inner_channels, 1)
        self.lat2 = nn.Conv2d(24, inner_channels, 1)
        self.lat3 = nn.Conv2d(40, inner_channels, 1)
        self.lat4 = nn.Conv2d(80, inner_channels, 1)

        self.fuse = nn.Sequential(
            ConvBNAct(inner_channels * 4, inner_channels, 3),
            DepthwiseSeparableBlock(inner_channels, inner_channels),
            nn.Conv2d(inner_channels, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        c1 = self.stem(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)

        target = c1.shape[-2:]
        feats = [
            self.lat1(c1),
            F.interpolate(self.lat2(c2), target, mode="bilinear", align_corners=False),
            F.interpolate(self.lat3(c3), target, mode="bilinear", align_corners=False),
            F.interpolate(self.lat4(c4), target, mode="bilinear", align_corners=False),
        ]
        logits = self.fuse(torch.cat(feats, dim=1))
        return F.interpolate(logits, input_size, mode="bilinear", align_corners=False)


class CRNNRecognizer(nn.Module):
    def __init__(
        self,
        num_classes: int | None = None,
        charset: str = DEFAULT_CHARSET,
        hidden_size: int = 128,
        lstm_layers: int = 2,
    ) -> None:
        super().__init__()
        self.charset = charset
        self.num_classes = num_classes or (len(charset) + 1)

        self.cnn = nn.Sequential(
            ConvBNAct(1, 32, 3, 1),
            nn.MaxPool2d(2, 2),
            ConvBNAct(32, 64, 3, 1),
            nn.MaxPool2d(2, 2),
            ConvBNAct(64, 128, 3, 1),
            ConvBNAct(128, 128, 3, 1),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            ConvBNAct(128, 256, 3, 1),
            ConvBNAct(256, 256, 3, 1),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            ConvBNAct(256, 256, 3, 1),
        )
        self.sequence = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=False,
            dropout=0.1 if lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size * 2, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)
        features = F.adaptive_avg_pool2d(features, (1, features.shape[-1])).squeeze(2)
        sequence = features.permute(2, 0, 1).contiguous()
        sequence, _ = self.sequence(sequence)
        return self.classifier(sequence)
