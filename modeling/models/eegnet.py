import torch.nn as nn


class EEGNet(nn.Module):
    """Compact EEGNet: temporal, depthwise spatial, then separable convolutions."""
    def __init__(self, channels=4, classes=2, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, (1, 64), padding=(0, 32), bias=False), nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (channels, 1), groups=8, bias=False), nn.BatchNorm2d(16), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
            nn.Conv2d(16, 16, (1, 16), padding=(0, 8), groups=16, bias=False),
            nn.Conv2d(16, 16, 1, bias=False), nn.BatchNorm2d(16), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(dropout), nn.AdaptiveAvgPool2d((1, 8)))
        self.classifier = nn.Linear(16 * 8, classes)
    def forward(self, x): return self.classifier(self.features(x).flatten(1))

