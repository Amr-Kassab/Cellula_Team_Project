import torch.nn as nn


class EEGCNN(nn.Module):
    def __init__(self, channels=4, classes=2, dropout=0.45):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, (1, 25), padding=(0, 12), bias=False), nn.BatchNorm2d(16), nn.ELU(),
            nn.Conv2d(16, 32, (channels, 1), bias=False), nn.BatchNorm2d(32), nn.ELU(), nn.MaxPool2d((1, 4)), nn.Dropout(dropout),
            nn.Conv2d(32, 48, (1, 15), padding=(0, 7), bias=False), nn.BatchNorm2d(48), nn.ELU(), nn.MaxPool2d((1, 4)), nn.Dropout(dropout),
            nn.AdaptiveAvgPool2d((1, 12)))
        self.classifier = nn.Linear(48 * 12, classes)
    def forward(self, x): return self.classifier(self.features(x).flatten(1))

