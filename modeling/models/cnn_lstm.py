import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    """CNN output [B, features, 1, time] becomes LSTM sequence [B, time, features]."""
    def __init__(self, channels=4, classes=2, dropout=0.4):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(1, 16, (1, 25), padding=(0, 12), bias=False), nn.BatchNorm2d(16), nn.ELU(),
                                 nn.Conv2d(16, 32, (channels, 1), bias=False), nn.BatchNorm2d(32), nn.ELU(),
                                 nn.MaxPool2d((1, 5)), nn.Dropout(dropout))
        self.lstm = nn.LSTM(32, 48, batch_first=True, dropout=dropout, num_layers=2)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(48, classes))
    def forward(self, x):
        z = self.cnn(x).squeeze(2).transpose(1, 2)  # [batch, temporal sequence, 32]
        _, (hidden, _) = self.lstm(z)
        return self.classifier(hidden[-1])

