import torch
import torch.nn as nn


class EEGTransformer(nn.Module):
    def __init__(self, channels=4, classes=2, embed_dim=48, patch_size=25, layers=2, heads=4, dropout=0.35):
        super().__init__()
        self.patch_embed = nn.Conv1d(channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.position = nn.Parameter(torch.zeros(1, 30, embed_dim))  # floor((751 - 25) / 25) + 1 patches
        block = nn.TransformerEncoderLayer(embed_dim, heads, dim_feedforward=96, dropout=dropout, batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Dropout(dropout), nn.Linear(embed_dim, classes))
    def forward(self, x):
        z = self.patch_embed(x.squeeze(1)).transpose(1, 2)
        z = z + self.position[:, :z.size(1)]
        return self.head(self.encoder(z).mean(dim=1))
