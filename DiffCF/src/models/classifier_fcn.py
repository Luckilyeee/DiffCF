import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, base_channels=128):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=8, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_channels, num_classes)

    def forward(self, x):
        assert x.ndim == 3, "Expected [B, C, T]"
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x).squeeze(-1)
        logits = self.fc(x)
        return logits

    def encode(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x).squeeze(-1)
        return x

    def predict_proba(self, x):
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

