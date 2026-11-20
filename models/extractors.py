

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


class SensorLSTM(nn.Module):

    def __init__(self, in_channels=3, hidden=128, layers=2, out_dim=128, dropout=cfg.DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(in_channels, hidden, layers, batch_first=True,
                            bidirectional=False, dropout=dropout if layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, out_dim)
        self.drop = nn.Dropout(dropout)
        self.out_dim = out_dim

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(self.drop(out))


class MultiScale1DCNN(nn.Module):


    def __init__(self, in_channels=1, base_ch=16, kernels=(3, 5, 7, 9), out_dim=128,
                 dropout=cfg.DROPOUT):
        super().__init__()
        self.convs = nn.ModuleList()
        for k in kernels:
            self.convs.append(nn.Sequential(
                nn.Conv1d(in_channels, base_ch, k, padding=k // 2),
                nn.InstanceNorm1d(base_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(4),
                nn.Conv1d(base_ch, base_ch * 2, k, padding=k // 2),
                nn.InstanceNorm1d(base_ch * 2),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(8),
            ))
        total_ch = base_ch * 2 * len(kernels)
        self.fc = nn.Linear(total_ch * 8, out_dim)
        self.drop = nn.Dropout(dropout)
        self.out_dim = out_dim

    def forward(self, x):
        feats = []
        for conv in self.convs:
            f = conv(x)
            f = f.view(f.size(0), -1)
            feats.append(f)
        fused = torch.cat(feats, dim=1)
        return self.fc(self.drop(fused))


class Acoustic1DCNN(nn.Module):

    def __init__(self, in_channels=1, base_ch=16, out_dim=128, dropout=cfg.DROPOUT):
        super().__init__()
        self.blocks = nn.Sequential(
            self._block(in_channels, base_ch),
            self._block(base_ch, base_ch * 2),
            self._block(base_ch * 2, base_ch * 4),
            self._block(base_ch * 4, base_ch * 8),
        )
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.fc = nn.Linear(base_ch * 8 * 8, out_dim)
        self.drop = nn.Dropout(dropout)
        self.out_dim = out_dim

    def _block(self, cin, cout):
        return nn.Sequential(
            nn.Conv1d(cin, cout, kernel_size=3, padding=1),
            nn.BatchNorm1d(cout),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
        )

    def forward(self, x):
        out = self.blocks(x)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        return self.fc(self.drop(out))


class WindowEncoder(nn.Module):

    def __init__(self, extractor, out_dim=128):
        super().__init__()
        self.extractor = extractor
        self.out_dim = out_dim

    def forward(self, x):
        B, N, C, T = x.shape
        x = x.view(B * N, C, T)
        feats = self.extractor(x)
        return feats.view(B, N, -1)
