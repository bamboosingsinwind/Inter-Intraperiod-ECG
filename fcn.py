import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self, n_in):
        super(FCN, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(n_in, 128, kernel_size=7, padding=6, dilation=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=8, dilation=4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=3, padding=8, dilation=8),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        self.proj_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

    def forward(self, x):

        h = self.encoder(x)
        out = self.proj_head(h)

        return out, h

class FCN1(nn.Module):
    def __init__(self, n_in):
        super(FCN1, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(n_in, 128, kernel_size=7, padding=6, dilation=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=8, dilation=4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=3, padding=8, dilation=4),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, kernel_size=3, padding=6, dilation=4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        self.proj_head = nn.Sequential(
            nn.Linear(256+2, 128+2),
            nn.BatchNorm1d(128+2),
            nn.ReLU(),
            nn.Linear(128+2, 2)
        )

    def forward(self, x,feat):

        h = self.encoder(x)
        h = torch.cat((h,feat),1)
        out = self.proj_head(h)

        return out#, h

# model = FCN(n_in=8)
# inp = torch.randn(19,8,600)
# p,h = model(inp)
# print(p.shape," ",h.shape)