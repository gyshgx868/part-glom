import torch

import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


class TNet(nn.Module):
    def __init__(self, in_channels):
        super(TNet, self).__init__()
        self.in_channels = in_channels
        self.conv = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv1d(in_channels, 64, kernel_size=1)),
            ('bn0', nn.BatchNorm1d(64)),
            ('relu0', nn.ReLU(inplace=True)),
            ('conv1', nn.Conv1d(64, 128, kernel_size=1)),
            ('bn1', nn.BatchNorm1d(128)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv1d(128, 1024, kernel_size=1)),
            ('bn2', nn.BatchNorm1d(1024)),
            ('relu2', nn.ReLU(inplace=True)),
        ]))
        self.fc = nn.Sequential(OrderedDict([
            ('fc0', nn.Linear(1024, 512)),
            ('bn0', nn.BatchNorm1d(512)),
            ('relu0', nn.ReLU(inplace=True)),
            ('fc1', nn.Linear(512, 256)),
            ('bn1', nn.BatchNorm1d(256)),
            ('relu1', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(256, in_channels**2)),
        ]))

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.conv(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = self.fc(x)
        identity = torch.eye(self.in_channels).view(-1, self.in_channels**2)
        identity = identity.repeat(batch_size, 1).to(x.device)
        x = x + identity
        x = x.view(-1, self.in_channels, self.in_channels)
        return x


class PointClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(PointClassifier, self).__init__()
        self.fc = nn.Conv1d(in_channels, num_classes, kernel_size=1)

    def forward(self, features):
        # features: (B, C, N)
        features = F.dropout(features, p=0.5, training=self.training)
        features = self.fc(features)
        features = features.permute(0, 2, 1)  # (B, N, L)
        return features


class StackedLatentLinear(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(StackedLatentLinear, self).__init__()
        self.num_classes = num_classes
        self.hidden = nn.ModuleList()
        for _ in range(num_classes):
            self.hidden.append(nn.Sequential(OrderedDict([
                ('conv', nn.Conv1d(in_channels, out_channels, kernel_size=1)),
                ('bn', nn.BatchNorm1d(out_channels)),
                ('relu', nn.ReLU(inplace=True)),
            ])))

    def forward(self, features):
        embeddings = []
        for i in range(self.num_classes):
            h = self.hidden[i](features)
            embeddings.append(h.unsqueeze(1))
        embeddings = torch.cat(embeddings, dim=1)  # (B, L, C, N)
        B, L, C, N = embeddings.size()
        embeddings = embeddings.permute(0, 3, 1, 2)
        embeddings = embeddings.reshape(B * N, L, C)  # (B*N, L, C)
        return embeddings
