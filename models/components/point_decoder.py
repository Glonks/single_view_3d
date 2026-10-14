import torch
import torch.nn as nn


class PointDecoder(nn.Module):
    def __init__(self, num_points=1000, feature_dim=512, device='cuda'):
        super().__init__()

        self.device = device
        self.num_points = num_points

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_points * 3),
        )

        self.to(device)

    def forward(self, features):
        batch_size = features.shape[0]

        points = (
            self.mlp(features)
            .reshape(batch_size, self.num_points, 3)
        )

        return torch.tanh(points)
