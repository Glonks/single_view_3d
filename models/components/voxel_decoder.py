import math
import torch
import torch.nn as nn


class VoxelDecoder(nn.Module):
    def __init__(self, feature_dim=512, occupancy_prior=0.03, device='cuda'):
        super().__init__()

        self.device = device

        self.feature_to_volume = nn.Linear(feature_dim, 2048 * 2 * 2 * 2)

        self.deconv_block = nn.Sequential(
            nn.ConvTranspose3d(2048, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.ConvTranspose3d(512, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.ConvTranspose3d(8, 1, kernel_size=1),
        )

        with torch.no_grad():
            self.deconv_block[-1].bias.fill_(math.log(occupancy_prior / (1 - occupancy_prior)))

        self.to(device)

    def forward(self, features):
        batch_size = features.shape[0]

        volume = (
            self.feature_to_volume(features)
            .reshape(batch_size, 2048, 2, 2, 2)
        )

        return self.deconv_block(volume)