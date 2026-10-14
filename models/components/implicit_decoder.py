import torch.nn as nn
import torch.nn.functional as F

from .conditional_batch_norm import ConditionalBatchNorm
from .conditional_residual_block import ConditionalResidualBlock


class ImplicitDecoder(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=256, num_blocks=5, device='cuda'):
        super().__init__()

        self.device = device

        self.point_linear = nn.Conv1d(3, hidden_dim, kernel_size=1)

        self.blocks = nn.ModuleList([
            ConditionalResidualBlock(feature_dim, hidden_dim)
            for _
            in range(num_blocks)
        ])

        self.norm = ConditionalBatchNorm(feature_dim, hidden_dim)
        self.output_linear = nn.Conv1d(hidden_dim, 1, kernel_size=1)

        self.to(device)

    def forward(self, features, query_points):
        x = self.point_linear(query_points.transpose(1, 2))

        for block in self.blocks:
            x = block(x, features)

        return (
            self.output_linear(F.relu(self.norm(x, features)))
            .squeeze(1)
        )
