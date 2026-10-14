import torch.nn as nn
import torch.nn.functional as F

from .conditional_batch_norm import ConditionalBatchNorm


class ConditionalResidualBlock(nn.Module):
    def __init__(self, condition_dim, hidden_dim):
        super().__init__()

        self.norm1 = ConditionalBatchNorm(condition_dim, hidden_dim)
        self.linear1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)

        self.norm2 = ConditionalBatchNorm(condition_dim, hidden_dim)
        self.linear2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)

        nn.init.zeros_(self.linear2.weight)

    def forward(self, x, condition):
        hidden = self.linear1(F.relu(self.norm1(x, condition)))

        hidden = self.linear2(F.relu(self.norm2(hidden, condition)))

        return x + hidden
