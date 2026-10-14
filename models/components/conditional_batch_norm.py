import torch.nn as nn


class ConditionalBatchNorm(nn.Module):
    def __init__(self, condition_dim, feature_dim):
        super().__init__()

        self.batch_norm = nn.BatchNorm1d(feature_dim, affine=False)

        self.scale_linear = nn.Linear(condition_dim, feature_dim)
        self.shift_linear = nn.Linear(condition_dim, feature_dim)

        nn.init.zeros_(self.scale_linear.weight)
        nn.init.ones_(self.scale_linear.bias)
        nn.init.zeros_(self.shift_linear.weight)
        nn.init.zeros_(self.shift_linear.bias)

    def forward(self, x, condition):
        scale = self.scale_linear(condition).unsqueeze(-1)
        shift = self.shift_linear(condition).unsqueeze(-1)

        return scale * self.batch_norm(x) + shift
