import torch

from datasets import cell_centers

from .components import ImageEncoder, ImplicitDecoder
from .reconstruction_net import ReconstructionNet


class ImplicitNet(ReconstructionNet):
    def __init__(self, pretrained: bool = True, device: str = 'cuda'):
        super().__init__()

        self.device = device

        self.encoder = ImageEncoder(pretrained=pretrained, device=device)
        self.decoder = ImplicitDecoder(device=device)

        self.to(device)

    def forward(self, batch, query_points):
        return self.decoder(self.encode(batch), query_points)

    def predict_grid(self, batch, grid_size: int = 32, chunk_size: int = 8192):
        features = self.encode(batch)
        batch_size = features.shape[0]

        query_points = cell_centers(grid_size, self.device)

        logits = torch.cat([
            self.decoder(features, chunk.expand(batch_size, -1, -1))
            for chunk
            in query_points.split(chunk_size)
        ], dim=1)

        return (
            torch.sigmoid(logits)
            .reshape(batch_size, 1, grid_size, grid_size, grid_size)
        )
