from .components import ImageEncoder, VoxelDecoder
from .reconstruction_net import ReconstructionNet


class VoxelNet(ReconstructionNet):
    def __init__(self, occupancy_prior: float = 0.03, pretrained: bool = True, device: str = 'cuda'):
        super().__init__()

        self.device = device

        self.encoder = ImageEncoder(pretrained=pretrained, device=device)
        self.decoder = VoxelDecoder(occupancy_prior=occupancy_prior, device=device)

        self.to(device)

    def forward(self, batch):
        return self.decoder(self.encode(batch))
