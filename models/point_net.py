from .components import ImageEncoder, PointDecoder
from .reconstruction_net import ReconstructionNet


class PointNet(ReconstructionNet):
    CHECKPOINT_KEYS = ('num_points',)

    def __init__(self, num_points: int = 1000, pretrained: bool = True, device: str = 'cuda'):
        super().__init__()

        self.device = device

        self.encoder = ImageEncoder(pretrained=pretrained, device=device)
        self.decoder = PointDecoder(num_points=num_points, device=device)

        self.to(device)

    def forward(self, batch):
        return self.decoder(self.encode(batch))
