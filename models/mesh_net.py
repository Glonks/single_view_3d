from .components import ImageEncoder, MeshDecoder
from .reconstruction_net import ReconstructionNet


class MeshNet(ReconstructionNet):
    CHECKPOINT_KEYS = ('ico_level',)

    def __init__(self, ico_level: int = 4, pretrained: bool = True, device: str = 'cuda'):
        super().__init__()

        self.device = device

        self.encoder = ImageEncoder(pretrained=pretrained, device=device)
        self.decoder = MeshDecoder(ico_level=ico_level, device=device)

        self.to(device)

    def forward(self, batch):
        return self.decoder(self.encode(batch))
