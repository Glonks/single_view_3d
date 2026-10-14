import torch
import torch.nn as nn


class ReconstructionNet(nn.Module):
    CHECKPOINT_KEYS = ()

    def encode(self, batch):
        return self.encoder(batch['image'].to(self.device, non_blocking=True))

    @classmethod
    def load(cls, path: str, device: str = 'cuda'):
        checkpoint = torch.load(path, map_location='cpu', mmap=True)

        model = cls(
            **{key: checkpoint[key] for key in cls.CHECKPOINT_KEYS},
            pretrained=False,
            device=device
        )
        model.load_state_dict(checkpoint['model'])

        return model
