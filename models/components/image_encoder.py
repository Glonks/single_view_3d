import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    def __init__(self, pretrained=True, device='cuda'):
        super().__init__()

        self.device = device

        resnet = torchvision.models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.to(device)

    def forward(self, images):
        images = F.interpolate(
            images,
            size=224,
            mode='bilinear',
            antialias=True,
            align_corners=False
        )

        return (
            self.backbone((images - self.mean) / self.std)
            .flatten(1)
        )
