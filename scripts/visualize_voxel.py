import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tyro
import torch

from dataclasses import dataclass, field

from datasets import R2N2Chairs
from models import VoxelNet
from evaluation import voxels_to_mesh
from visualization import DEMO_INDICES, DEMO_VIEW_INDICES, render_comparisons


@dataclass
class Config:
    names:        list[str] = field(default_factory=lambda: ['prediction'])
    checkpoints:  list[str] = field(default_factory=lambda: ['runs/voxel_images_fixed_centered/best.pt'])
    indices:      list[int] = field(default_factory=lambda: list(DEMO_INDICES))
    view_indices: list[int] = field(default_factory=lambda: list(DEMO_VIEW_INDICES))
    dataset_root: str       = R2N2Chairs.DATASET_ROOT
    output_dir:   str       = 'results/voxel'
    image_size:   int       = 256
    device:       str       = 'cuda'


def predict(model, batch):
    return voxels_to_mesh(torch.sigmoid(model(batch))[0])


def main(config: Config):
    print(config)

    models = [
        VoxelNet.load(checkpoint, device=config.device).eval()
        for checkpoint
        in config.checkpoints
    ]

    render_comparisons(config, models, predict)


if __name__ == '__main__':
    config = tyro.cli(Config)

    main(config)
