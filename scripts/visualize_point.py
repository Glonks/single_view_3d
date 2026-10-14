import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tyro

from dataclasses import dataclass, field
from pytorch3d.structures import Pointclouds

from datasets import R2N2Chairs
from models import PointNet
from visualization import DEMO_INDICES, DEMO_VIEW_INDICES, render_comparisons


@dataclass
class Config:
    names:        list[str] = field(default_factory=lambda: ['prediction'])
    checkpoints:  list[str] = field(default_factory=lambda: ['runs/point_images_repulsion_5_centered/best.pt'])
    indices:      list[int] = field(default_factory=lambda: list(DEMO_INDICES))
    view_indices: list[int] = field(default_factory=lambda: list(DEMO_VIEW_INDICES))
    dataset_root: str       = R2N2Chairs.DATASET_ROOT
    output_dir:   str       = 'results/point'
    image_size:   int       = 256
    device:       str       = 'cuda'


def predict(model, batch):
    return Pointclouds(points=model(batch).float())


def main(config: Config):
    print(config)

    models = [
        PointNet.load(checkpoint, device=config.device).eval()
        for checkpoint
        in config.checkpoints
    ]

    render_comparisons(config, models, predict)


if __name__ == '__main__':
    config = tyro.cli(Config)

    main(config)
