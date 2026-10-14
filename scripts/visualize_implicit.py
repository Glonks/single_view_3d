import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tyro

from dataclasses import dataclass, field
from functools import partial

from datasets import R2N2Chairs
from models import ImplicitNet
from evaluation import voxels_to_mesh
from visualization import DEMO_INDICES, DEMO_VIEW_INDICES, render_comparisons


@dataclass
class Config:
    names:        list[str] = field(default_factory=lambda: ['prediction'])
    checkpoints:  list[str] = field(default_factory=lambda: ['runs/implicit_ablation_uniform_queries/best.pt'])
    grid_size:    int       = 32
    indices:      list[int] = field(default_factory=lambda: list(DEMO_INDICES))
    view_indices: list[int] = field(default_factory=lambda: list(DEMO_VIEW_INDICES))
    dataset_root: str       = R2N2Chairs.DATASET_ROOT
    output_dir:   str       = 'results/implicit'
    image_size:   int       = 256
    device:       str       = 'cuda'


def predict(model, batch, grid_size):
    return voxels_to_mesh(model.predict_grid(batch, grid_size=grid_size)[0])


def main(config: Config):
    print(config)

    models = [
        ImplicitNet.load(checkpoint, device=config.device).eval()
        for checkpoint
        in config.checkpoints
    ]

    render_comparisons(config, models, partial(predict, grid_size=config.grid_size))


if __name__ == '__main__':
    config = tyro.cli(Config)

    main(config)
