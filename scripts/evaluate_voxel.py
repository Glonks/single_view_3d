import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tyro
import torch

from dataclasses import dataclass

from datasets import make_test_loader
from models import VoxelNet
from evaluation import evaluate_voxel_model, summarize, format_f1, save_results


@dataclass
class Config:
    checkpoint:  str
    num_points:  int   = 1000
    isovalue:    float = 0.5
    seed:        int   = 0
    batch_size:  int   = 32
    num_workers: int   = 4
    output:      str   = ''
    device:      str   = 'cuda'


def main(config: Config):
    print(config)

    model = VoxelNet.load(config.checkpoint, device=config.device)

    loader = make_test_loader(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        num_points=config.num_points,
        return_voxels=False,
        seed=config.seed
    )

    model.eval()

    per_sample = evaluate_voxel_model(
        lambda batch: torch.sigmoid(model(batch)),
        loader,
        isovalue=config.isovalue,
        seed=config.seed
    )

    summary = summarize(per_sample)

    print(
        f'{format_f1(summary)} | '
        f'empty: {summary["empty"]} of {len(per_sample)}'
    )

    output = config.output or f'{os.path.splitext(config.checkpoint)[0]}_eval_isovalue{config.isovalue}_seed{config.seed}.json'
    save_results(output, config, summary, per_sample)


if __name__ == '__main__':
    config = tyro.cli(Config)

    main(config)
