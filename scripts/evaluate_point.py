import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tyro

from dataclasses import dataclass

from datasets import make_test_loader
from models import PointNet
from evaluation import evaluate_point_model, summarize, format_f1, save_results


@dataclass
class Config:
    checkpoint:  str
    seed:        int = 0
    batch_size:  int = 32
    num_workers: int = 4
    output:      str = ''
    device:      str = 'cuda'


def main(config: Config):
    print(config)

    model = PointNet.load(config.checkpoint, device=config.device)

    loader = make_test_loader(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        num_points=model.decoder.num_points,
        return_voxels=False,
        seed=config.seed
    )

    model.eval()

    per_sample = evaluate_point_model(model, loader)

    summary = summarize(per_sample)

    print(format_f1(summary))

    output = config.output or f'{os.path.splitext(config.checkpoint)[0]}_eval_seed{config.seed}.json'
    save_results(output, config, summary, per_sample)


if __name__ == '__main__':
    config = tyro.cli(Config)

    main(config)
