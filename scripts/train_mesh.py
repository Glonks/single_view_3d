import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import tyro
import torch
import shutil

from dataclasses import dataclass
from torch.utils.data import DataLoader
from typing import Literal
from pytorch3d.ops import knn_points, sample_points_from_meshes
from pytorch3d.loss import mesh_laplacian_smoothing

from datasets import R2N2Chairs, make_test_loader
from models import MeshNet
from evaluation import evaluate_mesh_model, summarize
from logger import Logger


@dataclass
class Config:
    name:                  str                                      = 'mesh'
    ico_level:             int                                      = 4
    num_points:            int                                      = 1000
    num_steps:             int                                      = 30000
    batch_size:            int                                      = 32
    learning_rate:         float                                    = 4e-4
    encoder_learning_rate: float                                    = 1e-4
    coverage_weight:       float                                    = 1.0
    smoothness_weight:     float                                    = 0.1
    eval_every:            int                                      = 2000
    log_every:             int                                      = 50
    num_workers:           int                                      = 4
    seed:                  int                                      = 0
    device:                str                                      = 'cuda'
    wandb_mode:            Literal['online', 'offline', 'disabled'] = 'online'


def infinite_batches(loader):
    while True:
        for batch in loader:
            yield batch


def compute_chamfer_loss(predicted_points, ground_truth_points, coverage_weight):
    return (
        knn_points(predicted_points, ground_truth_points, K=1).dists.mean() +
        coverage_weight * knn_points(ground_truth_points, predicted_points, K=1).dists.mean()
    )


def evaluate(model, loader, num_points, seed):
    model.eval()

    per_sample = evaluate_mesh_model(
        model,
        loader,
        num_points=num_points,
        seed=seed
    )

    summary = summarize(per_sample)

    model.train()

    return {
        'Test/F1@0.05': summary['f1@0.05'],
        'Test/F1@0.02': summary['f1@0.02'],
        'Test/F1@0.01': summary['f1@0.01'],
    }


def main(config: Config):
    print(config)

    torch.manual_seed(config.seed)

    torch.set_float32_matmul_precision('high')

    logger = Logger('single_view_3d', config, wandb_mode=config.wandb_mode)

    train_dataset = R2N2Chairs(
        'train',
        view_mode='random',
        num_points=config.num_points,
        return_voxels=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0
    )

    test_loader = make_test_loader(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        num_points=config.num_points,
        return_voxels=False,
        seed=config.seed
    )

    model = MeshNet(ico_level=config.ico_level, device=config.device)

    optimizer = torch.optim.Adam([
        {'params': model.decoder.parameters(), 'lr': config.learning_rate},
        {'params': model.encoder.parameters(), 'lr': config.encoder_learning_rate},
    ])

    model.train()

    batches = infinite_batches(train_loader)
    best_f1 = -1.0
    start_time = time.time()
    for step in range(1, config.num_steps + 1):
        batch = next(batches)

        ground_truth_points = batch['surface_points'].to(model.device, non_blocking=True)

        predicted_meshes = model(batch)

        predicted_points = sample_points_from_meshes(predicted_meshes, config.num_points)

        chamfer_loss = compute_chamfer_loss(predicted_points, ground_truth_points, config.coverage_weight)
        smoothness_loss = mesh_laplacian_smoothing(predicted_meshes, method='uniform')

        loss = chamfer_loss + config.smoothness_weight * smoothness_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % config.log_every == 0:
            logger.log({
                'Train/Loss':             loss.item(),
                'Train/Chamfer loss':     chamfer_loss.item(),
                'Train/Smoothness loss':  smoothness_loss.item(),
                'Train/Seconds per step': (time.time() - start_time) / step,
            }, step=step, overwrite=True)

        if (step % config.eval_every == 0) or (step == config.num_steps):
            metrics = evaluate(model, test_loader, config.num_points, config.seed)

            logger.log(metrics, step=step)

            checkpoint = {
                'step':      step,
                'ico_level': config.ico_level,
                'model':     model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'f1':        metrics['Test/F1@0.05'],
            }
            torch.save(checkpoint, logger.path('last.pt'))

            if metrics['Test/F1@0.05'] > best_f1:
                best_f1 = metrics['Test/F1@0.05']

                print('New best model - Saving')

                shutil.copyfile(logger.path('last.pt'), logger.path('best.pt'))

    logger.finish()


if __name__ == '__main__':
    config = tyro.cli(Config)

    main(config)
