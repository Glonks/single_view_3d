import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import tyro
import torch
import shutil
import torch.nn.functional as F

from dataclasses import dataclass
from torch.utils.data import DataLoader
from typing import Literal

from datasets import R2N2Chairs, make_test_loader
from models import VoxelNet
from evaluation import evaluate_voxel_model, summarize
from logger import Logger


@dataclass
class Config:
    name:                  str                                      = 'voxel'
    num_steps:             int                                      = 30000
    batch_size:            int                                      = 32
    learning_rate:         float                                    = 4e-4
    encoder_learning_rate: float                                    = 1e-4
    positive_weight:       float                                    = 1.0
    occupancy_prior:       float                                    = 0.03
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


def compute_voxel_loss(logits, voxels, positive_weight):
    return F.binary_cross_entropy_with_logits(
        logits,
        voxels,
        pos_weight=positive_weight
    )


def evaluate(model, loader, config: Config):
    model.eval()

    per_sample = evaluate_voxel_model(
        lambda batch: torch.sigmoid(model(batch)),
        loader,
        seed=config.seed
    )

    summary = summarize(per_sample)

    model.train()

    return {
        'Test/F1@0.05': summary['f1@0.05'],
        'Test/F1@0.02': summary['f1@0.02'],
        'Test/F1@0.01': summary['f1@0.01'],
        'Test/Empty':   summary['empty'],
    }


def main(config: Config):
    print(config)

    torch.manual_seed(config.seed)

    logger = Logger('single_view_3d', config, wandb_mode=config.wandb_mode)

    train_dataset = R2N2Chairs('train', view_mode='random')
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
        return_voxels=False,
        seed=config.seed
    )

    model = VoxelNet(occupancy_prior=config.occupancy_prior, device=config.device)

    optimizer = torch.optim.Adam([
        {'params': model.decoder.parameters(), 'lr': config.learning_rate},
        {'params': model.encoder.parameters(), 'lr': config.encoder_learning_rate},
    ])

    positive_weight = torch.tensor(config.positive_weight, device=model.device)

    model.train()

    batches = infinite_batches(train_loader)
    best_f1 = -1.0
    start_time = time.time()
    for step in range(1, config.num_steps + 1):
        batch = next(batches)

        voxels = batch['voxels'].to(model.device, non_blocking=True)

        logits = model(batch)

        loss = compute_voxel_loss(logits, voxels, positive_weight)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % config.log_every == 0:
            logger.log({
                'Train/Loss':             loss.item(),
                'Train/Seconds per step': (time.time() - start_time) / step,
            }, step=step, overwrite=True)

        if (step % config.eval_every == 0) or (step == config.num_steps):
            metrics = evaluate(model, test_loader, config)

            logger.log(metrics, step=step)

            checkpoint = {
                'step':      step,
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
