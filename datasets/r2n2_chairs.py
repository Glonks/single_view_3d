import os
import json
import torch
import numpy as np

from typing import Literal
from torch.utils.data import Dataset, DataLoader

from .voxelization import voxelize


class R2N2Chairs(Dataset):
    SYNSET_ID    = '03001627'
    CACHE_ROOT   = 'data/r2n2_chairs'
    DATASET_ROOT = 'data/r2n2_shapenet_dataset'

    def __init__(
        self,
        split: Literal['train', 'test'],
        view_mode: Literal['random', 'fixed'] = 'random',
        num_points: int = 1000,
        return_voxels: bool = True,
        seed: int = 0
    ):
        if split not in {'train', 'test'}:
            raise ValueError('Unknown split')

        split_dir = os.path.join(self.CACHE_ROOT, split)
        if not os.path.exists(os.path.join(split_dir, 'meta.json')):
            raise FileNotFoundError(f'No cache in {split_dir} - run scripts/prepare_data.py first')

        with open(os.path.join(split_dir, 'meta.json')) as file:
            metadata = json.load(file)

        self.model_ids = metadata['model_ids']
        self.num_views = len(metadata['view_indices'])

        self.view_mode     = view_mode
        self.num_points    = num_points
        self.return_voxels = return_voxels
        self.seed          = seed

        self.images         = np.load(os.path.join(split_dir, 'images.npy'), mmap_mode='r')
        self.surface_points = np.load(os.path.join(split_dir, 'surface_points.npy'), mmap_mode='r')

        mesh_bounds = np.load(os.path.join(split_dir, 'mesh_bounds.npy'))
        self.mesh_centers = mesh_bounds.mean(axis=1)

        if return_voxels:
            self.voxels = self._load_voxels(split_dir, mesh_bounds)

    @staticmethod
    def mesh_path(dataset_root, model_id):
        return os.path.join(dataset_root, 'shapenet', R2N2Chairs.SYNSET_ID, model_id, 'model.obj')

    def _load_voxels(self, split_dir, mesh_bounds):
        voxels_path = os.path.join(split_dir, 'voxels.npy')

        if os.path.exists(voxels_path):
            return np.load(voxels_path, mmap_mode='r')

        mesh_bounds = torch.from_numpy(mesh_bounds - self.mesh_centers[:, None])
        binvox_coordinates = torch.from_numpy(np.load(os.path.join(split_dir, 'binvox_coords.npy')))
        binvox_offsets = np.load(os.path.join(split_dir, 'binvox_offsets.npy'))

        voxels = np.stack([
            voxelize(
                binvox_coordinates[binvox_offsets[model_index]:binvox_offsets[model_index + 1]],
                mesh_bounds[model_index]
            )
            .numpy()
            .astype(np.bool_)
            for model_index
            in range(len(self.model_ids))
        ])

        np.save(voxels_path, voxels)

        return voxels

    def __len__(self):
        return len(self.model_ids)

    def __getitem__(self, index):
        return self.get_sample(index)

    def get_sample(self, index, view_index=None):
        seed = self.seed + index if (self.view_mode == 'fixed') or (view_index is not None) else None
        generator = np.random.default_rng(seed)

        drawn_view_index = generator.integers(self.num_views)
        if view_index is None:
            view_index = drawn_view_index

        point_indices = generator.choice(self.surface_points.shape[1], self.num_points, replace=False)

        sample = {
            'index':      index,
            'view_index': view_index,
            'image': (
                torch.from_numpy(self.images[index, view_index].astype(np.float32) / 255.0)
                .permute(2, 0, 1)
            ),
            'surface_points': torch.from_numpy(
                self.surface_points[index, np.sort(point_indices)].astype(np.float32) - self.mesh_centers[index]
            ),
        }

        if self.return_voxels:
            sample['voxels'] = (
                torch.from_numpy(self.voxels[index].astype(np.float32))
                .unsqueeze(0)
            )

        return sample


def make_test_loader(
    batch_size: int = 32,
    num_workers: int = 4,
    num_points: int = 1000,
    return_voxels: bool = True,
    seed: int = 0
):
    dataset = R2N2Chairs(
        'test',
        view_mode='fixed',
        num_points=num_points,
        return_voxels=return_voxels,
        seed=seed
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )
