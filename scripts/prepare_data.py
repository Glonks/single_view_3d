import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import glob
import tyro
import torch
import numpy as np

from PIL import Image
from dataclasses import dataclass
from multiprocessing import Pool
from pytorch3d.io import load_obj
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from pytorch3d.datasets.r2n2.utils import read_binvox_coords

from datasets import R2N2Chairs


@dataclass
class Config:
    dataset_root:       str = R2N2Chairs.DATASET_ROOT
    cache_root:         str = R2N2Chairs.CACHE_ROOT
    num_surface_points: int = 20000
    num_workers:        int = 8
    limit:              int = 0


def get_rendering_dir(dataset_root, model_id):
    return os.path.join(
        dataset_root,
        'r2n2/ShapeNetRendering',
        R2N2Chairs.SYNSET_ID,
        model_id,
        'rendering'
    )


def load_shape(arguments):
    dataset_root, model_id, view_indices, num_surface_points = arguments

    rendering_dir = get_rendering_dir(dataset_root, model_id)

    images = np.stack([
        np.array(Image.open(os.path.join(rendering_dir, f'{view_index:02d}.png')))[..., :3]
        for view_index
        in view_indices
    ])

    vertices, faces, _ = load_obj(
        R2N2Chairs.mesh_path(dataset_root, model_id),
        load_textures=False
    )

    torch.manual_seed(0)
    surface_points = sample_points_from_meshes(
        Meshes([vertices], [faces.verts_idx]),
        num_surface_points
    )[0]

    mesh_bounds = torch.stack([
        vertices.min(dim=0)[0],
        vertices.max(dim=0)[0],
    ])

    voxel_path = os.path.join(
        dataset_root,
        'r2n2/ShapeNetVoxels',
        R2N2Chairs.SYNSET_ID,
        model_id,
        'model.binvox'
    )
    with open(voxel_path, 'rb') as file:
        binvox_coordinates = read_binvox_coords(file)

    return {
        'images':         images.astype(np.uint8),
        'surface_points': surface_points.numpy().astype(np.float16),
        'mesh_bounds':    mesh_bounds.numpy().astype(np.float32),
        'binvox_coords':  binvox_coordinates.numpy().astype(np.uint8),
    }


def prepare_split(config: Config, split, models):
    split_dir = os.path.join(config.cache_root, split)
    os.makedirs(split_dir, exist_ok=True)

    for voxels_path in glob.glob(os.path.join(split_dir, 'voxels*.npy')):
        os.remove(voxels_path)

    model_ids = sorted(models)
    if config.limit > 0:
        model_ids = model_ids[:config.limit]

    num_models = len(model_ids)

    view_indices = models[model_ids[0]]
    assert all(models[model_id] == view_indices for model_id in model_ids), 'all models must have the same views'

    def open_array(name, shape, dtype):
        return np.lib.format.open_memmap(
            os.path.join(split_dir, f'{name}.npy'),
            mode='w+',
            dtype=dtype,
            shape=shape
        )

    arrays = {
        'images':         open_array('images', (num_models, len(view_indices), 137, 137, 3), np.uint8),
        'surface_points': open_array('surface_points', (num_models, config.num_surface_points, 3), np.float16),
        'mesh_bounds':    open_array('mesh_bounds', (num_models, 2, 3), np.float32),
    }

    binvox_coordinates = []

    jobs = [
        (config.dataset_root, model_id, view_indices, config.num_surface_points)
        for model_id
        in model_ids
    ]

    with Pool(config.num_workers) as pool:
        for model_index, model in enumerate(pool.imap(load_shape, jobs, chunksize=4)):
            for name, array in arrays.items():
                array[model_index] = model[name]

            binvox_coordinates.append(model['binvox_coords'])

            if (model_index % 100 == 0) or (model_index == num_models - 1):
                print(f'\r{split}: {model_index + 1} / {num_models}', end='')

    print()

    for array in arrays.values():
        array.flush()

    binvox_offsets = np.cumsum([0] + [len(coordinates) for coordinates in binvox_coordinates])
    np.save(os.path.join(split_dir, 'binvox_coords.npy'), np.concatenate(binvox_coordinates))
    np.save(os.path.join(split_dir, 'binvox_offsets.npy'), binvox_offsets)

    with open(os.path.join(split_dir, 'meta.json'), 'w') as file:
        json.dump({
            'model_ids':    model_ids,
            'view_indices': view_indices,
        }, file, indent=2)


def main(config: Config):
    print(config)

    with open(os.path.join(config.dataset_root, f'split_{R2N2Chairs.SYNSET_ID}.json')) as file:
        splits = json.load(file)

    for split in ['train', 'test']:
        prepare_split(config, split, splits[split][R2N2Chairs.SYNSET_ID])


if __name__ == '__main__':
    config = tyro.cli(Config)

    main(config)
