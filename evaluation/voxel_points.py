import torch
import mcubes
import numpy as np

from typing import Optional
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes

from datasets import BOX_HALF_SIZE, cell_size


def grid_to_world(points, grid_size: int):
    return (points[..., [2, 1, 0]] - 1 + 0.5) * cell_size(grid_size) - BOX_HALF_SIZE


def _marching_cubes(probabilities, isovalue):
    grid = (
        probabilities
        .reshape(probabilities.shape[-3:])
        .cpu()
        .numpy()
    )

    return mcubes.marching_cubes(np.pad(grid, 1), isovalue)


def voxels_to_mesh(probabilities, isovalue: float = 0.5):
    vertices, faces = _marching_cubes(probabilities, isovalue)

    if len(faces) == 0:
        return None

    vertices = grid_to_world(torch.tensor(vertices).float(), probabilities.shape[-1])

    faces = torch.tensor(faces.astype(int))[:, [0, 2, 1]]

    return Meshes([vertices], [faces])


def voxels_to_points(
    probabilities,
    num_points: int,
    isovalue: float = 0.5,
    generator: Optional[torch.Generator] = None
):
    vertices, faces = _marching_cubes(probabilities, isovalue)

    if len(faces) == 0:
        return None

    mesh = Meshes(
        [torch.tensor(vertices).float()],
        [torch.tensor(faces.astype(int))]
    )

    with torch.random.fork_rng(devices=[]):
        if generator is not None:
            torch.manual_seed(int(torch.randint(2 ** 31, (1,), generator=generator)))

        points = sample_points_from_meshes(mesh, num_points)

    return grid_to_world(points, probabilities.shape[-1])
