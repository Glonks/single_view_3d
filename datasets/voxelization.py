import torch

from .voxel_grid import BOX_HALF_SIZE, cell_size


def voxelize(binvox_coordinates, mesh_bounds, grid_size=32):
    cells = binvox_coordinates.float()[:, [2, 1, 0]] * torch.tensor([1.0, 1.0, -1.0])

    offsets = torch.tensor([-0.25, 0.25])
    offsets = torch.stack(torch.meshgrid(offsets, offsets, offsets, indexing='ij'), dim=-1).reshape(-1, 3)
    points = (cells[:, None] + offsets[None]).reshape(-1, 3)

    cell_min = cells.min(dim=0)[0] - 0.5
    cell_max = cells.max(dim=0)[0] + 0.5

    scale = (mesh_bounds[1] - mesh_bounds[0]) / (cell_max - cell_min)
    points = (points - cell_min) * scale + mesh_bounds[0]

    indices = torch.floor((points + BOX_HALF_SIZE) / cell_size(grid_size)).long()

    inbounds = (
        (indices >= 0) &
        (indices < grid_size)
    ).all(dim=1)
    indices = indices[inbounds]

    voxels = torch.zeros(grid_size, grid_size, grid_size)
    voxels[indices[:, 2], indices[:, 1], indices[:, 0]] = 1.0

    return voxels
