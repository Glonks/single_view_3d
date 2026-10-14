import torch
import torch.nn.functional as F


BOX_HALF_SIZE = 0.5


def cell_size(grid_size: int):
    return 2 * BOX_HALF_SIZE / grid_size


def cell_centers(grid_size, device):
    coordinates = (torch.arange(grid_size, device=device) + 0.5) * cell_size(grid_size) - BOX_HALF_SIZE

    z, y, x = torch.meshgrid(
        coordinates,
        coordinates,
        coordinates,
        indexing='ij'
    )

    return torch.stack([x, y, z], dim=-1).reshape(-1, 3)


def read_occupancy(voxels, query_points):
    grid = (
        (query_points / BOX_HALF_SIZE)
        .reshape(query_points.shape[0], 1, 1, -1, 3)
    )

    return (
        F.grid_sample(
            voxels,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )
        .reshape(query_points.shape[0], -1)
    )
