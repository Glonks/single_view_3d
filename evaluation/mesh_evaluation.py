import torch

from pytorch3d.ops import sample_points_from_meshes

from .point_evaluation import evaluate_point_model


def evaluate_mesh_model(predict, loader, num_points: int = 1000, seed: int = 0):
    def predict_points(batch):
        return sample_points_from_meshes(predict(batch), num_points)

    with torch.random.fork_rng(devices=range(torch.cuda.device_count())):
        torch.manual_seed(seed)

        return evaluate_point_model(predict_points, loader)
