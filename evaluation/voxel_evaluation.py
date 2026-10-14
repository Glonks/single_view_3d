import torch

from .voxel_points import voxels_to_points
from .sampling_metrics import compute_sampling_metrics, empty_metrics


def evaluate_voxel_model(predict, loader, isovalue: float = 0.5, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)

    per_sample = []
    with torch.no_grad():
        for batch in loader:
            probabilities = predict(batch).cpu()

            for sample_index in range(probabilities.shape[0]):
                ground_truth_points = batch['surface_points'][sample_index:sample_index + 1]

                predicted_points = voxels_to_points(
                    probabilities[sample_index],
                    ground_truth_points.shape[1],
                    isovalue=isovalue,
                    generator=generator
                )

                if predicted_points is None:
                    metrics = empty_metrics()
                else:
                    metrics = compute_sampling_metrics(predicted_points, ground_truth_points)

                per_sample.append({
                    'index':      int(batch['index'][sample_index]),
                    'view_index': int(batch['view_index'][sample_index]),
                    'empty':      predicted_points is None,
                    **{
                        name: float(value[0])
                        for name, value
                        in metrics.items()
                    },
                })

    return per_sample

