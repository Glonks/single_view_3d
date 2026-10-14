import torch

from .sampling_metrics import compute_sampling_metrics


def evaluate_point_model(predict, loader):
    per_sample = []
    with torch.no_grad():
        for batch in loader:
            predicted_points = predict(batch).float()
            ground_truth_points = batch['surface_points'].to(predicted_points.device)

            metrics = compute_sampling_metrics(predicted_points, ground_truth_points)

            for sample_index in range(predicted_points.shape[0]):
                per_sample.append({
                    'index':      int(batch['index'][sample_index]),
                    'view_index': int(batch['view_index'][sample_index]),
                    'empty':      False,
                    **{
                        name: float(value[sample_index])
                        for name, value
                        in metrics.items()
                    },
                })

    return per_sample
