import torch

from pytorch3d.ops import knn_points


THRESHOLDS = [0.01, 0.02, 0.05]


def compute_sampling_metrics(predicted_points, ground_truth_points, thresholds=THRESHOLDS, epsilon=1e-8):
    predicted_to_ground_truth = (
        knn_points(predicted_points, ground_truth_points, K=1)
        .dists[..., 0]
        .sqrt()
        .cpu()
    )

    ground_truth_to_predicted = (
        knn_points(ground_truth_points, predicted_points, K=1)
        .dists[..., 0]
        .sqrt()
        .cpu()
    )

    metrics = {}
    for threshold in thresholds:
        precision = 100.0 * (predicted_to_ground_truth < threshold).float().mean(dim=1)
        recall = 100.0 * (ground_truth_to_predicted < threshold).float().mean(dim=1)

        metrics[f'f1@{threshold}'] = 2.0 * precision * recall / (precision + recall + epsilon)

    return metrics


def empty_metrics(thresholds=THRESHOLDS):
    return {
        f'f1@{threshold}': torch.zeros(1)
        for threshold
        in thresholds
    }
