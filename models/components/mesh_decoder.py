import torch
import torch.nn as nn

from pytorch3d.utils import ico_sphere
from pytorch3d.structures import Meshes


class MeshDecoder(nn.Module):
    def __init__(self, ico_level: int = 4, feature_dim: int = 512, device: str = 'cuda'):
        super().__init__()

        self.device = device

        template = ico_sphere(ico_level)

        self.register_buffer('template_vertices', template.verts_packed(), persistent=False)
        self.register_buffer('template_faces', template.faces_packed(), persistent=False)

        self.num_vertices = self.template_vertices.shape[0]

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.num_vertices * 3),
        )

        with torch.no_grad():
            self.mlp[-1].weight.zero_()
            self.mlp[-1].bias.zero_()

        self.to(device)

    def forward(self, features):
        batch_size = features.shape[0]

        offsets = (
            self.mlp(features)
            .reshape(batch_size, self.num_vertices, 3)
        )

        vertices = self.template_vertices + offsets

        faces = (
            self.template_faces
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )

        return Meshes(verts=vertices, faces=faces)
