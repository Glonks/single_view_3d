import torch
import numpy as np

from pytorch3d.renderer import (
    AlphaCompositor,
    FoVPerspectiveCameras,
    HardPhongShader,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    RasterizationSettings,
    TexturesVertex,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes, Pointclouds


def add_color(structure, color=(0.7, 0.7, 1.0)):
    color = torch.tensor(color, device=structure.device)

    if isinstance(structure, Meshes):
        vertices = structure.verts_padded()

        return Meshes(
            structure.verts_list(),
            structure.faces_list(),
            textures=TexturesVertex(torch.ones_like(vertices) * color)
        )

    points = structure.points_padded()

    return Pointclouds(points=points, features=torch.ones_like(points) * color)


def make_turntable(
    structures: list,
    image_size: int = 256,
    num_frames: int = 36,
    distance: float = 1.2,
    elevation: float = 30.0,
    device: str = 'cuda'
):
    structures = [
        None if structure is None else add_color(structure.to(device))
        for structure
        in structures
    ]

    mesh_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            raster_settings=RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1)
        ),
        shader=HardPhongShader(device=device)
    )
    points_renderer = PointsRenderer(
        rasterizer=PointsRasterizer(
            raster_settings=PointsRasterizationSettings(image_size=image_size, radius=0.01)
        ),
        compositor=AlphaCompositor(background_color=(1, 1, 1))
    )

    empty_panel = np.ones((image_size, image_size, 3))

    frames = []
    for frame_index in range(num_frames):
        R, T = look_at_view_transform(
            dist=distance,
            elev=elevation,
            azim=360.0 * frame_index / num_frames
        )

        cameras = FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)
        lights = PointLights(location=cameras.get_camera_center(), device=device)

        panels = []
        for structure in structures:
            if structure is None:
                panels.append(empty_panel)
            elif isinstance(structure, Meshes):
                panels.append(mesh_renderer(structure, cameras=cameras, lights=lights)[0, ..., :3].cpu().numpy())
            else:
                panels.append(points_renderer(structure, cameras=cameras)[0, ..., :3].cpu().numpy())

        frames.append((np.hstack(panels).clip(0, 1) * 255).astype(np.uint8))

    return frames
