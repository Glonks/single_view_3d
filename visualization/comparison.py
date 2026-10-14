import os
import torch
import imageio
import numpy as np
import torch.nn.functional as F

from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import default_collate
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes

from datasets import R2N2Chairs

from .turntable import make_turntable


DEMO_INDICES      = [513, 548, 604, 96, 50, 250, 450, 650]
DEMO_VIEW_INDICES = [1, 9, 2, 3, 0, 0, 0, 0]


def load_ground_truth_mesh(dataset_root, model_id):
    vertices, faces, _ = load_obj(
        R2N2Chairs.mesh_path(dataset_root, model_id),
        load_textures=False
    )

    return Meshes([vertices], [faces.verts_idx])


def make_input_panel(image, image_size):
    return (
        F.interpolate(
            image,
            size=(image_size, image_size),
            mode='bilinear',
            align_corners=False
        )[0]
        .permute(1, 2, 0)
        .clamp(0, 1)
        .mul(255)
        .numpy()
        .astype(np.uint8)
    )


def make_header(labels, image_size):
    header = Image.new('RGB', (len(labels) * image_size, 32), 'white')
    draw = ImageDraw.Draw(header)
    font = ImageFont.load_default(size=20)

    for label_index, label in enumerate(labels):
        draw.text((label_index * image_size + 8, 6), label, fill='black', font=font)

    return np.array(header)


def render_comparisons(config, models, predict):
    assert len(config.names) == len(models), 'one name per checkpoint'
    assert len(config.indices) == len(config.view_indices), 'one view per sample'

    os.makedirs(config.output_dir, exist_ok=True)

    dataset = R2N2Chairs('test', return_voxels=False)

    first_frames = []
    for index, view_index in zip(config.indices, config.view_indices):
        batch = default_collate([dataset.get_sample(index, view_index)])

        mesh_center = torch.from_numpy(dataset.mesh_centers[index]).float().to(config.device)

        structures = []
        labels = ['input']
        with torch.no_grad():
            for name, model in zip(config.names, models):
                structure = predict(model, batch)

                if structure is None:
                    structures.append(None)
                    labels.append(f'{name} empty')

                    continue

                structure = structure.to(config.device)

                if isinstance(structure, Meshes):
                    structures.append(structure.offset_verts(mesh_center))
                else:
                    structures.append(structure.offset(mesh_center))

                labels.append(name)

        labels.append('ground truth')

        ground_truth_mesh = load_ground_truth_mesh(config.dataset_root, dataset.model_ids[index])

        turntable = make_turntable(
            structures + [ground_truth_mesh],
            image_size=config.image_size,
            device=config.device
        )

        header = make_header(labels, config.image_size)
        input_panel = make_input_panel(batch['image'], config.image_size)

        frames = [
            np.vstack([header, np.hstack([input_panel, frame])])
            for frame
            in turntable
        ]

        output = os.path.join(config.output_dir, f'test_{index}_view_{view_index}')
        imageio.mimsave(f'{output}.gif', frames, duration=1000 // 15, loop=0)
        imageio.imwrite(f'{output}.png', frames[0])

        first_frames.append(frames[0])

        print(f'Saved {output}.gif')

    imageio.imwrite(os.path.join(config.output_dir, 'all_samples.png'), np.vstack(first_frames))
