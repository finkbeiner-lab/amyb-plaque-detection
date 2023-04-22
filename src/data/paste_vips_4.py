from typing import Any, Callable, List, Mapping, Optional, Tuple
from dataclasses import dataclass, field, asdict

import os
import sys

import numpy as np
import pyvips

import torch
from torch import nn, Tensor

import torchvision
from torchvision.transforms import ToTensor, ToPILImage


def get_tile(
    tile: Tuple[int, int],
    step: Tuple[int, int],
    size: Tuple[int, int],
    offset: Tuple[int, int],
) -> np.ndarray:
    return torch.as_tensor(tuple([(_tile * _step) + (i * (_size - 1)) + _offset for i in range(2) for _tile, _step, _size, _offset in zip(tile, step, size, offset)]))


# BEGIN models.model_utils

def evaluate(model, device, image, thresh=None, mask_thresh=None):
    model.train(False)
    out = model.forward([image.to(device)], None)[0]
    if thresh is not None:
        idxs = out['scores'] >= thresh
        out = dict([(k, v[idxs]) for k, v in out.items()])
    if 'masks' in out.keys():
        out['masks'] = (out['masks'].squeeze(1) > (0.5 if mask_thresh is None else mask_thresh)).to(torch.bool)
    return out

def show(image, target, label_names=None, label_colors=None):
    labels = [f'{label}: {target["scores"][i].item():.2f}' if 'scores' in target.keys() else f'{label}' for i, label in enumerate(target['labels'] if label_names is None else [label_names[label - 1] for label in target['labels']])]
    colors = None if label_colors is None else [label_colors[label - 1] for label in target['labels']]
    image = torchvision.utils.draw_bounding_boxes(image, target['boxes'], labels=labels, colors=colors)
    if 'masks' in target.keys():
        image = torchvision.utils.draw_segmentation_masks(image, target['masks'].to(torch.bool), alpha=0.5, colors=(['red'] * len(target['labels'])))
    return image

# END model.model_utils


def progress_wrapper(it, progress=False, **kwargs):
    if progress:
        if 'tqdm' not in globals().keys():
            import tqdm
        return tqdm.tqdm(it, **kwargs)
    return it

def paste_vips_targets(vips_img, coords, targets, progress=False, progress_desc=None, **kwargs):
    """
    Replaces each tile with the result of get_vips_tile

    vips_img: a pyvips image
    coords: a [(x1, y1, x2, y2), ...]
    targets: a [dict(labels=[...], boxes=[...], masks=[...])] (labels.size() == (n,), boxes.size() == (n, 4), masks.size() == (n, h, w))

    progress: optional tqdm progress bar
    progress_desc: optional tqdm progress bar description

    label_names: optional list of names to use for boxes, indexed by label
    label_colors: optional list of colors to use for masks, indexed by label
    """


    vips_img_out = vips_img.copy()

    for coord, target in progress_wrapper(zip(coords, targets), progress=progress, desc=progress_desc):
        crop = vips_img.crop(*coord[:2], *(1 + coord[2:] - coord[:2])).numpy()

        image = torch.as_tensor(crop).permute(2, 0, 1)

        out = show(image, target, **kwargs).permute(1, 2, 0).numpy()
        vips_img_out = vips_img_out.insert(pyvips.Image.new_from_array(out), *coord[:2])

    return vips_img_out

def paste_vips_tiles(model, device, vips_img, coords, progress=False, progress_desc=None, **kwargs):
    """
    Replaces each tile with the result of get_vips_tile

    model: an nn.Module
    device: a torch.device
    vips_img: a pyvips image
    coords: a [(x1, y1, x2, y2), ...]

    progress: optional tqdm progress bar
    progress_desc: optional tqdm progress bar description

    label_names: optional list of names to use for boxes, indexed by label
    label_colors: optional list of colors to use for masks, indexed by label
    """

    vips_img_out = vips_img.copy()

    for coord in progress_wrapper(coords, progress=progress, desc=progress_desc):
        crop = vips_img.crop(*coord[:2], *(1 + coord[2:] - coord[:2])).numpy()

        image = torch.as_tensor(crop).permute(2, 0, 1)
        target = evaluate(model, device, image.to(torch.float) / 255)

        out = show(image, target, **kwargs).permute(1, 2, 0).numpy()
        vips_img_out = vips_img_out.insert(pyvips.Image.new_from_array(out), *coord[:2])

    return vips_img_out

def paste_vips_slide(model, device, slide_dir, tiles_dir, out_dir, slide_names, step, size, offset, **kwargs):
    for slide_name in slide_names:
        slide_fname = os.path.join(slide_dir, f'XE{slide_name}_1_AmyB_1.mrxs')
        tiles_fname = os.path.join(tiles_dir, f'XE{slide_name}_1_AmyB_1.tiles.npy')
        out_fname = os.path.join(out_dir, f'XE{slide_name}_1_AmyB_1.tif')

        vips_img = pyvips.Image.new_from_file(slide_fname)[:3]
        vips_tiles = torch.as_tensor(np.load(tiles_fname))

        # coords = [get_tile(tile, step, size, offset) for tile in vips_tiles]
        coords = torch.as_tensor([[38054, 82248, 39077, 83271]])

        vips_img_out = paste_vips_tiles(model, device, vips_img, coords, progress=True, progress_desc=slide_name, **kwargs)
        # vips_img_out.write_to_file(out_fname, compression='lzw')
        return vips_img_out


### Code below here is specific to a local machine ###

__pkg = os.path.abspath(os.path.join(__file__, *('..'.split() * 2)))
if __pkg not in sys.path:
    sys.path.append(__pkg)

import PIL

from models import rcnn_conf



if __name__ == '__main__':
    slide_dir = '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/amy-def-mfg-test/'
    tiles_dir = '/gladstone/finkbeiner/steve/work/data/npsad_data/slide_masks/'
    out_dir = '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/amy-def-mfg-test-outputs/'

    weights_dir = '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/models/'
    weights_name = 'autumn-capybara-474_mrcnn_model_30.pth'
    weights_fname = os.path.join(weights_dir, weights_name)


    # Slide-level parameters

    slide_names = '09-028'.split(' ')


    # Tile-level parameters

    step, size, offset = [1024, 1024, 0]
    step, size, offset = [tuple([_] * 2) for _ in (step, size, offset)]


    # Model-level parameters

    device = torch.device('cpu')

    model = rcnn_conf.rcnn_conf(num_classes=5).module()
    model.to(device)
    model.load_state_dict(torch.load(weights_fname, map_location=device))

    label_names='Core Diffuse Neuritic CAA'.split()
    label_colors='red green blue black'.split()


    # Run

    vips_img_out = paste_vips_slide(model, device, slide_dir, tiles_dir, out_dir, slide_names, step, size, offset, label_names=label_names, label_colors=label_colors)

    coords = torch.as_tensor([[38054, 82248, 39077, 83271]])
    crop = PIL.Image.fromarray(vips_img_out.crop(*coords[:2], *(1 + coords[2:] - coords[:2])).numpy())



    # coords = torch.as_tensor([[38054, 82248, 39077, 83271]])
    # coords = [get_tile(tile, step, size, offset) for tile in vips_tiles] # List[Tuple[int, int, int, int]] ([(x1, y1, x2, y2), ...])
    # coords = [get_tile(tile, step, size, offset) for tile in vips_tiles[torch.randperm(vips_tiles.size()[0])][:5]] # List[Tuple[int, int, int, int]] ([(x1, y1, x2, y2), ...])

    # Run

    # vips_img_out = paste_vips_tiles(model, device, vips_img, coords, progress=True, progress_desc=slide_name, label_names=label_names, label_colors=label_colors)
    # crops_out = [vips_img_out.crop(*coord[:2], *(1 + coord[2:] - coord[:2])).numpy() for coord in coords]
    # crops_out = [PIL.Image.fromarray(crop) for crop in crops_out]
