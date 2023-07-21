from typing import Any, Callable, List, Mapping, Optional, Tuple
from dataclasses import dataclass, field, asdict

import os
import sys

__pkg = os.path.abspath(os.path.join(__file__, *('..'.split() * 2)))
if __pkg not in sys.path:
    sys.path.append(__pkg)

import numpy as np
import PIL
import pyvips

import torch
from torch import nn, Tensor

import torchvision
from torchvision.transforms import ToTensor, ToPILImage

from models import rcnn_conf
# from data import datasets


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

def target_offset(target_src, target_dst, offset):
    offset_box = torch.tensor(tuple(offset[:2]) * 2)
    boxes_dst = target_src['boxes'] + offset_box

    masks_src = target_src['masks']
    masks_dst = torch.zeros(target_src['masks'].size()[:1] + target_dst['masks'].size()[1:]).to(torch.bool)
    masks_dst[:, offset_box[1]:(offset_box[1] + masks_src.size()[1]), offset_box[0]:(offset_box[0] + masks_src.size()[2])] = masks_src

    target_dst['scores'] = torch.concatenate([target_dst['scores'], target_src['scores']], axis=0)
    target_dst['labels'] = torch.concatenate([target_dst['labels'], target_src['labels']], axis=0)
    target_dst['boxes'] = torch.concatenate([target_dst['boxes'], boxes_dst], axis=0)
    target_dst['masks'] = torch.concatenate([target_dst['masks'], masks_dst], axis=0)
    return target_dst


def tile_overlap_map(tiles, bounds):
    validate = lambda bounds, coords: bounds[0] <= coords[0] and bounds[1] <= coords[1] and bounds[2] >= coords[0] and bounds[3] >= coords[1]

    # overlaps[(x, y)] = [((2 * x) + dx, (2 * y) + dy) for dy in range(-1, 2) for dx in range(-1, 2)]
    overlaps = dict()
    for x, y in tiles:
        if validate(bounds, (2 * x, 2 * y)):
            for xi, yi in [((2 * x) + dx, (2 * y) + dy) for dy in range(-1, 2) for dx in range(-1, 2)]:
                if validate(bounds, (xi, yi)):
                    overlaps.setdefault((x, y), list()).append((xi, yi))

    overlaps_inv = dict()
    for k, v in overlaps.items():
        for t in v:
            overlaps_inv.setdefault(t, list()).append(k)

    return overlaps, overlaps_inv

def loadable_tiles(out_dir, tiles):
    return [(x, y) for x, y in tiles if os.path.isfile(os.path.join(out_dir, f'{x},{y}.pt'))]

def load_tiles(out_dir, tiles):
    return [torch.load(os.path.join(out_dir, f'{x},{y}.pt')) for tile in tiles]


if __name__ == '__main__':
    slide_dir = '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/amy-def-mfg-test/'
    tiles_dir = '/gladstone/finkbeiner/steve/work/data/npsad_data/slide_masks/'
    # out_dir = '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/amy-def-mfg-test-outputs/'
    out_dir = '/gladstone/finkbeiner/steve/work/data/npsad_data/gennadi/half_tiles_2'

    weights_dir = '/gladstone/finkbeiner/steve/work/data/npsad_data/gennadi/weights'
    weights_name = 'autumn-capybara-474_mrcnn_model_30.pth'
    weights_fname = os.path.join(weights_dir, weights_name)

    # Slide-level parameters
    # slide_names = '09-028'.split(' ')
    slide_names = '09-035'.split(' ')
    slide_fnames = [os.path.join(slide_dir, f'XE{slide_name}_1_AmyB_1.mrxs') for slide_name in slide_names]
    tile_fnames = [os.path.join(tiles_dir, f'XE{slide_name}_1_AmyB_1.tiles.npy') for slide_name in slide_names]
    tile_out_dirs = [os.path.join(out_dir, slide_name) for slide_name in slide_names]

    # Tile-level parameters

    idx = 0
    vips_img = pyvips.Image.new_from_file(slide_fnames[idx])[:3]
    tiles = list(map(tuple, np.load(tile_fnames[idx])))

    step_size = 512
    bounds = (0, 0, (vips_img.width // step_size) - 2, (vips_img.height // step_size) - 2)

    # step, size, offset = [step_size, step_size * 2, 0]
    # step, size, offset = [tuple([_] * 2) for _ in (step, size, offset)]
    #
    # dataset = datasets.VipsDataset(slide_fnames[0], step, size, offset)
    # dataset.tiles.extend(list(map(tuple, np.load(tile_fnames[0]))))

    # Model-level parameters

    device = torch.device('cuda', 0)

    model = rcnn_conf.rcnn_conf(num_classes=5).module()
    model.to(device)
    model.load_state_dict(torch.load(weights_fname, map_location=device))

    label_names='Core Diffuse Neuritic CAA'.split()
    label_colors='red green blue black'.split()


    tile_map, tile_map_inv = tile_overlap_map(tiles, bounds)
    tiles_inv = list(tile_map_inv.keys())

    # for x, y in progress_wrapper(tiles_inv, progress=True, desc=slide_names[idx]):
    #     vips_tile = vips_img.crop(x * step_size, y * step_size, 2 * step_size, 2 * step_size)
    #
    #     image = torch.tensor(vips_tile.numpy()).permute(2, 0, 1) / 255.
    #     target = evaluate(model, device, image)
    #     target = dict([(k, v.detach().to(torch.device('cpu'))) for k, v in target.items()])
    #
    #     if target['labels'].size()[0] > 0:
    #         torch.save(target, os.path.join(tile_out_dirs[idx], f'{x},{y}.pt'))

    def target_merge(targets, tiles, tile, step_size, fns=None):
        offsets = [tuple([(a + 1 - b * 2) * step_size for a, b in zip(_, tile)]) for _ in tiles]

        tile_idxs = torch.zeros((0,), dtype=torch.long)
        inst_idxs = torch.zeros((0,), dtype=torch.long)
        target = dict(
            boxes=torch.zeros((0, 4), dtype=torch.float),
            labels=torch.zeros((0,), dtype=torch.long),
            scores=torch.zeros((0,), dtype=torch.float),
            masks=torch.zeros((0, 4 * step_size, 4 * step_size), dtype=torch.bool),
        )

        for i, (_target, _offset) in enumerate(zip(targets, offsets)):
            tile_idxs = torch.concatenate([tile_idxs, torch.ones((len(_target['labels']),), dtype=torch.long) * i])
            inst_idxs = torch.concatenate([inst_idxs, torch.arange(len(_target['labels']), dtype=torch.long)])
            target = target_offset(_target, target, _offset)

        for fn in (fns if fns is not None else list()):
            target, keep_idxs = fn(target)
            tile_idxs = tile_idxs[keep_idxs]
            inst_idxs = inst_idxs[keep_idxs]
            target = dict([(k, v[keep_idxs, ...]) for k, v in target.items()])

        return tile_idxs, inst_idxs, target

    def clip(tile, box):
        clipped = box.clone()
        clipped[..., :2] = torch.maximum(tile[..., :2], clipped[..., :2]) - tile[..., :2]
        clipped[..., 2:] = torch.minimum(tile[..., 2:], clipped[..., 2:]) - tile[..., :2]
        in_bounds = (clipped[..., :2] <= clipped[..., 2:]).all(dim=-1)
        return clipped, in_bounds

    def clip_merge(target, tile):
        clipped_boxes, in_bounds = clip(tile, target['boxes'])
        target['boxes'] = clipped_boxes
        target['masks'] = target['masks'][..., tile[0]:tile[2], tile[1]:tile[3]]
        keep_idxs = torch.where(in_bounds)[0]
        return target, keep_idxs

    nms_thresh = 0.5
    tile = (15, 153)
    tiles = tile_map[tile]
    tiles = [(x, y) for x, y in tiles if os.path.isfile(os.path.join(out_dir, slide_names[idx], f'{x},{y}.pt'))]
    targets = [torch.load(os.path.join(out_dir, slide_names[idx], f'{x},{y}.pt')) for x, y in tiles]
    tile_idxs, inst_idxs, target = target_merge(targets, tiles, tile, step_size, fns=[
        lambda _: (_, torchvision.ops.remove_small_boxes(_['boxes'], step_size * 1e-2)),
        lambda _: clip_merge(_, torch.tensor([step_size, step_size, 3 * step_size, 3 * step_size], dtype=torch.long)),
        lambda _: torchvision.ops.nms(_['boxes'], _['scores'], nms_thresh),
    ])
    # target_copy = dict([(k, v.clone()) for k, v in target.items()])


    # nms_thresh = 0.5
    # t = (x, y)
    # overlap_tiles = tile_map[(x, y)]
    # overlap_tiles = [(x, y) for x, y in overlap_tiles if os.path.exists(os.path.join(out_dir, slide_names[idx], f'{x},{y}.pt'))]
    # targets = [torch.load(os.path.join(out_dir, slide_names[idx], f'{x},{y}.pt')) for x, y in overlap_tiles]
    # overlap_offsets = [tuple([a + 1 - b * 2 for a, b in zip(_, t)]) for _ in overlap_tiles]
    # target = dict(boxes=torch.zeros((0, 4), dtype=torch.float), labels=torch.zeros((0,), dtype=torch.long), scores=torch.zeros((0,), dtype=torch.float), masks=torch.zeros((0, 4 * step_size, 4 * step_size), dtype=torch.bool))
    # tile_idxs = torch.zeros((0,), dtype=torch.long)
    # obj_idxs = torch.zeros((0,), dtype=torch.long)
    # for tar, off in zip(targets, overlap_offsets):
    #   tile_idxs = torch.concatenate([tile_idxs, torch.ones((len(tar['labels']),), dtype=torch.long)])
    #   obj_idxs = torch.concatenate([obj_idxs, torch.arange(len(tar['labels']), dtype=torch.long)])
    #   target = target_offset(tar, target, off)
    # target_nms_idxs = torchvision.ops.nms(target['boxes'], target['scores'], nms_thresh)
    # target_batched_nms_idxs = torchvision.ops.nms(target['boxes'], target['scores'], target['labels'], nms_thresh)
    # tile_nms_idxs = tile_idxs[target_nms_idxs]
    # obj_nms_idxs = obj_idxs[target_nms_idxs]
    # target_nms = dict([(k, v[target_nms_idxs, ...]) for k, v in target.items()])


    # region = [50, 50, 3, 3]
    #
    # vips_crop = dataset.vips_img.crop(dataset.vips_offset[0] + (region[0] * size[0]), dataset.vips_offset[1] + (region[1] * size[1]), region[2] * size[0], region[3] * size[1])
    #
    # tiles = [((region[0] * 2) + x, (region[1] * 2) + y) for y in range((region[3] * 2) - 1) for x in range((region[2] * 2) - 1)]
    # dataset.tiles.extend(tiles)
    #
    # images = list()
    # targets = list()
    # vizs = list()
    # offsets = list()
    #
    # for i, tile in progress_wrapper(enumerate(tiles), progress=True):
    #     image = dataset[i]
    #     target = evaluate(model, device, image)
    #
    #     image = (image * 255).to(torch.uint8)
    #     target = dict([(k, v.detach().to(torch.device('cpu'))) for k, v in target.items()])
    #     viz = show(image, target, label_names=label_names, label_colors=label_colors)
    #     offset = tuple([(sp * t) - (sz * r) for sp, sz, t, r in zip(step, size, tile, region[:2])])
    #
    #     images.append(image)
    #     targets.append(target)
    #     vizs.append(viz)
    #     offsets.append(offset)
    #
    # nms_thresh = 0.5
    #
    # image_out = torch.as_tensor(vips_crop.numpy()).permute(2, 0, 1)
    # target_out = dict(
    #     scores=torch.zeros((0,)).to(torch.float),
    #     labels=torch.zeros((0,)).to(torch.long),
    #     boxes=torch.zeros((0, 4)).to(torch.float),
    #     masks=torch.zeros((0, *image_out.size()[1:][::-1])).to(torch.bool),
    # )
    # for target, offset in zip(targets, offsets):
    #     target_out = target_offset(target, target_out, offset)
    # viz_out = show(image_out, target_out, label_names=label_names, label_colors=label_colors)
    #
    # target_out_nms_idxs = torchvision.ops.nms(target_out['boxes'], target_out['scores'], nms_thresh)
    # target_out_nms = dict([(k, v[target_out_nms_idxs, ...]) for k, v in target_out.items()])
    # viz_out_nms = show(image_out, target_out_nms, label_names=label_names, label_colors=label_colors)
    #
    # target_out_batched_nms_idxs = torchvision.ops.batched_nms(target_out['boxes'], target_out['scores'], target_out['labels'], nms_thresh)
    # target_out_batched_nms = dict([(k, v[target_out_batched_nms_idxs, ...]) for k, v in target_out.items()])
    # viz_out_batched_nms = show(image_out, target_out_batched_nms, label_names=label_names, label_colors=label_colors)



    # Run

    # paste_vips_slide(model, device, slide_dir, tiles_dir, out_dir, slide_names, step, size, offset, label_names=label_names, label_colors=label_colors)
