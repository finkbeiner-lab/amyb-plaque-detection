from typing import Any, Callable, List, Mapping, Optional, Tuple
from dataclasses import dataclass, field, asdict

import os
import pdb
import sys
__pkg = os.path.abspath(os.path.join(__file__, *('..'.split() * 2)))
if __pkg not in sys.path:
    sys.path.append(__pkg)

from collections import OrderedDict

from skimage.draw.draw import polygon

import json
import numpy as np
import pyvips

import torch
from torch import nn, Tensor
from torch.utils.data import Dataset

import torchvision
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks


intdiv = lambda x1, x2: (x1 // x2) if x1 >= 0 else -((x2 - (x1 + 1)) // x2)
overlap = lambda x1, x2, m, l, b: tuple(map(lambda _: intdiv(_, m), ((x1 - b) - (l - m), x2 - b)))

def tiles_per_box(
    box: Tuple[int, int, int, int],
    step: Tuple[int, int],
    size: Tuple[int, int],
    offset: Tuple[int, int],
) -> List[Tuple[int, int]]:
    xs, ys = [list(range(i1, i2 + 1)) for i1, i2 in [overlap(*_) for _ in zip(box[:2], box[2:], step, size, offset)]]
    return [(x, y) for x in xs for y in ys]

def boxes_per_tiles(
    boxes: List[Tuple[int, int, int, int]],
    step: Tuple[int, int],
    size: Tuple[int, int],
    offset: Tuple[int, int],
) -> Mapping[Tuple[int, int], int]:
    tiles = OrderedDict()
    for i, t in enumerate(map(lambda box: tiles_per_box(box, step, size, offset), boxes)):
        tiles.setdefault(t, list()).append(i)
    return tiles

def get_tile(
    tile: Tuple[int, int],
    step: Tuple[int, int],
    size: Tuple[int, int],
    offset: Tuple[int, int],
) -> np.ndarray:
    return np.array(tuple([(_tile * _step) + (i * (_size - 1)) + _offset for i in range(2) for _tile, _step, _size, _offset in zip(tile, step, size, offset)]))

def get_clipped(
    tile: np.ndarray,
    box: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    clipped = box.copy()
    clipped[:2] = np.maximum(tile[:2], clipped[:2]) - tile[:2]
    clipped[2:] = np.minimum(tile[2:], clipped[2:]) - tile[:2]

    offset = clipped + np.concatenate([tile[:2] - box[:2]] * 2)

    return clipped, offset


class TileDataset(Dataset):
    def __init__(
        self,
        step: Tuple[int, int],
        size: Tuple[int, int],
        offset: Optional[Tuple[int, int]] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        self.tiles = list()

        self.step = step
        self.size = size
        self.offset = (0, 0) if offset is None else offset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(
        self,
        idx: int,
    ) -> np.ndarray:
        return get_tile(self.tiles[idx], self.step, self.size, self.offset)
        # return np.array(get_tile(self.tiles[idx], self.step, self.size, self.offset))


class VipsDataset(TileDataset):
    def __init__(
        self,
        vips_img_name: str,
        step: Tuple[int, int],
        size: Tuple[int, int],
        offset: Optional[Tuple[int, int]] = None,
        bands: Optional[int] = 3,
        **kwargs,
    ) -> None:
        self.vips_img = pyvips.Image.new_from_file(vips_img_name, level=0)[:bands]
        self.vips_offset = tuple(map(lambda axis: int(self.vips_img.get(f'openslide.bounds-{axis}')), 'xy'))
        super().__init__(
            step,
            size,
            offset=self.vips_offset if offset is None else tuple(map(sum, zip(self.vips_offset, offset))),
            **kwargs,
        )

    @staticmethod
    def read_vips(vips_img, tile):
        x, y = tile[:2]
        w, h = (tile[2:] - tile[:2]) + 1
        return vips_img.crop(x, y, w, h).numpy()

    def __getitem__(
        self,
        idx: int,
    ) -> Any:
        return VipsDataset.read_vips(self.vips_img, super().__getitem__(idx))
        # tile = super().__getitem__(idx)
        # x, y = tile[:2]
        # w, h = (tile[2:] - tile[:2]) + 1
        # vips_crop = self.vips_img.crop(x, y, w, h)
        # return vips_crop.numpy()
        # return np.ndarray(
        #     buffer=vips_crop.write_to_memory(),
        #     dtype=np.uint8,
        #     shape=(vips_crop.height, vips_crop.width, vips_crop.bands),
        # )


class JsonDataset(TileDataset):
    def __init__(
        self,
        json_name: str,
        label_names: List[str],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        with open(json_name, 'r') as f:
            self.json = json.loads(f.read())

        self.labels, self.boxes, self.masks = zip(*map(lambda _: JsonDataset.read_json(_, label_names), self.json.get('annotations')))

        self.tile_map = OrderedDict()
        for i, (box, mask) in enumerate(zip(self.boxes, self.masks)):
            for tile in tiles_per_box(box, self.step, self.size, self.offset):
                self.tile_map.setdefault(tile, list()).append(i)
        self.tiles = list(self.tile_map.keys())

    @staticmethod
    def read_json(json_obj, label_names):
        label = np.array(1 + label_names.index(json_obj.get('pathClasses')[0])).astype(np.intc)

        points = [np.array(json_obj.get(ax)).astype(np.intc) for ax in 'xy']
        bbox = np.array([f(ax) for f in (np.amin, np.amax) for ax in points])

        x, y = bbox[:2]
        w, h = (bbox[2:] - bbox[:2]) + 1
        xx, yy = polygon(points[0] - x, points[1] - y)

        mask = np.zeros((h, w)).astype(np.uint8)
        mask[yy, xx] = 1

        return label, bbox, mask

    def __getitem__(
        self,
        idx: int,
    ) -> Any:
        tile, id = super().__getitem__(idx), self.tiles[idx]

        labels, boxes, masks = list(), list(), list()
        for label, box, mask in [[a[i] for a in [self.labels, self.boxes, self.masks]] for i in self.tile_map[id]]:
            labels.append(label)
            box_clipped, box_offset = get_clipped(tile, box)
            boxes.append(box_clipped)
            masks.append(mask[box_offset[1]:box_offset[3] + 1, box_offset[0]:box_offset[2] + 1])

        return labels, boxes, masks



    # def tiles_for_box(self, box):
    #     return tiles_per_box(box, self.step, self.size, self.offset)
    #
    # def tiles_for_mask(self, box, mask):
    #     tiles = self.tiles_for_box(box)
    #     if len(tiles) == 0:
    #         return None
    #
    #     for tile in tiles:
    #         clipped, offset = get_clipped(tile, box)
    #         x1, y1, x2, y2 = offset
    #         mask[y1:y2 + 1, x1:x2 + 1]
        # tile = np.array(get_tile(tiles[0], self.step, self.size, self.offset))
        #
        # clip, offsets = [box.copy() for _ in range(2)]
        # clip[:2] = np.maximum(tile[:2], clip[:2]) - tile[:2]
        # clip[2:] = np.minimum(tile[2:], clip[2:]) - tile[:2]
        # offsets[:2] = tile[:2] + clip[:2] - box[:2]
        # offsets[2:] = box[:2] + clip[2:] - clip[:2]
        # # box_offsets[:2] = tile[:2] + box[:2] - box_offsets[:2]
        # # box_offsets[2:] = box[2:] - box[:2] + box_offsets[:2]
        #
        # return tile, clip, offsets, box







if __name__ == '__main__':
    label_names = 'Core Diffuse Neuritic CAA'.split()
    vips_img_dir = '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/amy-def-mfg'
    json_dir = '/home/gryan/projects/qupath/annotations/amyloid'
    vips_img_names = ['09-063', '10-033']

    vips_img_fnames = [os.path.join(vips_img_dir, f'XE{vips_img_name}_1_AmyB_1.mrxs') for vips_img_name in vips_img_names]
    json_fnames = [os.path.join(json_dir, f'{vips_img_name}.json') for vips_img_name in vips_img_names]

    step, size = [tuple([1024] * 2)] * 2

    ds = JsonDataset(json_fnames[0], label_names, step=step, size=size)


    # @staticmethod
    # def vips_crop(vips_img, x, y, w, h, bands=3):
    #     x, y = tuple(map(sum, zip((x, y), [int(vips_img.get(f'openslide.bounds-{coord}')) for coord in 'x y'.split()])))
    #     vips_crop = vips_img[:bands].crop(x, y, w, h)
    #     return np.ndarray(buffer=vips_crop.write_to_memory(), dtype=np.uint8, shape=(vips_crop.height, vips_crop.width, vips_crop.bands))

# class VipsDataset(Dataset):
#     def __init__(
#         self,
#     )
