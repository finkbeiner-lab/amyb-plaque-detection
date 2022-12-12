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
) -> Tuple[int, int, int, int]:
    return tuple([(_tile * _step) + (i * _size) + _offset for i in range(2) for _tile, _step, _size, _offset in zip(tile, step, size, offset)])


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
    ) -> Tuple[int, int, int, int]:
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

    def __getitem__(
        self,
        idx: int,
    ) -> Any:
        x1, y1, x2, y2 = super().__getitem__(idx)
        vips_crop = self.vips_img.crop(x1, y1, x2 - x1, y2 - y1)
        return vips_crop.numpy()
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
        # super().__init__(**kwargs)
        with open(json_name, 'r') as f:
            self.json_obj = json.loads(f.read())
        self.annotation_obj = self.json_obj.get('annotations')
        self.tile_obj = self.json_obj.get('tiles')

    @staticmethod
    def read_json(json_obj, label_names):
        label = np.array(1 + label_names.index(json_obj.get('pathClasses')[0])).astype(np.intc)
        points = [np.array(json_obj.get(ax)).astype(np.intc) for ax in 'xy']
        bbox = [f(ax) for f in (np.amin, np.amax) for ax in points]
        mask = np.zeros((bbox[2:] - bbox[:2])[::-1]).astype(np.uint8)

        


if __name__ == '__main__':
    vips_img_dir = '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/amy-def-mfg'
    json_dir = '/home/gryan/projects/qupath/annotations/amyloid'
    vips_img_names = ['09-063', '10-033']

    vips_img_fnames = [os.path.join(vips_img_dir, f'XE{vips_img_name}_1_AmyB_1.mrxs') for vips_img_name in vips_img_names]
    json_fnames = [os.path.join(json_dir, f'{vips_img_name}.json') for vips_img_name in vips_img_names]



    # @staticmethod
    # def vips_crop(vips_img, x, y, w, h, bands=3):
    #     x, y = tuple(map(sum, zip((x, y), [int(vips_img.get(f'openslide.bounds-{coord}')) for coord in 'x y'.split()])))
    #     vips_crop = vips_img[:bands].crop(x, y, w, h)
    #     return np.ndarray(buffer=vips_crop.write_to_memory(), dtype=np.uint8, shape=(vips_crop.height, vips_crop.width, vips_crop.bands))

# class VipsDataset(Dataset):
#     def __init__(
#         self,
#     )
