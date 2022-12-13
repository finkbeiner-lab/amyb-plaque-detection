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

def tile_separation(
    step: Tuple[int, int],
    size: Tuple[int, int],
) -> List[Tuple[int, int]]:
    return tuple([(_size + _step - 1) // _step for _step, _size in zip(step, size)])

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

def get_clipped_mask(
    tile: np.ndarray,
    box: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    clipped = box.copy()
    clipped[:2] = np.maximum(tile[:2], clipped[:2]) - tile[:2]
    clipped[2:] = np.minimum(tile[2:], clipped[2:]) - tile[:2]

    offset = clipped + np.concatenate([tile[:2] - box[:2]] * 2)
    mask = mask[offset[1]:offset[3] + 1, offset[0]:offset[2] + 1]

    return clipped, mask


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
        self.labels = np.array(self.labels)
        self.boxes = np.array(self.boxes)

        self.tile_map = OrderedDict()
        for i, (box, mask) in enumerate(zip(self.boxes, self.masks)):
            for tile in tiles_per_box(box, self.step, self.size, self.offset):
                if get_clipped_mask(get_tile(tile, self.step, self.size, self.offset), box, mask)[1].sum() > 100:
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
        tile, bounds = self.tiles[idx], super().__getitem__(idx)
        ids = self.tile_map[tile]

        labels, boxes, masks = np.zeros((len(ids),)), np.zeros((len(ids), 4)), np.zeros([len(ids), *((bounds[2:] - bounds[:2]) + 1)])
        for i, (label, box, mask) in enumerate(zip(self.labels[ids], self.boxes[ids], [self.masks[id] for id in ids])):
            box, mask = get_clipped_mask(bounds, box, mask)
            labels[i] = label
            boxes[i] = box
            masks[i, box[1]:box[3] + 1, box[0]:box[2] + 1] = mask

        return dict(
            labels=torch.as_tensor(labels).to(dtype=torch.long),
            boxes=torch.as_tensor(boxes).to(dtype=torch.float),
            masks=torch.as_tensor(masks).to(dtype=torch.bool),
        )


class VipsJsonDataset(JsonDataset):
    def __init__(
        self,
        vips_img_name: str,
        json_name: str,
        label_names: List[str],
        bands: Optional[int] = 3,
        **kwargs,
    ) -> None:
        self.vips_img = pyvips.Image.new_from_file(vips_img_name, level=0)[:bands]
        self.vips_offset = tuple(map(lambda axis: int(self.vips_img.get(f'openslide.bounds-{axis}')), 'xy'))
        super().__init__(
            json_name,
            label_names,
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
        tile = get_tile(self.tiles[idx], self.step, self.size, tuple(map(sum, zip(self.vips_offset, self.offset))))
        return ToTensor()(VipsDataset.read_vips(self.vips_img, tile)), super().__getitem__(idx)



def show(i, t):
    i = (ToTensor()(i) * 255).to(torch.uint8)
    i = torchvision.utils.draw_bounding_boxes(i, t['boxes'])
    i = torchvision.utils.draw_segmentation_masks(i, t['masks'])
    return i





if __name__ == '__main__':
    label_names = 'Core Diffuse Neuritic CAA'.split()
    vips_img_dir = '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/amy-def-mfg'
    json_dir = '/home/gryan/projects/qupath/annotations/amyloid'
    vips_img_names = ['09-063', '10-033']

    vips_img_fnames = [os.path.join(vips_img_dir, f'XE{vips_img_name}_1_AmyB_1.mrxs') for vips_img_name in vips_img_names]
    json_fnames = [os.path.join(json_dir, f'{vips_img_name}.json') for vips_img_name in vips_img_names]

    # step, size = [tuple([1024] * 2)] * 2
    # ds = JsonDataset(json_fnames[0], label_names, step=step, size=size)
    # vds = VipsDataset(vips_img_fnames[0], ds.step, ds.size, ds.offset)
    # vds.tiles = ds.tiles


    tile_size = 1024
    ds_train = JsonDataset(json_fnames[0], label_names, step=(tile_size // 2, tile_size // 2), size=(tile_size, tile_size))
    ds_test = JsonDataset(json_fnames[0], label_names, step=(tile_size, tile_size), size=(tile_size, tile_size))
    
    # # ds_test_tiles = np.array(ds_test.tiles)
    # # ds_test_tiles = list(map(tuple, ds_test_tiles[np.random.permutation(np.arange(ds_test_tiles.shape[0]))]))
    # # test_tiles = ds_test_tiles[:10]
    test_tiles = [(71, 65), (73, 68), (68, 39), (61, 32), (25, 102), (74, 64), (63, 75), (71, 66), (72, 67), (67, 66)]
    
    test_boxes = [get_tile(tile, ds_test.step, ds_test.size, ds_test.offset) for tile in test_tiles]
    train_tiles = list(set(sum([tiles_per_box(box, ds_train.step, ds_train.size, ds_train.offset) for box in test_boxes], start=list())))
    
    test_idxs = [i for i, tile in enumerate(ds_test.tiles) if tile in test_tiles]
    train_idxs = [i for i, tile in enumerate(ds_train.tiles) if tile not in train_tiles]
    
    ds_train = torch.utils.data.dataset.Subset(ds_train, train_idxs)
    ds_test = torch.utils.data.dataset.Subset(ds_test, test_idxs)
