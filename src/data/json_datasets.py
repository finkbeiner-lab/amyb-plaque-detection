import os
import pdb
import sys
sys.path.append(os.path.join(os.getcwd()))

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

import data
from data.dataset_utils import bboxes_to_tiles_map



class JsonDataset(Dataset):
    def __init__(self, json_name, label_names, size=(1024, 1024,), stride=(1024, 1024,), offset=(0, 0,),):
        self.size, self.stride, self.offset = size, stride, offset

        labels, bboxes, masks = zip(*tuple(map(lambda json_obj: JsonDataset.json_to_instance(json_obj, label_names), json.loads(open(json_name, 'r').read()).get('annotations'))))
        self.labels, self.bboxes, self.masks = np.array(labels), np.array(bboxes), masks

        self.tiles = bboxes_to_tiles_map(self.bboxes, self.size, self.stride, self.offset)

    @staticmethod
    def json_to_instance(json_obj, label_names):
        label = np.array(1 + label_names.index(json_obj.get('pathClasses')[0])).astype(np.intc)
        points = [np.array(json_obj.get(c)).astype(np.intc) for c in 'x y'.split()]
        bbox = np.array([f(axis) for f in (np.amin, np.amax) for axis in points])
        mask = np.zeros((bbox[2:] - bbox[:2])[::-1]).astype(np.uint8)

        mask_y, mask_x = polygon(points[1] - bbox[1], points[0] - bbox[0], mask.shape)
        mask[mask_y, mask_x] = 1

        return label, bbox, mask

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile = list(self.tiles.keys())[idx]
        tile_min = np.array([(coord * stride) + offset for coord, stride, offset in zip(tile, self.stride, self.offset)])
        tile_max = np.array([coord + size for coord, size in zip(tile_min, self.size)])

        idxs = self.tiles.get(tile)
        labels, bboxes, bbox_offsets = self.labels[idxs], self.bboxes[idxs], self.bboxes[idxs]

        bboxes[:, :2] = np.maximum(tile_min, bboxes[:, :2]) - tile_min
        bboxes[:, 2:] = np.minimum(tile_max, bboxes[:, 2:]) - tile_min

        bbox_offsets[:, :2] = tile_min + bboxes[:, :2] - bbox_offsets[:, :2]
        bbox_offsets[:, 2:] = bboxes[:, 2:] - bboxes[:, :2] + bbox_offsets[:, :2]

        masks = np.zeros(tuple([len(idxs), *((tile_max - tile_min)[::-1])])).astype(np.uint8)
        for i, idx in enumerate(idxs):
            masks[i, bboxes[i, 1]:bboxes[i, 3], bboxes[i, 0]:bboxes[i, 2]] = self.masks[idx][bbox_offsets[i, 1]:bbox_offsets[i, 3], bbox_offsets[i, 0]:bbox_offsets[i, 2]]

        return dict(
            labels=torch.as_tensor(labels).to(dtype=torch.long),
            boxes=torch.as_tensor(bboxes).to(dtype=torch.long),
            masks=torch.as_tensor(masks).to(dtype=torch.bool),
        )

class VipsDataset(JsonDataset):
    def __init__(self, vips_img_name, json_name, **kwargs):
        super().__init__(json_name, **kwargs)
        self.vips_img = pyvips.Image.new_from_file(vips_img_name, level=0)
        self.transform = ToTensor()

    @staticmethod
    def vips_crop(vips_img, x, y, w, h, bands=3):
        x, y = tuple(map(sum, zip((x, y), [int(vips_img.get(f'openslide.bounds-{coord}')) for coord in 'x y'.split()])))
        vips_crop = vips_img[:bands].crop(x, y, w, h)
        return np.ndarray(buffer=vips_crop.write_to_memory(), dtype=np.uint8, shape=(vips_crop.height, vips_crop.width, vips_crop.bands))

    def __getitem__(self, idx):
        return self.transform(VipsDataset.vips_crop(self.vips_img, *[(coord * stride) + offset for coord, stride, offset in zip(list(self.tiles.keys())[idx], self.stride, self.offset)], *self.size, bands=3)), super().__getitem__(idx)


if __name__ == '__main__':
    label_names = 'Pre Mature Ghost'.split()

    vips_img_dir = '/gladstone/finkbeiner/steve/work/data/npsad_data/gennadi/tau_ad_mrxs/'
    vips_img_name = 'XE16-014_1_Tau_1'
    vips_img_fname = os.path.join(vips_img_dir, f'{vips_img_name}.mrxs')

    json_dir = '/home/gryan/projects/qupath/annotations/tau/'
    json_fname = os.path.join(json_dir, f'{vips_img_name}.json')

    dataset = VipsDataset(vips_img_fname, json_fname, label_names=label_names,)

    to_pil = ToPILImage()
    for image, target in dataset:
        if len(input('Show next tile: ')) > 0:
            break
        image = draw_bounding_boxes((image * 255).to(torch.uint8), target['boxes'], colors='black')
        image = draw_segmentation_masks(image, target['masks'])
        to_pil(image).show()
