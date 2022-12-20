import os
import sys
__pkg = os.path.abspath(os.path.join(__file__, *('..'.split() * 2)))
if __pkg not in sys.path:
    sys.path.append(__pkg)

from functools import reduce

import cv2
import numpy as np
import pyvips

import torch
from torch import nn, Tensor

import torchvision
from torchvision.transforms import ToPILImage


def crop_tile(slide, out_dir, tile, ext='png'):
    out_file = os.path.join(out_dir, '_'.join(map(lambda _: '_'.join(_), zip('xy', map(str, tile[:2])))) + f'.{ext}')
    crop = slide.crop(*tile).numpy()
    ToPILImage()(crop).save(out_file)


def read_slide_tiles(slide_name, slide_dir, tile_dir, out_dir, tile_size, ext='png'):
    slide_path = os.path.join(slide_dir, f'{slide_name}.mrxs')
    tile_path = os.path.join(tile_dir, f'{slide_name}.tiles.npy')
    out_dir = os.path.join(out_dir, slide_name)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    slide = pyvips.Image.new_from_file(slide_path, level=0)
    tiles = np.load(tile_path, allow_pickle=False)
    tiles = np.concatenate([tiles + i for i in range(2)], axis=1)
    tiles *= np.concatenate([tile_size] * 2)
    tiles[:, 2:] -= tiles[:, :2]

    for tile in tiles:
        crop_tile(slide, out_dir, tile, ext=ext)

    return len(tiles)




if __name__ == '__main__':
    slide_dir = '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/amy-def-mfg-test'
    tile_dir = '/gladstone/finkbeiner/steve/work/data/npsad_data/slide_masks'
    out_dir = '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/amyb_wsi/test/images'
    tile_size = (1024, 1024)

    slide_names = ['.'.join(file.split('.')[:-1]) for file in next(os.walk(slide_dir))[2]]

    for slide_name in slide_names:
        num_tiles = read_slide_tiles(slide_name, slide_dir, tile_dir, out_dir, tile_size)
        print(f'{num_tiles}:{slide_name}')
