import argparse
import os
import sys
__pkg = os.path.abspath(os.path.join(__file__, *('..'.split() * 2)))
if __pkg not in sys.path:
    sys.path.append(__pkg)

from functools import reduce

import cv2
import numpy as np
import pyvips
import tqdm

import torch
from torch import nn, Tensor

import torchvision
from torchvision.transforms import ToPILImage


def get_cropped(slide, level):
    x, y = [int(slide.get(f'openslide.bounds-{_}')) // (2 ** level) for _ in 'xy']
    return slide.crop(x, y, slide.width - x, slide.height - y)

def get_tiles(h, w, size, level):
    return [(y, x) for x in range(((w * (2 ** level)) // size) + 1) for y in range(((h * (2 ** level)) // size) + 1)]


def tile_mask(mask, size, f=None):
    if f is None:
        f = lambda a: a.sum() > 0

    ht, wt = np.array(mask.shape) // size
    ts = np.array([(y, x) for x in range(wt + 1) for y in range(ht + 1)])
    coords = np.concatenate([ts + i for i in range(2)], axis=1) * size

    keep = np.array([i for i, (y1, x1, y2, x2) in enumerate(coords) if f(mask[y1:y2, x1:x2])])
    return coords[keep], ts[keep][:, [1, 0]]

def fill_mask(mask, k, i=1, r=0):
    for _ in range(i):
        mask = cv2.blur(mask, (k, k))
        mask = (mask > (255 * r)).astype(np.uint8) * 255
    return mask

def close_mask(mask, k, i=1):
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k)), iterations=i)

def mask_contours(mask):
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return list(map(lambda t: t[0], sorted([(contour, cv2.contourArea(contour)) for contour in contours], key=lambda t: -t[1])))

def fill_contours(contours, shape):
    fill = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(fill, contours, -1, 255, -1)
    return fill


def get_slide_mask(
    slide,
    tile_size,
    fill_params=(9, 4, 1e-2),
    close_params=(9, 4),
    viz=True,
):
    k1, i1, r1 = fill_params
    k2, i2 = close_params

    im = slide[:3].numpy()
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    thresh, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    mask = fill_mask(mask, k1, i1, r1)
    mask = close_mask(mask, k2, i2)
    contours = mask_contours(mask)
    mask = fill_contours(contours[:1], mask.shape)

    f_pos, f_neg = lambda _: _.sum() > 0, lambda _: _.sum() == 0
    tiles_pos, coords_neg = tile_mask(mask, tile_size, f=f_pos)[1], tile_mask(mask, tile_size, f=f_neg)[0]

    if viz:
        for y1, x1, y2, x2 in coords_neg:
            im[y1:y2, x1:x2] = 0
    mask_out = ToPILImage()(mask)
    viz_out = ToPILImage()(im) if viz else None

    return tiles_pos, mask_out, viz_out


def save_slide_masks(slide_name, slide_dir, out_dir, tile_size, level):
    tile_size //= 2 ** level
    slide = pyvips.Image.new_from_file(os.path.join(slide_dir, f'{slide_name}.mrxs'), level=level)
    # slide = get_cropped(slide, level)

    tile_out, mask_out, viz_out = [os.path.join(out_dir, f'{slide_name}.{suffix}') for suffix in ('tiles.npy', 'mask.png', 'viz.png')]
    tiles, mask, viz = get_slide_mask(slide, tile_size)

    np.save(tile_out, tiles, allow_pickle=False)
    mask.save(mask_out)
    viz.save(viz_out)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('slide_dir', help='directory to read *.mrxs slides from')
    parser.add_argument('output_dir', help='directory to write mask, tiles, and visualization to')
    parser.add_argument('tile_size', type=int, help='size of tiles referenced in the output')
    parser.add_argument('level', type=int, help='downsample level at which slides are read')
    parser.add_argument('--slide_names', nargs='+', help='filenames (without extension) of slides to process (defaults to all)')

    args = parser.parse_args()
    slide_dir, output_dir, tile_size, level, slide_names = map(args.__getattribute__, 'slide_dir output_dir tile_size level slide_names'.split())
    if slide_names is None:
        slide_names = sorted(['.'.join(fname.split('.')[:-1]) for fname in next(os.walk(slide_dir))[2] if fname.split('.')[-1] == 'mrxs'])


    for slide_name in tqdm.tqdm(slide_names):
        save_slide_masks(slide_name, slide_dir, output_dir, tile_size, level)

