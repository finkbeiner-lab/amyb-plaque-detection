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
import pandas as pd


def norm(x: Tensor):
    dims = tuple(range(1, len(x.size())))
    n = reduce(lambda a, b: a * b, x.size()[1:], 1)
    s1, s2 = [(x ** (i + 1)).sum(dim=tuple(range(1, len(x.size())))) for i in range(2)]
    return torch.as_tensor(n), s1 / n, s2 / n

def summarize(norms):
    n, s1, s2 = map(torch.stack, zip(*norms))
    factors = (n / n.sum()).unsqueeze(1)
    mean = (s1 * factors).sum(dim=0)
    std = ((s2 * factors).sum(dim=0) - (mean ** 2)).sqrt()
    return mean, std


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

def fill_mask(mask, k, r=0):
    mask = cv2.blur(mask, (k, k))
    return (mask > (255 * r)).astype(np.uint8) * 255

def close_mask(mask, k, i):
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k)), iterations=i)

def mask_contours(mask):
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return list(map(lambda t: t[0], sorted([(contour, cv2.contourArea(contour)) for contour in contours], key=lambda t: -t[1])))

def fill_contours(contours, shape):
    fill = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(fill, contours, -1, 255, -1)
    return fill


if __name__ == '__main__':
    #vips_img_dir = '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/amy-def-mfg-test'
    #vips_img_dir = '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/amy-def-mfg-test'
    out_dir = '/gladstone/finkbeiner/steve/work/data/npsad_data/slide_masks'
    csv_test_amyb_mdf = pd.read_csv("/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Metadata/overlap_with_metadata_srna_flag/AmyB-MFG_hasClinicalDataflag_hassnRNAflag.csv")
    vips_img_dir_new = csv_test_amyb_mdf[csv_test_amyb_mdf["hasClinicalData"]==True]["path"].values
    print(len(vips_img_dir_new))
    vips_img_names = os.listdir(out_dir)
    print(vips_img_names[:2])
    files_to_run = []
    for file in vips_img_dir_new:
        if file.split("/")[-1].split(".")[0]+".mask.png" in vips_img_names:
            files_to_run.append(file)
            
            
    print(len(files_to_run))
    
    
    
    out_dir_new = "/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/amyb_wsi/test-patients/images-npy"
    
    
    
    
    #vips_img_skip = '07-057 08-018 09-006 09-041 10-005 10-006 10-009 10-018 10-019 10-020 10-021 10-026 11-008 11-018 11-025 11-029 12-009 12-023 12-031 12-036 13-028 14-033 16-027'
    #vips_img_names = next(os.walk(vips_img_dir))[2]
    #vips_img_names = [file[2:2 + 6] for file in vips_img_names]
    #vips_img_names = sorted(list(set(vips_img_names).difference(vips_img_skip)))

    #vips_img_fnames = [os.path.join(vips_img_dir, f'XE{vips_img_name}_1_AmyB_1.mrxs') for vips_img_name in vips_img_names]
    vips_img_fnames = files_to_run

    tile_size = 128
    level = 3
    k1, i1, r1 = 9, 4, .01
    k2, i2 = 9, 4

    for i, fname in enumerate(vips_img_fnames):
        print(i)
        #name = os.path.join(out_dir, '.'.join(os.path.split(fname)[1].split('.')[:-1]))
        name = os.path.join(out_dir_new, '.'.join(os.path.split(fname)[1].split('.')[:-1]))
        tile_name, mask_name, viz_name = [f'{name}.{suffix}' for suffix in ('tiles.npy', 'mask.png', 'viz.png')]
        #print(name,tile_name, mask_name, viz_name )
        try:
            slide = pyvips.Image.new_from_file(fname)
        except pyvips.error.Error:
            pass
        # slide = get_cropped(slide, level)

        im = slide.numpy()[..., :3]
        gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        thresh, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        filled_mask = mask
        for _ in range(i1):
            filled_mask = fill_mask(filled_mask, k1, r1)
        filled_mask = close_mask(filled_mask, k2, i2)
        contours = mask_contours(filled_mask)
        filled_mask = fill_contours(contours[:1], filled_mask.shape)

        f_pos, f_neg = lambda _: _.sum() > 0, lambda _: _.sum() == 0
        tiles_pos, coords_neg = tile_mask(filled_mask, tile_size, f=f_pos)[1], tile_mask(filled_mask, tile_size, f=f_neg)[0]

        for y1, x1, y2, x2 in coords_neg:
            im[y1:y2, x1:x2] = 0
        mask_out = ToPILImage()(filled_mask)
        viz_out = ToPILImage()(im)

        np.save(tile_name, tiles_pos, allow_pickle=False)
        mask_out.save(mask_name)
        viz_out.save(viz_name)
   