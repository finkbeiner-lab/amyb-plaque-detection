""" This code is called in optimized_generate_predictions_tiles.py file under inference folder"""

from typing import Any, List, Optional, Tuple
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

# Check if OpenSlide support is available
#if 'openslide' in pyvips.foreign.find_loaders():
#    print("OpenSlide support is available in pyvips.")
#else:
#    print("OpenSlide support is NOT available in pyvips.")




def tile_as_tensor(slide, tile):
    device = torch.device('cuda', 0)
    return (torch.as_tensor(slide.crop(*tile).numpy()).permute(2, 0, 1) / 255).to(device)

def crop_tile(slide, out_dir, tile, ext='png'):
    """
    Crops a portion of the slide based on the specified tile and saves or returns the result.

    Args:
        slide: The slide object (e.g., pyvips.Image) to crop from.
        out_dir (str): Output directory where the cropped tile will be saved (filename will be generated).
        tile (tuple): The coordinates (x, y, width, height) for cropping the slide.
        ext (str): The file extension for the output image (default is 'png').

    Returns:
        tuple: A tuple containing the output filename and the cropped tile data as a numpy array.
    """
    # Generate the output filename based on the tile coordinates and extension
    out_file = out_dir + '_'.join(map(lambda _: '_'.join(_), zip('xy', map(str, tile[:2])))) + f'.{ext}'
    # Crop the slide using the tile coordinates and convert to numpy array
    crop = slide.crop(*tile).numpy()
    # Uncomment to save the cropped tile to the disk
    # ToPILImage()(crop).save(out_file)
    # Return the output filename and the cropped image
    return out_file, crop

def crop_lambda(out_dir, slide_name, ext='png'):
    """
    Returns a lambda function that crops tiles from a slide and returns the cropped data.

    Args:
        out_dir (str): Output directory path (not used, slide_name is used instead).
        slide_name (str): Used as the output directory prefix for cropped tile filenames.
        ext (str): Image file extension for output files (default: 'png').

    Returns:
        Callable: A lambda function that takes (slide, tiles) and returns a list of
                  (filename, cropped image array) tuples for each tile.
    """
    out_dir = slide_name  # Override out_dir with slide_name
    return lambda slide, tiles: [crop_tile(slide, out_dir, tile, ext=ext) for tile in tiles]


def slide_tile_map(slide_name, slide_dir, tile_dir, size, f=None):
    """
    Loads a slide and its corresponding tile coordinates, applies transformation 
    to tile coordinates, and returns processed slide and tiles using the function `f`.

    Args:
        slide_name (str): Name of the slide file (without extension).
        slide_dir (str): Directory containing slide files (.mrxs).
        tile_dir (str): Directory containing corresponding tile .npy files.
        size (tuple): Tile size in pixels, e.g., (1024, 1024).
        f (callable, optional): A function that takes (slide, tiles) and returns transformed data.
                                Defaults to a function that returns the inputs unchanged.

    Returns:
        Result of applying function `f` to (slide, tiles).
    """
    if f is None:
        f = lambda slide, tiles: (slide, tiles)

    slide_path = os.path.join(slide_dir, f'{slide_name}.mrxs')
    tile_path = os.path.join(tile_dir, f'{slide_name}.tiles.npy')

    # Load slide using pyvips and limit to RGB channels
    slide = pyvips.Image.new_from_file(slide_path, level=0)[:3]
    print(slide.width, slide.height)

    # Load tile coordinates and scale them based on tile size
    tiles = np.load(tile_path, allow_pickle=False)
    tiles = np.concatenate([tiles + i for i in range(2)], axis=1) * np.array(size * 2)

    # Ensure tile coordinates are within image bounds
    tiles = np.minimum(np.maximum(tiles, 0), np.array([slide.width, slide.height] * 2))
    tiles[:, 2:] -= tiles[:, :2]

    # Sanity check: no negative coordinates or zero-sized tiles
    assert (tiles[:, :2] >= 0).all() and (tiles[:, 2:] > 0).all()
    return f(slide, tiles)


if __name__ == '__main__':
    slide_dir = '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/amy-def-mfg-test'
    #slide_dir = '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/Full-Minerva-Data/AmyB-MFG'
    csv_test_amyb_mdf = pd.read_csv("/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Metadata/overlap_with_metadata_srna_flag/AmyB-MFG_hasClinicalDataflag_hassnRNAflag.csv")
    vips_img_dir_new = csv_test_amyb_mdf[csv_test_amyb_mdf["hasClinicalData"]==True]["path"].values
    tile_dir = '/gladstone/finkbeiner/steve/work/data/npsad_data/slide_masks'
    #tile_dir = '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/amyb_wsi-backup/test-patients/images-npy'
    out_dir = '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/amyb_wsi/test/images'
    tile_size = (1024, 1024)
    # watch out for '.DS_Store'
    slide_names = ['.'.join(file.split('.')[:-1]) for file in next(os.walk(slide_dir))[2]]
    #slide_names = [x.split("/")[-1].split(".")[0] for x in vips_img_dir_new] 
    #slide_names = ['XE16-057_1_AmyB_1', 'XE10-020_1_AmyB_1', 'XE14-051_1_AmyB_1', 'XE12-037_1_AmyB_1', 'XE16-027_1_AmyB_1', 'XE17-013_1_AmyB_1', 'XE15-007_1_AmyB_1', 'XE09-006_1_AmyB_1', 'XE07-060_1_AmyB_1', 'XE10-019_1_AmyB_1', 'XE10-042_1_AmyB_1', 'XE12-007_1_AmyB_1', 'XE10-006_1_AmyB_1', 'XE09-035_1_AmyB_1', 'XE17-014_1_AmyB_1', 'XE13-017_1_AmyB_1', 'XE15-022_1_AmyB_1', 'XE17-059_1_AmyB_1', 'XE10-018_1_AmyB_1', 'XE18-040_1_AmyB_1', 'XE12-036_1_AmyB_1', 'XE08-047_1_AmyB_1', 'XE17-030_1_AmyB_1', 'XE14-004_1_AmyB_1', 'XE10-005_1_AmyB_1', 'XE07-067_1_AmyB_1', 'XE12-031_1_AmyB_1', 'XE12-009_1_AmyB_1', 'XE16-014_1_AmyB_1', 'XE09-063_1_AmyB_1', 'XE12-012_1_AmyB_1', 'XE17-022_1_AmyB_1', 'XE17-048_1_AmyB_1', 'XE13-018_1_AmyB_1', 'XE10-026_1_AmyB_1', 'XE10-033_1_AmyB_1', 'XE08-033_1_AmyB_1', 'XE17-039_1_AmyB_1', 'XE17-029_1_AmyB_1', 'XE13-028_1_AmyB_1', 'XE08-018_1_AmyB_1', 'XE14-047_1_AmyB_1', 'XE16-023_1_AmyB_1', 'XE17-010_1_AmyB_1', 'XE12-042_1_AmyB_1', 'XE18-001_1_AmyB_1', 'XE12-023_1_AmyB_1', 'XE14-037_1_AmyB_1', 'XE11-025_1_AmyB_1', 'XE18-004_1_AmyB_1', 'XE12-010_1_AmyB_1', 'XE08-016_1_AmyB_1', 'XE09-056_1_AmyB_1', 'XE12-016_1_AmyB_1', 'XE14-033_1_AmyB_1', 'XE07-057_1_AmyB_1', 'XE11-027_1_AmyB_1', 'XE17-065_1_AmyB_1', 'XE07-056_1_AmyB_1', 'XE08-015_1_AmyB_1', 'XE09-013_1_AmyB_1', 'XE16-033_1_AmyB_1', 'XE13-007_1_AmyB_1']
    print(slide_names[0])
    for slide_name in slide_names:
        num_tiles = slide_tile_map(slide_name, slide_dir, tile_dir, tile_size, f=crop_lambda(out_dir, slide_name))
        print(num_tiles[0])
        break
