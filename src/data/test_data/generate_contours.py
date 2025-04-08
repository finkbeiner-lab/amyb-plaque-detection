from collections import OrderedDict
import os
import glob
import numpy as np
import cv2
from PIL import Image
import pyvips
import tqdm
import pdb




def get_mask_contours(x):
    """
    Args:
        x: np.ndarray
    Returns:
        thresh: int (value used to threshold mask)
        mask: np.ndarray (thresholded mask)
        contours: List[np.ndarray] (list of contours enclosing mask in order of area)
    """
    assert x.dtype == np.uint8
    if len(x.shape) == 3:
        if x.shape[-1] == 1:
            x = x[..., 0]
        else:
            assert x.shape[-1] == 3
            x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    else:
        assert len(x.shape) == 2

    thresh, mask = cv2.threshold(x, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    return int(thresh), mask, list(map(lambda t: t[0], sorted([(contour, cv2.contourArea(contour)) for contour in contours], key=lambda t: -t[1])))

def getROI(arr, roi):
    """
    Args:
        arr: numpy array of dimension at least 2, from which the ROI is extracted
        roi: [[x1, x2], [y1, y2]]
    Returns:
        sub_arr: the subarray given by the provided ROI (not a copy; may be used to write to a subset of an image or array)
    """
    assert roi.shape == (2,) * 2 and (roi[:, 1] - roi[:, 0] > 0).all()
    return arr[roi[0, 0]:roi[0, 1], roi[1, 0]:roi[1, 1], ...]

def roi_to_tile_roi(roi, tile_size):
    """
    Args:
        roi: [[x1, x2], [y1, y2]]
        tile_size: size of the grid the ROI is measured against
    Returns:
        tile_roi: the minimal grid rectangle of step size tile_size which contains the ROI in its entirety
    """
    assert roi.shape == (2,) * 2 and (roi[:, 1] - roi[:, 0] > 0).all()
    tile_roi = roi.copy()
    tile_roi[:, 1] += np.array([tile_size - 1] * 2)
    tile_roi //= tile_size
    return tile_roi

def roi_to_tiles(roi, tile_size, ratio=1):
    """
    Args:
        roi: [[x1, x2], [y1, y2]]
        tile_size: size of the grid the ROI is measured against
        ratio: downscaling ratio of the ROI wrt. the original image to be tiled
    Returns:
        tile_roi: the minimal grid (step tile_size, in original image scale) which contains the ROI in its entirety
        tiles: a numpy array of shape (N, 2) listing the tiles occurring in the tile_roi
    """
    tile_roi = roi_to_tile_roi(roi * ratio, tile_size)
    tiles = np.array([[i, j] for i in range(*tile_roi[0]) for j in range(*tile_roi[1])])
    return tile_roi, tiles

def visualize_rois(rois, fill, shape, dtype):
    """
    Args:
        rois: rois to draw onto the blank image
        fill: character to fill the rois with
        shape: use this shape for the blank image
        dtype: use this dtype for the blank image
    Returns:
    """
    fill_rois = np.zeros(shape, dtype)
    for roi in rois:
        getROI(fill_rois, roi).fill(np.array(fill, dtype))
    return fill_rois

def get_slide_tiles(slide_vips, slide_level, tile_size, tile_cond=None, interactive=False, top_n=1, visualize=False):
    """
    Args:
        slide_vips: vips image of the slide
        slide_level: slide_vips is downsampled slide at scale 2 ** slide_level
        tile_size: tile size wrt. the original slide
        tile_cond: a boolean-returning callable which filters the tiles in the slide masks' ROI into background and foreground
        interactive: allow users to interactively select the most accurate slide contour masks
        top_n: only keep top_n of the contour masks (precedes interactive selection)
        visualize: return additional visualizations (note: the function only incurs visualization overhead if this option is selected)
    Returns:
        selected_tiles: a numpy array of shape (N, 2) listing the selected tiles ([x, y] -> [[x * tile_size, (x + 1) * tile_size], [y * tile_size, (y + 1) * tile_size]])
        selected_rois: a numpy array of shape (N, 2, 2) listing the ROIs of the selected tiles wrt. the original slide in [[x1, x2], [y1, y2]] format
        thresh: the threshold value used by get_mask_contours to find the contour masks
        visuals: an optional binary (0/255) RGB array showing the overlaps of the main slide masks, the slide masks' ROI, and the selected tiles
    """
    if tile_cond is None:
        tile_cond = lambda a, r: getROI(a, r).sum() > 0

    slide_arr = np.ndarray(buffer=slide_vips.write_to_memory(), dtype=np.uint8, shape=tuple(map(lambda k: slide_vips.get(k), 'height width bands'.split())))[..., :3]
    slide_arr = cv2.cvtColor(slide_arr, cv2.COLOR_RGB2GRAY)
    shape, dtype, fill, ratio = slide_arr.shape, slide_arr.dtype, 255, 2 ** slide_level

    thresh, _, contours = get_mask_contours(slide_arr)
    print(f'Total of {len(contours)} contours found at threshold value {thresh}.')
    contours = contours[:top_n]
    selected_contours = list()
    
    if interactive:
        print(f'Select from among {len(contours)} contours.')
        for contour in contours:
            fill_contour = np.zeros(shape, dtype)
            cv2.drawContours(fill_contour, [contour], -1, fill, -1)
            Image.fromarray(fill_contour).show()
            if get_resp('Select this contour (y/n): '):
                selected_contours.append(contour)
            if get_resp('Done selecting contours (y/n): '):
                break
        contours = selected_contours
        print(f'Total of {len(contours)} contours selected.')


    fill_mask = np.zeros(shape, dtype)
    cv2.drawContours(fill_mask, contours, -1, fill, -1)

    mask_roi = np.array([[f(a) for f in [np.amin, np.amax]] for a in fill_mask.nonzero()])
    tile_roi, tiles = roi_to_tiles(mask_roi, tile_size, ratio=ratio)
    grid_rois = np.array([[list(range(k, k + 2)) for k in item] for item in tiles])

    selected_idxs = np.array([i for i, (_, v) in enumerate(zip(tiles, (grid_rois * tile_size) // ratio)) if tile_cond(fill_mask, v)])
    selected_tiles, selected_rois = [a[selected_idxs] for a in [tiles, grid_rois * tile_size]]

    visuals = None
    if visualize:
        fill_roi = visualize_rois([mask_roi], fill, shape, dtype)
        fill_tiles = visualize_rois(selected_rois // ratio, fill, shape, dtype)
        visuals = np.concatenate([a[..., None] for a in [fill_mask, fill_roi, fill_tiles]], axis=-1)

    return selected_tiles, selected_rois, thresh, visuals



def process_slides(base_dir, save_dir, slide_level=4, tile_size=1024):
    """
    Args:
        base_dir: directory to search for *.mrxs
        save_dir: directory to dump *_tiles.txt, *_visuals.png
        slide_level: use downsampled slide at scale 2 ** slide_level
        tile_size: tile the (full size) slide in multiples of 1024
    Returns:
        None
    """

    file_names = sorted(glob.glob(os.path.join(base_dir, '*.mrxs')))
    selected_tiles = list()

    for file_name in file_names[:1]:
        img_name = os.path.split(file_name)[1]
        img_name = '.'.join(img_name.split('.')[:-1])
        img_vips, img_vips_ds = [pyvips.Image.new_from_file(file_name, level=level) for level in (0, slide_level,)]

        tiles, _, thresh, visuals = get_slide_tiles(img_vips_ds, slide_level, tile_size, interactive=True, visualize=True, top_n=3)

        tile_selection = '\n'.join([','.join(map(str, t)) for t in tiles])
        with open(os.path.join(save_dir, f'{img_name}_tiles.txt'), 'w') as fh:
            fh.write(tile_selection)

        Image.fromarray(visuals).save(os.path.join(save_dir, f'{img_name}_visuals_{thresh}.png'))



def get_resp(prompt, responses=('n', 'y')):
    """
    Args:
        prompt: message to show at prompt time
        responses: valid responses this function will accept
    Returns:
        index of the first valid response to be received
    """

    resp = input(prompt)
    while resp not in responses:
        resp = input(prompt)
    return responses.index(resp)



if __name__ == '__main__':
    base_dir = '/Users/gennadiryan/Documents/gladstone/projects/slide_utils/slides/mrxs'
    save_dir = base_dir + '_out'

    process_slides(base_dir, save_dir)
