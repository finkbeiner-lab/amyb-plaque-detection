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

    return thresh, mask, list(map(lambda t: t[0], sorted([(contour, cv2.contourArea(contour)) for contour in contours], key=lambda t: -t[1])))

def getROI(arr, roi):
    assert roi.shape == (2,) * 2 and (roi[:, 1] - roi[:, 0] > 0).all()
    return arr.copy()[roi[0, 0]:roi[0, 1], roi[1, 0]:roi[1, 1], ...]

def setROI(arr, sub_arr, roi):
    assert roi.shape == (2,) * 2 and (roi[:, 1] - roi[:, 0] > 0).all()
    assert not (sub_arr.shape[:2] - (roi[:, 1] - roi[:, 0])).any()
    arr = arr.copy()
    arr[roi[0, 0]:roi[0, 1], roi[1, 0]:roi[1, 1], ...] = sub_arr
    return arr

def blank_uint8(x, shape):
    return np.ones(shape, dtype=np.uint8) * np.array(x).astype(np.uint8)

def roi_to_tile_roi(roi, tile_size):
    assert roi.shape == (2,) * 2 and (roi[:, 1] - roi[:, 0] > 0).all()
    roi = roi.copy()
    roi[:, 1] += np.array([tile_size - 1] * 2)
    roi //= tile_size
    return roi



def roi_to_tiles(roi, tile_size, ratio=1):
    tile_roi = roi_to_tile_roi(roi * ratio, tile_size)
    tiles = [(i, j) for i in range(*tile_roi[0]) for j in range(*tile_roi[1])]
    tile_rois = np.array([[list(range(k, k + 2)) for k in item] for item in tiles])
    return tile_roi, tiles, tile_rois



def tileWSI(base_dir, save_dir, slide_level=4, tile_size=1024, top_n=1):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_names = sorted(glob.glob(os.path.join(base_dir, '*.mrxs')))
    file_names = list()
    file_names.append(os.path.join(os.path.split(base_dir)[0], 'amy-def', 'XE19-010_1_AmyB_1.mrxs'))

    files_to_contours = dict()
    files_to_tiles = dict()



    for file_name in tqdm.tqdm(file_names[:1]):
        img_vips, img_vips_ds = [pyvips.Image.new_from_file(file_name, level=level) for level in (0, slide_level,)]
        img_arr = np.ndarray(
            buffer=img_vips_ds.write_to_memory(),
            dtype=np.uint8,
            shape=tuple(map(lambda k: img_vips_ds.get(k), 'height width bands'.split())),
        )[..., :3]
        img_arr_gray = cv2.cvtColor(img_arr.copy(), cv2.COLOR_RGB2GRAY)

        _, _, contours = get_mask_contours(img_arr_gray)

        # selected_contours = list()
        # print('Showing contours for selection (in reverse order of area): ')
        #
        contours = contours[:top_n]
        for contour in contours:
            arr = np.zeros(img_arr_gray.shape, img_arr_gray.dtype)
            cv2.drawContours(arr, [contour], -1, 255, -1)

            Image.fromarray(arr).show()

        #     if get_resp('Select this contour (y/n): '):
        #         selected_contours.append(contour)
        #     if get_resp('Done selecting contours (y/n): '):
        #         break


        print(f'Selected {len(contours)} contour(s) from file {file_name}')


        factor = 2 ** slide_level

        # Instantiate numpy arrays to draw the slide's ROI mask, ROI bbox, and tiled ROIs.
        fill_mask, fill_roi, fill_tiles = [np.zeros(img_arr_gray.shape, img_arr_gray.dtype) for _ in range(3)]

        # Draw ROI mask
        cv2.drawContours(fill_mask, contours, -1, 255, -1)

        # Compute ROI bbox from ROI mask
        mask_roi = np.array([[f(a) for f in [np.amin, np.amax]] for a in fill_mask.nonzero()])

        # Draw ROI bbox
        # fill_roi = setROI(fill_roi, np.ones(mask_roi[:, 1] - mask_roi[:, 0]) * 255, mask_roi)
        fill_roi[mask_roi[0, 0]:mask_roi[0, 1], mask_roi[1, 0]:mask_roi[1, 1]] = np.ones(mask_roi[:, 1] - mask_roi[:, 0]) * 255

        # Compute tiles' ROI (from upsampled coordinates); broadcast to per-tile ROIs; rescale for downsampled tiles
        tile_roi, tiles, tile_rois = roi_to_tiles(mask_roi, tile_size, ratio=factor)
        tile_rois *= tile_size
        tile_rois_ds = tile_rois // factor
        tile_map = [(k, v) for k, v in zip(tiles, tile_rois_ds)]
        tile_map = [(k, getROI(fill_mask, v)) for k, v in tile_map]

        # tile_roi = roi_to_tile_roi(mask_roi * factor, tile_size)
        # tile_rois = dict([((i, j), np.array([[k, k + 1] for k in (i, j)]) * tile_size) for i in range(*tile_roi[0]) for j in range(*tile_roi[1])])
        # tile_rois_ds = dict([(k, v // factor) for k, v in tile_rois.items()])
        # tile_map = dict([(k, fill_mask[v[0, 0]:v[0, 1], v[1, 0]:v[1, 1]]) for k, v in tile_rois_ds.items()])

        # tile_rois = [(a * tile_size) // factor for a in tile_points]
        # tiles = dict([((i, j), fill_mask[x1:x2, y1:y2]) for i, (x1, x2) in enumerate(tile_rois[0]) for j, (y1, y2) in enumerate(tile_rois[1])])

        # Filter for nonzero tiles
        tiles_nonzero = [k for k, v in tile_map if v.sum() > 0]

        for k in tiles_nonzero:
            coords = tile_rois_ds[k]
            # fill_tiles = setROI(fill_tiles, np.ones(coords[:, 1] - coords[:, 0]) * 255, coords)
            fill_tiles[coords[0, 0]:coords[0, 1], coords[1, 0]:coords[1, 1]] = np.ones(coords[:, 1] - coords[:, 0]) * 255


        fill = np.concatenate([a[..., None] for a in [fill_mask, fill_tiles, fill_roi]], axis=-1)

        Image.fromarray(fill).show()

        pdb.set_trace()

        # tile_rois = [np.concatenate([a[:-1], a[1:]], axis=0)]
        # tile_rois = lambda i, j: np.array([tile_points[0][i:i + 2], tile_points[1][j:j + 2]])
        # _tile_rois = []
        # _tile_rois = np.array([group_fn(2, a) for a in tile_points])
        # tile_map = dict([((i, j), (tile_rois(i, j) * tile_size) // factor) for i in range(len(tile_points[0]) - 1) for j in range(len(tile_points[1]) - 1)])
        # tile_map = dict([(k, fill[v[0, 0]:v[0, 1], v[1, 0]:v[1, 1]]) for k, v in tile_map.items()])
        # tile_nonzero = [k for k, v in tile_map.items() if v.sum() > 0]


        # tile_dict = [((tile_points[0][i], tile_points[1][j]), fill[]) for i in range(len(tile_points[0])) for j in range(len(tile_points[1]))]
        # tile_dict = [((xk, yk), [list(xv), list(yv)]) for xk, xv in zip(tile_points[0], tile_coords[0]) for yk, yv in zip(tile_points[1], tile_coords[1])]
        # tile_dict = [ for i, j in ]
        # tile_sum_dict = [(k, fill[v[0]]) for k, v in tile_dict]


        # tile_coords = [(a * tile_size) // factor for a in tile_points]
        # tile_grid = [ for xi in tile_points[0]]
        # tile_dict = dict([((xi, xj), fill[xi:xj, yi:yj].sum()) for xi, xj in zip(tile_coords[0][:-1], tile_coords[0][1:]) for yi, yj in zip(tile_coords[1][:-1], tile_coords[1][1:])])
        # tile_nonzero =

        # def getROI(x, y, w, h, ds=(2 ** slide_level)):
        #     return [[x // ds, y // ds], [(x + w) // ds, (y + h) // ds]]
        #
        # img_height, img_width = tuple(map(lambda k: img_vips.get(k), 'height width'.split()))
        # img_tiles = [getROI(*(i, j), *([tile_size] * 2)) for i in range(0, img_width, tile_size) for j in range(0, img_height, tile_size)]
        # img_tiles = [tile for tile in img_tiles if True]
        #
        # pdb.set_trace()
        #
        # keep_tiles = list()
        # for xi, yi in img_tiles:
        #     xii, yii = [_ + tile_size for _ in (xi, yi)]
        #     xi //= img_scale
        #     yi //= img_scale
        #     xii //= img_scale
        #     yii //= img_scale
        #     print(xi, yi, xii, yii)

        # def tile_roi_to_points(roi):
        #     assert roi.shape == (2,) * 2 and (roi[:, 1] - roi[:, 0] > 0).all()
        #     roi = roi.copy()
        #     roi[:, 1] += np.array([1] * 2)
        #     points = [np.concatenate([t[..., None] for t in [a[:-1], a[1:]]], axis=1) for a in [np.arange(*a) for a in roi]]
        #     return points





def get_resp(prompt, responses=('n', 'y')):
    resp = input(prompt)
    while resp not in responses:
        resp = input(prompt)
    return responses.index(resp)



if __name__ == '__main__':
    base_dir, save_dir = [f'/gladstone/finkbeiner/steve/work/data/npsad_data/gennadi/single_{i}' for i in 'slide output'.split()]

    results = tileWSI(base_dir, save_dir)
