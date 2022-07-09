import os
import glob
import numpy as np
import cv2
from PIL import Image
import pyvips
import tqdm
import pdb



def tileWSI(base_dir, save_dir, slide_level=4, tile_size=1024):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_names = sorted(glob.glob(os.path.join(base_dir, '*.mrxs')))
    file_names = list()
    file_names.append(os.path.join(os.path.split(base_dir)[0], 'amy-def', 'XE19-010_1_AmyB_1.mrxs'))

    files_to_contours = dict()
    files_to_tiles = dict()

    for file_name in tqdm.tqdm(file_names[:1]):
        img_name = '.'.join(os.path.split(file_name)[-1].split('.'))
        img_vips, img_vips_ds = [pyvips.Image.new_from_file(file_name, level=level) for level in (0, slide_level,)]

        img_arr_ds = np.ndarray(
            buffer=img_vips_ds.write_to_memory(),
            dtype=np.uint8,
            shape=tuple(map(lambda k: img_vips_ds.get(k), 'height width bands'.split())),
        )[..., :3]

        img_arr_gray = cv2.cvtColor(img_arr_ds, cv2.COLOR_RGB2GRAY)
        thresh, img_arr_gray = cv2.threshold(img_arr_gray.copy(), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)

        img_contours, _ = cv2.findContours(img_arr_gray.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        img_contours_by_area = sorted([(cv2.contourArea(contour), contour) for contour in img_contours], key=lambda t: -t[0])
        img_contours = list(map(lambda t: t[1], img_contours_by_area))

        selected_contours = list()
        print('Showing contours for selection (in reverse order of area): ')
        for k in range(len(img_contours)):
            fill = np.zeros(img_arr_gray.shape, img_arr_gray.dtype)
            cv2.drawContours(fill, [img_contours[k]], -1, 255, -1)
            Image.fromarray(fill).show()

            if get_resp('Select this contour (y/n): '):
                selected_contours.append(img_contours[k])
            if get_resp('Done selecting contours (y/n): '):
                break



        # group_fn = lambda n, a: list(zip(*tuple([a[i:len(a) + 1 + i - n] for i in range(n)])))


        files_to_contours[file_name] = selected_contours
        print(f'Selected {len(selected_contours)} contour(s) from file {file_name}')
        factor = 2 ** slide_level
        f_min, f_max = np.amin, np.amax

        fill = np.zeros(img_arr_gray.shape, img_arr_gray.dtype)
        cv2.drawContours(fill, selected_contours, -1, 255, -1)
        fill_roi = np.array([[f(a) for f in [f_min, f_max]] for a in fill.nonzero()]) * factor
        fill_roi[:, 1] += np.array([tile_size - 1] * 2)
        fill_roi //= tile_size

        _fill = np.zeros(img_arr_gray.shape, img_arr_gray.dtype)

        tile_points = [np.arange(*a) for a in fill_roi]
        tile_rois = [np.array([[i, j] for i, j in zip(a[:-1], a[1:])]) for a in tile_points]
        tile_rois = [(a * tile_size) // factor for a in tile_rois]
        tiles = dict([((i, j), fill[x1:x2, y1:y2]) for i, (x1, x2) in enumerate(tile_rois[0]) for j, (y1, y2) in enumerate(tile_rois[1])])
        tiles_nonzero = [k for k, v in tiles.items() if v.sum() > 0]
        for k, v in tiles.items():
            if v.sum() > 0:
                coords = np.array([a[i] for i, a in zip(k, tile_rois)])
                _fill[coords[0][0]:coords[0][1], coords[1][0]:coords[1][1]] = np.zeros(coords[1] - coords[0])
                # print(coords)

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


        pdb.set_trace()



        fill_roi = np.array([[f(a) for a in fill.nonzero()] for f in (np.amin, np.amax)]) * factor
        fill_roi[1] += np.array([tile_size - 1] * 2)
        fill_roi //= tile_size

        coords_roi = fill_roi * factor
        coords_roi[1] += np.array([tile_size - 1] * 2)
        coords_roi //= tile_size
        coords_x, coords_y = (*coords_roi.transpose((1, 0)),)


        # fill_roi = np.ndarray([[f(a) for a in fill.nonzero()] for f in (np.min, np.max)])


        grid_roi = [(i, j) for i in range(tile_size * (fill_roi[0][0] // tile_size), tile_size * (fill_roi[1][0] // tile_size), tile_size) for j in range(tile_size * (fill_roi[0][1] // tile_size), tile_size * (fill_roi[1][1] // tile_size), tile_size)]
        map_roi = [(t, fill[t[0]:t[0] + tile_size, t[1]:t[1] + tile_size].sum()) for t in grid_roi]

        pdb.set_trace()


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





        pdb.set_trace()


def get_resp(prompt, responses=('n', 'y')):
    resp = input(prompt)
    while resp not in responses:
        resp = input(prompt)
    return responses.index(resp)



if __name__ == '__main__':
    base_dir, save_dir = [f'/gladstone/finkbeiner/steve/work/data/npsad_data/gennadi/single_{i}' for i in 'slide output'.split()]

    results = tileWSI(base_dir, save_dir)
