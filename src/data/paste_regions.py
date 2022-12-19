import pyvips
import glob
import os
import numpy as np
import skimage


def vips_to_np(vips_img):
    return np.ndarray(buffer=vips_img.write_to_memory(), dtype=np.uint8, shape=(vips_img.height, vips_img.width, vips_img.bands))


def vips_fetch_map(vips_img, rois):
    """
    vips_img: a pyvips.Image object
    rois:     a numpy.ndarray of shape (None, 2, 2) containing [[x1, y1], [x2-x1, y2-y1]]
    returns:  a pyvips.Image objct with
    """

    assert len(rois.shape) == 3 and rois.shape[1] == rois.shape[2] == 2

    coords, offsets = rois[:, 0], rois[:, 1]
    bounds = np.array([coords.min(axis=0), (coords + offsets).max(axis=0)])
    assert (bounds[0] >= 0).all() and (bounds[1, 0] < vips_img.width).all() and (bounds[1, 1] < vips_img.height).all()

    fetched = list()
    for coords, offset in rois:
        fetched.append(vips_img.crop(*coords, *offset))

    return fetched


def vips_paste_map(vips_img, regions, rois, copy=False):
    """
    vips_img: a pyvips.Image object
    regions:  an iterator of pyvips.Image objects to paste into the image
    rois:     a numpy.ndarray of shape (None, 2, 2) containing [[x1, y1], [x2-x1, y2-y1]]
    returns:  a pyvips.Image object with each region pasted sequentially to each respective ROI
    """

    assert len(rois.shape) == 3 and rois.shape[0] == len(regions) and rois.shape[1] == rois.shape[2] == 2

    coords, offsets = rois[:, 0], rois[:, 1]
    bounds = np.array([coords.min(axis=0), (coords + offsets).max(axis=0)])
    assert (bounds[0] >= 0).all() and (bounds[1, 0] < vips_img.width).all() and (bounds[1, 1] < vips_img.height).all()

    if copy:
        vips_img = vips_img.copy()

    for region, (coords, offset) in zip(regions, rois):
        assert region.width == offset[0] and region.height == offset[1]
        assert region.bands == vips_img.bands and region.format == vips_img.format
        vips_img = vips_img.insert(region, *coords)

    return vips_img


def prepare_to_paste(imgs_to_paste):
    imgs = list()
    rois = list()
    for file_name in imgs_to_paste:
        imgs.append(pyvips.Image.new_from_file(file_name))

        img_name = os.path.split(file_name)[1]
        img_name = '.'.join(img_name.split('.')[:-1])
        img_attrs = img_name.split('_')[::-1]
        assert img_attrs[4] == 'x' and img_attrs[2] == 'y' and img_attrs[3].isdigit() and img_attrs[1].isdigit()
        coords = np.array([[int(img_attrs[3]), int(img_attrs[1])], [imgs[-1].width, imgs[-1].height]])

        rois.append(coords)

    return imgs, np.array(rois)


if __name__ == '__main__':
    detections_folder = '/Users/gennadiryan/Documents/gladstone/projects/gennadi_evaluation/amyb-plaque-detection/src/data/test_data/detections'
    imgs_to_paste = os.path.join(detections_folder, '*.png')
    imgs_to_paste = glob.glob(imgs_to_paste)

    imgs, rois = prepare_to_paste(imgs_to_paste)
    assert (rois[:, 1, :] == 1024).all() # offsets should all be equal to tile size of 1024

    slide_folder = '/Users/gennadiryan/Documents/gladstone/projects/slide_utils/slides/mrxs/'
    vips_img = pyvips.Image.new_from_file(os.path.join(slide_folder, 'XE19-010_1_AmyB_1.mrxs'), level=0)[:3]

    processed = vips_paste_map(vips_img, imgs, rois)
