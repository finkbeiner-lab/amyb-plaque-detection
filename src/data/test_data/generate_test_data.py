import sys
#TODO change hardcoding
sys.path.append('../../../')
from concurrent.futures import process
import os
import glob
from posixpath import dirname
import re
from turtle import pd
from cv2 import THRESH_BINARY_INV, THRESH_OTSU
import numpy as np
from requests import delete
import cv2
import pyvips as Vips
from tqdm import tqdm
import pyfiglet
import argparse
import pdb
import skimage.io as io
from src.utils import vips_utils, normalize
import matplotlib.pyplot as plt
from skimage.io import imread, imsave

import time
from timeit import default_timer as timer
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import subprocess
from PIL import Image



__author__ = 'Vivek Gopal Ramaswamy'

class GenerateTestData:
    """
    This is a class for preprocessing the WSI.

    """
    def __init__(self, wsi_home_dir, save_dir, ref_slide_path, slide_level, downscale_factor, tile_size):
        self.wsi_home_dir = wsi_home_dir
        self.save_dir = save_dir
        self.ref_slide_path = ref_slide_path
        self.slide_level = slide_level
        self.downscale_factor = downscale_factor
        self.tile_size = tile_size
        self.file_name = ""
        # set max thread workers to be 1000
        self.workers = 100
    
    def normalization(self):
        print("Init Normalization")
        ref_image = Vips.Image.new_from_file(self.ref_slide_path, level=self.slide_level)
        normalizer = normalize.Reinhard()
        normalizer.fit(ref_image)
        return normalizer

    def get_points_in_contour(self, contour, downscaled_w ,downscaled_h, stride=64):
        # 1024/16 = 64  Stride calculation
        points = []
        for x in range(0, downscaled_w, stride): 
            for y in range(0, downscaled_h, stride): 
                inside = cv2.pointPolygonTest(contour, (x, y), False) 
                if inside >= 0: 
                    points.append((x * self.downscale_factor, y * self.downscale_factor)) # points time scale factor print(f'Collected {len(self.points)} points') return self.points 
            
        return points
    
    def crop_process(self, i, x, y, vips_orig_img, savesubdir, orig_w, orig_h):
        
        print("Crop slide thread Started.", i)
        x1 = y * self.tile_size
        y1 = x * self.tile_size
      
        savecroppath = os.path.join(savesubdir, f'{self.file_name}_x_{x}_y_{y}.png')

        # row is y, col is x
        # if y + self.tilesize < orig_h and x + self.tilesize < orig_w:
            # TODO change to vips cropping
        crop = vips_orig_img.crop(x1, y1, 1024, 1024)
        crop.write_to_file(savecroppath)

        print("Thread Stopped ", i)

    def crop_slide(self, vips_orig_img, points, orig_w, orig_h):

        """Crop slide from points with cropsize"""
        pdb.set_trace()
        savesubdir = os.path.join(self.save_dir, self.file_name)
        if not os.path.exists(savesubdir):
            os.makedirs(savesubdir, exist_ok=False)
        
        exe = ThreadPoolExecutor(max_workers=self.workers)
        futures = [exe.submit(self.crop_process, i, x, y, vips_orig_img, savesubdir, orig_w, orig_h) for i, (x, y) in enumerate(points)]
        done, not_done = wait(futures, return_when=ALL_COMPLETED)
        exe.shutdown()
        
           
    def getContour(self, thresh, vips_array, plot_countor=False):
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key=cv2.contourArea)
        
        # TODO find good name for downscaled_h
        x,y,downscaled_w,downscaled_h = cv2.boundingRect(cnt)
        vips_array = cv2.rectangle(vips_array.copy(),(x, y),(x + downscaled_w,y + downscaled_h),(0,255,0),20)
        

        # TODO: remove the debug comments
        # print(x)
        # print(y)
        # print(x+downscaled_w)
        # print(y+downscaled_h)

        if plot_countor:
            plt.imshow(vips_array)
            plt.savefig('test.tif')
            plt.show()
        
        return cnt, downscaled_w, downscaled_h
    
    def getVipsInfo(self, vips_img):
        # # Get bounds-x and bounds-y offeset
        vfields = [f.split('.') for f in vips_img.get_fields()]
        vfields = [f for f in vfields if f[0] == 'openslide']
        vfields = dict([('.'.join(k[1:]), vips_img.get('.'.join(k))) for k in vfields])
        
        return vfields
    
    def getROI(self, arr, roi):
        """
        Args:
            arr: numpy array of dimension at least 2, from which the ROI is extracted
            roi: [[x1, x2], [y1, y2]]
        Returns:
            sub_arr: the subarray given by the provided ROI (not a copy; may be used to write to a subset of an image or array)
        """
        assert roi.shape == (2,) * 2 and (roi[:, 1] - roi[:, 0] > 0).all()
        return arr[roi[0, 0]:roi[0, 1], roi[1, 0]:roi[1, 1], ...]


    def visualize_rois(self, rois, fill, shape, dtype):
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
            self.getROI(fill_rois, roi).fill(np.array(fill, dtype))
        return fill_rois
    
    def get_mask_contours(self, x):
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
    
    def get_resp(self,prompt, responses=('n', 'y')):
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
    
    def roi_to_tile_roi(self, roi):
        """
        Args:
            roi: [[x1, x2], [y1, y2]]
            tile_size: size of the grid the ROI is measured against
        Returns:
            tile_roi: the minimal grid rectangle of step size tile_size which contains the ROI in its entirety
        """
        assert roi.shape == (2,) * 2 and (roi[:, 1] - roi[:, 0] > 0).all()
        tile_roi = roi.copy()
        tile_roi[:, 1] += np.array([self.tile_size - 1] * 2)
        tile_roi //= self.tile_size
        return tile_roi

    def roi_to_tiles(self, roi, ratio=1):
        """
        Args:
            roi: [[x1, x2], [y1, y2]]
            tile_size: size of the grid the ROI is measured against
            ratio: downscaling ratio of the ROI wrt. the original image to be tiled
        Returns:
            tile_roi: the minimal grid (step tile_size, in original image scale) which contains the ROI in its entirety
            tiles: a numpy array of shape (N, 2) listing the tiles occurring in the tile_roi
        """
        tile_roi = self.roi_to_tile_roi(roi * ratio)
        tiles = np.array([[i, j] for i in range(*tile_roi[0]) for j in range(*tile_roi[1])])
        return tile_roi, tiles
        
    def get_slide_tiles(self, slide_vips, tile_cond=None, interactive=False, top_n=1, visualize=False):
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
            tile_cond = lambda a, r: self.getROI(a, r).sum() > 0

        slide_arr = np.ndarray(buffer=slide_vips.write_to_memory(), dtype=np.uint8, shape=tuple(map(lambda k: slide_vips.get(k), 'height width bands'.split())))[..., :3]
        slide_arr = cv2.cvtColor(slide_arr, cv2.COLOR_RGB2GRAY)
        shape, dtype, fill, ratio = slide_arr.shape, slide_arr.dtype, 255, 2 ** self.slide_level

        thresh, _, contours = self.get_mask_contours(slide_arr)
        print(f'Total of {len(contours)} contours found at threshold value {thresh}.')
        contours = contours[:top_n]
        selected_contours = list()
        
        if interactive:
            print(f'Select from among {len(contours)} contours.')
            for contour in contours:
                fill_contour = np.zeros(shape, dtype)
                cv2.drawContours(fill_contour, [contour], -1, fill, -1)
                Image.fromarray(fill_contour).show()
                if self.get_resp('Select this contour (y/n): '):
                    selected_contours.append(contour)
                if self.get_resp('Done selecting contours (y/n): '):
                    break
            contours = selected_contours
            print(f'Total of {len(contours)} contours selected.')


        fill_mask = np.zeros(shape, dtype)
        cv2.drawContours(fill_mask, contours, -1, fill, -1)

        mask_roi = np.array([[f(a) for f in [np.amin, np.amax]] for a in fill_mask.nonzero()])
        tile_roi, tiles = self.roi_to_tiles(mask_roi, ratio=ratio)
        grid_rois = np.array([[list(range(k, k + 2)) for k in item] for item in tiles])

        selected_idxs = np.array([i for i, (_, v) in enumerate(zip(tiles, (grid_rois * self.tile_size) // ratio)) if tile_cond(fill_mask, v)])
        selected_tiles, selected_rois = [a[selected_idxs] for a in [tiles, grid_rois * self.tile_size]]

        visuals = None
        if visualize:
            fill_roi = self.visualize_rois([mask_roi], fill, shape, dtype)
            fill_tiles = self.visualize_rois(selected_rois // ratio, fill, shape, dtype)
            visuals = np.concatenate([a[..., None] for a in [fill_tiles, fill_tiles, fill_tiles]], axis=-1)

        return selected_tiles, selected_rois, thresh, visuals
           
                
    def process_slides(self):

        if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        
        threshold_dir = os.path.join(self.save_dir, "threshold")

        if not os.path.exists(threshold_dir):
                os.makedirs(threshold_dir)

        start = timer()
        # Normalization
        normalizer = self.normalization()
        end = timer()
        print("Time Taken for Ref Normalization (minutes): ", (end - start) /60) # Time in seconds, e.g. 5.38091952400282

        file_names = sorted(glob.glob(os.path.join(self.wsi_home_dir, '*.mrxs')))
       
        for file_name in file_names:
            print("Processing: ", file_name)
            img_name = os.path.split(file_name)[1]
            img_name = '.'.join(img_name.split('.')[:-1])
            self.file_name = img_name
            img_vips, img_vips_ds = [Vips.Image.new_from_file(file_name, level=level) for level in (0, self.slide_level,)]

            tiles, _, thresh, visuals = self.get_slide_tiles(img_vips_ds, interactive=True, visualize=False, top_n=1)
            Image.fromarray(visuals).save(os.path.join(threshold_dir, f'{img_name}_visuals_{thresh}.png'))

            vinfo = self.getVipsInfo(img_vips)
            orig_w, orig_h = int(vinfo['level[0].width']), int(vinfo['level[0].height'])

            self.crop_slide(img_vips, tiles, orig_w, orig_h)

            # Sleep for 180 seconds before next thread is started
            print('Start waiting')
            time.sleep(15)
           
if __name__ == '__main__':
    result = pyfiglet.figlet_format("Generate Data", font="slant")
    print(result)
    parser = argparse.ArgumentParser(description='Image Tiling for Model Prediction')
    parser.add_argument('wsi_home_dir',
                            help='Enter the path where the Test WSI images reside')
    parser.add_argument('save_dir',
                            help='Enter the path where you want the tiled image to reside')

    parser.add_argument('ref_slide_path',
                            help='Enter the path to the reference image for normalization ')
    args = parser.parse_args()

    generate_test_data = GenerateTestData(wsi_home_dir=args.wsi_home_dir, save_dir=args.save_dir, 
                                          ref_slide_path=args.ref_slide_path, slide_level=4, 
                                          downscale_factor=16, tile_size=1024)

    generate_test_data.process_slides()
    


    
    