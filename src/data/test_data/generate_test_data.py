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
# from requests import delete
import cv2
import pyvips as Vips
from tqdm import tqdm
# import pyfiglet
import argparse
import pdb
import skimage.io as io
# from utils import vips_utils
import matplotlib.pyplot as plt
from skimage.io import imread, imsave

import time

from threading import Thread


__author__ = 'Vivek Gopal Ramaswamy'

class GenerateTestData:
    """
    This is a class for preprocessing the WSI.

    """
    def __init__(self, base_dir, save_dir, tile_size=1024, slide_level=4):
        self.base_dir = base_dir
        self.save_dir = save_dir
        self.tile_size = tile_size
        self.slide_level = slide_level
        self.downscale_factor = 2 ** slide_level
        # self.file_name = ""

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

        savecroppath = os.path.join(savesubdir, f'{self.file_name}_x_{x}_y_{y}.png')

        # row is y, col is x
        if y + self.tile_size < orig_h and x + self.tile_size < orig_w:
            # TODO change to vips cropping
            crop = vips_orig_img.crop(x, y, 1024, 1024)
            crop.write_to_file(savecroppath)

        print("Thread Stopped ", i)


    def crop_slide(self, vips_orig_img, points, orig_w, orig_h):

        """Crop slide from points with cropsize"""
        savesubdir = os.path.join(self.save_dir, self.file_name)
        if not os.path.exists(savesubdir):
            os.makedirs(savesubdir, exist_ok=False)


        # Multithreading the tiling process
        list_threads = []
        for i, (x, y) in enumerate(points):
            t = Thread(target=self.crop_process, args=(i, x, y, vips_orig_img, savesubdir, orig_w, orig_h))
            list_threads.append(t)
            t.start()

    def getContour(self, thresh, vips_array, plot_countor=False):
        contours, hierarchy = cv2.findContours(thresh[1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pdb.set_trace()

        cnts = sorted([(cv2.contourArea(contour), contour) for contour in contours], key=lambda t: t[0])

        for cnt in cnts[::-1]:
            pdb.set_trace()
            frame = np.zeros(vips_array.shape[:-1])
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(frame, cnt, -1, (255, 0, 0), -1)

        if False:
            gray = cv2.cvtColor(vips_array, cv2.COLOR_BGR2GRAY)
            edged = cv2.Canny(gray, 30, 200)
            contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # cnt = max(contours, key=cv2.contourArea)

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


    def del_white_imgs(self, i, tile_img, intensity_threshold, del_img_count) :
        print("white_imgs thread started : ", i)
        x = io.imread(tile_img)

        dark_pixels = np.count_nonzero(x < int(intensity_threshold))

        light_pixels = np.count_nonzero(x > int(intensity_threshold))


        # 1024 * 1024 *3 / 96 percent white
        if light_pixels > 3019898:
            os.remove(tile_img)
            del_img_count = del_img_count + 1

        # print("Total deleted img tiles : ",  del_img_count)

    def stage1_filter(self):
        '''
        This function will filter out the tiles based on the ratio of
        dark pixels to light pixels.
        Parameters:
        tiled_images_path -- path to subtiled images
                    pdb.set_trace()

        The intesity ranges from 0 to 255. Default is 220. It means that
        The images with intesities less than 220 will be classified as
        dark and images above 220 will be classified as light.
        For effective filtering, the ratio of dark to white pixels should
        cat least be greater than 50 percent.
        Note* This filtering is also used on subtiles as well.
        '''

        del_img_count = 0
        intensity_threshold = 220
        tiled_folders = glob.glob(os.path.join(self.save_dir, "*"))

        # Multi
        for tile_folder in tiled_folders:
            tiled_images = glob.glob(os.path.join(tile_folder, "*.png"))
            for i, tile_img in enumerate(tiled_images):
                t = Thread(target=self.del_white_imgs, args=(i, tile_img, intensity_threshold, del_img_count))
                t.start()


    def tile_WSI(self):

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

         # TODO Remove this hardcoding later

        file_names = sorted(glob.glob(os.path.join(self.base_dir, '*.mrxs')))
        file_names = [os.path.join(self.base_dir, '../amy-def/XE19-010_1_AmyB_1.mrxs')]
        # image_names = ["/home/vivek/Datasets/AmyB/amyb_wsi/XE19-010_1_AmyB_1.mrxs"]

        for file_name in tqdm(file_names):

            # print(imagename)

            # Get file_name Ex:'XE19-010_1_AmyB_1'
            image_name = '.'.join(os.path.split(file_name)[-1].split('.'))
            # file_name = file_name[0].split("/")[-1]
            # self.file_name = file_name

            # Test
            vips_image, vips_thumbnail = [Vips.Image.new_from_file(file_name, level=level) for level in (0, self.slide_level)]
            # vips_img = Vips.Image.new_from_file(file_name, level=self.slide_level)
            # vips_img_orig = Vips.Image.new_from_file(imagename, level=0)
            vinfo = self.getVipsInfo(vips_image)
            orig_w, orig_h = int(vinfo['level[0].width']), int(vinfo['level[0].height'])

            # pdb.set_trace()
            vips_img = vips_thumbnail
            vips_img_orig = vips_image

            # vips_img = vips_img.crop(0, 1520,  5578,  13176-1700)

            vips_array = np.ndarray(buffer=vips_img.write_to_memory(), dtype=np.uint8, shape=(vips_img.height, vips_img.width, vips_img.bands))
            vips_array = vips_array[:,:,:3]


            # gray = cv2.cvtColor(vips_array, cv2.COLOR_RGB2GRAY)
            # gray = cv2.GaussianBlur(gray, (5, 5), 0)
            # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU)

            gray = cv2.cvtColor(vips_array, cv2.COLOR_RGB2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)
            contours, hierarchy = cv2.findContours(thresh[1], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # cv2.RETR_LIST
            # contours =

            pdb.set_trace()

            cnt, downscaled_w, downscaled_h = self.getContour(thresh, vips_array,plot_countor=False)

            points = self.get_points_in_contour(cnt, downscaled_w, downscaled_h)

            pdb.set_trace()



            # CROP
            # self.crop_slide(vips_img_orig, points, orig_w, orig_h)

        # Sleep for 180 seconds before next thread is started
        print('Start waiting')
        time.sleep(200)
        print("==========+Wait Over")
        self.stage1_filter()

if __name__ == '__main__':
    # result = pyfiglet.figlet_format("Generate Data", font="slant")
    # print(result)

    parser = argparse.ArgumentParser(description='Image Tiling for Model Prediction')
    parser.add_argument('base_dir',
                            help='Enter the path where the Test WSI images reside')
    parser.add_argument('save_dir',
                            help='Enter the path where you want the tiled image to reside')
    # args = parser.parse_args()
    # generate_test_data = GenerateTestData(base_dir=args.base_dir, save_dir=args.save_dir, slide_level=4, tile_size=1024)

    paths = [f'/gladstone/finkbeiner/steve/work/data/npsad_data/gennadi/single_{i}' for i in 'slide output'.split()]
    generate_test_data = GenerateTestData(base_dir=paths[0], save_dir=paths[1], slide_level=4, tile_size=1024)


    generate_test_data.tile_WSI()
