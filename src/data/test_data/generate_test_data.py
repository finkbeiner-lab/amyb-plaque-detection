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
        self.tilesize = tile_size
        self.file_name = ""
        # set max thread workers to be 1000
        self.workers = 1000
    
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
      
        savecroppath = os.path.join(savesubdir, f'{self.file_name}_x_{x}_y_{y}.png')

        # row is y, col is x
        if y + self.tilesize < orig_h and x + self.tilesize < orig_w:
            # TODO change to vips cropping
            crop = vips_orig_img.crop(x, y, 1024, 1024)
            crop.write_to_file(savecroppath)

        print("Thread Stopped ", i)

    def test_square(self, x):
        return (x*x)

    def crop_slide(self, vips_orig_img, points, orig_w, orig_h):

        """Crop slide from points with cropsize"""
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
           

    def del_white_imgs(self, i, tile_img, intensity_threshold, del_img_count) :
        print("white_imgs thread started : ", i)
        x = io.imread(tile_img)

        dark_pixels = np.count_nonzero(x < int(intensity_threshold))

        light_pixels = np.count_nonzero(x > int(intensity_threshold))

        print("light_pixels", light_pixels)
        

        # 1024 * 1024 *3 / 96 percent white
        if light_pixels > 3019898:
            print(" TRUE light_pixels", light_pixels)
            os.remove(tile_img)
            del_img_count = del_img_count + 1
                    
        # print("Total deleted img tiles : ",  del_img_count)

    def stage1_filter(self, img_name):
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
        tiled_folders = glob.glob(os.path.join(self.save_dir, img_name))
        
        # Multi
        for tile_folder in tiled_folders:
            tiled_images = glob.glob(os.path.join(tile_folder, "*.png"))
            exe = ThreadPoolExecutor(max_workers=self.workers)
            futures = [exe.submit(self.del_white_imgs, i, tile_img, intensity_threshold, del_img_count) for i, tile_img in enumerate(tiled_images)]
            done, not_done = wait(futures, return_when=ALL_COMPLETED)
            exe.shutdown()

                
    def tile_WSI(self):

        if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        
        threshold_dir = os.path.join(self.save_dir, "threshold")

        if not os.path.exists(threshold_dir):
                os.makedirs(threshold_dir)

        # start = timer()
        # # Normalization
        # normalizer = self.normalization()
        # end = timer()
        # print("Time Taken for Ref Normalization (minutes): ", (end - start) /60) # Time in seconds, e.g. 5.38091952400282

        imagenames = sorted(glob.glob(os.path.join(self.wsi_home_dir, '*.mrxs')))
        plt.figure(figsize=(10,10))
        plt.title("Thresholding")

        for imagename in tqdm(imagenames):

            print(imagename)

            # Get file_name Ex:'XE19-010_1_AmyB_1'
            file_name = imagename.split('.')
            file_name = file_name[0].split("/")[-1]
            self.file_name = file_name

            # # Test
            vips_img = Vips.Image.new_from_file(imagename, level=self.slide_level)
            # start = timer()
            vips_norm = vips_img 
            # normalizer.transform(vips_img)
            # end = timer()
            # print("Time Taken for Normalization: ", (end - start)/60)

            vips_img_orig = Vips.Image.new_from_file(imagename, level=0)
            vinfo = self.getVipsInfo(vips_img_orig)
            orig_w, orig_h = int(vinfo['level[0].width']), int(vinfo['level[0].height'])
           
            vips_img = vips_norm.crop(0, 1520,  5578,  13176-1700)
            
            vips_array = np.ndarray(buffer=vips_img.write_to_memory(), dtype=np.uint8, shape=(vips_img.height, vips_img.width, vips_img.bands))
            vips_array = vips_array[:,:,:3]

            gray = cv2.cvtColor(vips_array, cv2.COLOR_RGB2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Dialtion to fill holes
            thresh_dilation = 0
            for k in range(0, 3):
                thresh_dilation = cv2.dilate(thresh[1], None, iterations=k + 1)
            result_img = [vips_array, thresh_dilation]


            for j in range(2):
                plt.subplot(1, 2, j+1)
                plt.imshow(result_img[j])
            save_img =  file_name + ".png"
            plt.savefig(os.path.join(threshold_dir, save_img))
            
            cnt, downscaled_w, downscaled_h = self.getContour(thresh_dilation, vips_array,plot_countor=False)

            points = self.get_points_in_contour(cnt, downscaled_w, downscaled_h)
            
           
            # CROP
            self.crop_slide(vips_img_orig, points, orig_w, orig_h)
        
            # Sleep for 180 seconds before next thread is started
            print('Start waiting')
            time.sleep(15)
            print("==========+Wait Over")
            self.stage1_filter(file_name)
            src = os.path.join(self.save_dir, file_name)
           
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

    generate_test_data.tile_WSI()
    


    
    