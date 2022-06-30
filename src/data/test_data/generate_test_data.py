import sys
#TODO change hardcoding
sys.path.insert(1, '/home/vivek/Projects/amyb-plaque-detection')
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
from src.utils import vips_utils
import matplotlib.pyplot as plt
import shutil
from skimage.io import imread, imsave
import shutil
import natsort
import openslide


__author__ = 'Vivek Gopal Ramaswamy'

class GenerateTestData:
    """
    This is a class for preprocessing the WSI.

    """
    def __init__(self, wsi_home_dir, save_dir, slide_level, downscale_factor, tile_size):
        self.wsi_home_dir = wsi_home_dir
        self.save_dir = save_dir
        self.slide_level = slide_level
        self.downscale_factor = 16
        self.tilesize = tile_size
        self.file_name = ""

    def get_points_in_contour(self, contour, downscaled_w ,downscaled_h, stride=64):
        # 1024/16 = 64  Stride calculation
        points = []
        for x in range(0, downscaled_w, stride): 
            for y in range(0, downscaled_h, stride): 
                inside = cv2.pointPolygonTest(contour, (x, y), False) 
                if inside >= 0: 
                    points.append((x * self.downscale_factor, y * self.downscale_factor)) # points time scale factor print(f'Collected {len(self.points)} points') return self.points 
            
        return points

    def crop_slide(self, vips_orig_img, slide, points, orig_w, orig_h):

        """Crop slide from points with cropsize"""
        # name = file.split('/')[-1].split('.')[0]
        savesubdir = os.path.join(self.save_dir, self.file_name)
        if not os.path.exists(savesubdir):
            os.makedirs(savesubdir, exist_ok=False)
        for i, (x, y) in enumerate(points):
            savecroppath = os.path.join(savesubdir, f'{self.file_name}_x_{x}_y_{y}.png')
            savecroppath1 = os.path.join(savesubdir, f'{self.file_name}_x_{x}_y_{y}_vips.png')
            # row is y, col is x
            print(savecroppath)
            if y + self.tilesize < orig_h and x + self.tilesize < orig_w:
                # TODO change to vips cropping
                crop = vips_orig_img.crop(x, y, 1024, 1024)
                # extrema = crop.getextrema()

                # #Check if the image is transparent
                # if extrema[3][0] < 255:
                #     continue
                
                # crop.save(savecroppath)
                crop.write_to_file(savecroppath1)
        print(f'Saved crops to {savesubdir}')
    def getContour(self, thresh, vips_array, plot_countor=False):
        contours, hierarchy = cv2.findContours(thresh[1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key=cv2.contourArea)
        
        # TODO find good name for downscaled_h
        x,y,downscaled_w,downscaled_h = cv2.boundingRect(cnt)
        vips_array = cv2.rectangle(vips_array.copy(),(x, y),(x + downscaled_w,y + downscaled_h),(0,255,0),20)
        

        # TODO: remove the debug comments
        print(x)
        print(y)
        print(x+downscaled_w)
        print(y+downscaled_h)

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
           
    def rename_files(self):
        '''
        This function will renmae the files to x_y.jpg from 0_row_col.jpg format

        '''
        folders = glob.glob(os.path.join(self.save_dir, '*'))
        for folder in tqdm(folders):
            # TODO check if this is the right way
            tiled_images = glob.glob(os.path.join(folder, '*'))
            tiled_images = natsort.natsorted(tiled_images)

            for tile_img in tiled_images:
    
                if len(tile_img) == 0:
                    continue

                dirname, fname = os.path.split(tile_img)
                newdir_name = os.path.join(dirname, "Tiles")

                if not os.path.exists(newdir_name):
                    os.makedirs(newdir_name)

                # os.rename(tile_img, replace_string)
                destination_name = os.path.join(newdir_name, fname)
                shutil.move(tile_img, destination_name)
            # os.rmdir(dirname)

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
        
        for tile_folder in tiled_folders:
            tiled_images = glob.glob(os.path.join(tile_folder, "*.png"))
            for tile_img in tiled_images:
                print("\n tile : ", tile_img)

                
                x = io.imread(tile_img)

                dark_pixels = np.count_nonzero(x < int(intensity_threshold))

                light_pixels = np.count_nonzero(x > int(intensity_threshold))
                

                # 1024 * 1024 *3 / 96 percent white
                if light_pixels > 3019898:
                    print(light_pixels)
                    os.remove(tile_img)
                    del_img_count = del_img_count + 1
                            
                print("Total deleted img tiles : ",  del_img_count)
    
    def tile_WSI(self):

        if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

         # TODO Remove this hardcoding later
        
        imagenames = sorted(glob.glob(os.path.join(self.wsi_home_dir, '*.mrxs')))
        imagenames = ["/home/vivek/Datasets/AmyB/amyb_wsi/XE19-010_1_AmyB_1.mrxs"]
        for imagename in tqdm(imagenames):

            # Get file_name Ex:'XE19-010_1_AmyB_1'
            file_name = imagename.split('.')
            file_name = file_name[0].split("/")[-1]
            self.file_name = file_name
            file_name = file_name + "/Tiles"
            file_name = os.path.join(self.save_dir, file_name)

            # Test
            vips_img = Vips.Image.new_from_file(imagename, level=self.slide_level)
            vips_img_orig = Vips.Image.new_from_file(imagename, level=0)
            
            vips_img = vips_img.crop(0, 1520,  5578,  13176-1700)
            vinfo = self.getVipsInfo(vips_img_orig)
            print(vinfo)
            # vips_orig_array = np.ndarray(buffer=vips_img_orig.write_to_memory(), dtype=np.uint8, shape=(vips_img_orig.height, vips_img_orig.width, vips_img_orig.bands))
            # vips_orig_array = vips_orig_array[:,:,:3]


            # vips_img.crop(500, 0,  vinfo['level[4].width'],  vinfo['level[4].height'])
            vips_array = np.ndarray(buffer=vips_img.write_to_memory(), dtype=np.uint8, shape=(vips_img.height, vips_img.width, vips_img.bands))
            vips_array = vips_array[:,:,:3]

            gray = cv2.cvtColor(vips_array, cv2.COLOR_RGB2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

    
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+ cv2.THRESH_OTSU)
            # plt.imshow(thresh[1])
            # plt.show()

            
            cnt, downscaled_w, downscaled_h = self.getContour(thresh, vips_array,plot_countor=False)

            points = self.get_points_in_contour(cnt, downscaled_w, downscaled_h)
            slide = openslide.OpenSlide(imagename)
            
            orig_w, orig_h = slide.dimensions
            # print(orig_w)
            # print(orig_h)
            # print(vinfo['level[0].width'])
            # print(vinfo['level[0].height'])

            #CROP
            self.crop_slide(vips_img_orig, slide,  points, orig_w, orig_h)
            self.rename_files()
            self.stage1_filter()
           
if __name__ == '__main__':
    result = pyfiglet.figlet_format("Image Tiling", font="slant")
    print(result)
    parser = argparse.ArgumentParser(description='Image Tiling for Model Prediction')
    parser.add_argument('wsi_home_dir',
                            help='Enter the path where the Test WSI images reside')
    parser.add_argument('save_dir',
                            help='Enter the path where you want the tiled image to reside')
    args = parser.parse_args()

    
    generate_test_data = GenerateTestData(wsi_home_dir=args.wsi_home_dir, save_dir=args.save_dir, slide_level=4, 
                            downscale_factor=16, tile_size=1024)
    generate_test_data.tile_WSI()
    
    # rename_files(tiled_path=args.save_dir)
    # stage1_filter(tiled_path=args.save_dir)
    


    
    
