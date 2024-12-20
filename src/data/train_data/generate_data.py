"""Generate Mask Script

Step2: Annotations(polyong cooridnates) --> Image Mask
Mask Shape is 224 * 224

This script allows the user to generate mask image from
the json annotation file for segmentation

"""
# import sys
# sys.path.append("/opt/anaconda3/envs/gpu/lib/python3.12/site-packages")

import os
from os.path import exists
import glob
import json
import pdb
import numpy as np
from skimage import draw
import matplotlib.pyplot as plt
import argparse
import pyfiglet
from skimage import measure
from tqdm import tqdm
from PIL import Image
import pyvips as Vips

# import openslide

# Mask size should be same as image size
# TODO Remove hardcoding
ID_MASK_SHAPE = (1024, 1024)

# Color Coding
lablel2id = {'Cored':'50', 'Diffuse':'100',
             'Coarse-Grained':'150', 'CAA': '200', 'Unknown':'0'}

DATASET_PATH = "/Volumes/Finkbeiner-Steve/work/data/npsad_data/vivek/Datasets/amyb_wsi"

def save_img(img, file_name, tileX, tileY, label="mask"):
    im = Image.fromarray(img)

    base_name_with_ext = os.path.basename(file_name)

    # Remove the extension to get the folder name
    folder_name = os.path.splitext(base_name_with_ext)[0]

    folder_name = os.path.join(DATASET_PATH, folder_name)

    # Create the new folder only if it does not exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    
     # Mask Folder
    save_dir = os.path.join(folder_name, label)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_name = file_name + "_" + str(tileX)+"x" + "_" + str(tileY) + "y" + "_" + label + ".png"

    save_name = os.path.join(save_dir, file_name)
    im.save(save_name)


def polygon2id(image_shape, mask, ids, coords_x, coords_y):
    vertex_row_coords, vertex_col_coords = coords_y, coords_x
    fill_row_coords, fill_col_coords = draw.polygon(
        vertex_row_coords, vertex_col_coords, image_shape)

    # Row and col are flipped
    mask[fill_col_coords, fill_row_coords] = ids
    return mask

def polygon2mask1(image_shape, mask, color, coords_x, coords_y):
    """Compute a mask with labels having different colors
    from polygon.
    Parameters
    ----------
    image_shape : tuple of size 2.
        The shape of the mask.
    coords_x: X coordinates
    coords_y: Y coordinates
    mask : Mask with same size of the image (initially empty
    mask is given as input)
    Returns
    -------
    mask : 2-D ndarray of type 'bool'.
        The mask that corresponds to the input polygon.
    """

    vertex_row_coords, vertex_col_coords = coords_x, coords_y
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, image_shape)

    # Row and col are flipped
    mask[fill_col_coords, fill_row_coords] = color

    # mask[fill_row_coords, fill_col_coords] = color
    return mask



def get_vips_info(vips_img):
    # # Get bounds-x and bounds-y offeset
    vfields = [f.split('.') for f in vips_img.get_fields()]
    vfields = [f for f in vfields if f[0] == 'openslide']
    vfields = dict([('.'.join(k[1:]), vips_img.get('.'.join(k))) for k in vfields])

    return vfields


def process_json(WSI_path, json_path,  visualize=False):
    """This function is used to read and process the json files
    and generate save generated masks

    Parameters
    -----------
    json_path : path to json file
    save_dir : dir where the generated masks will be saved
    visualize : True , if you want to see the mask generated
    """
    

    imagenames = glob.glob(os.path.join(WSI_path, "*.mrxs"))
    imagenames = sorted(imagenames)
    plaque_dict = {'Cored': 0, 'Coarse-Grained': 0, 'Diffuse': 0, 'CAA': 0, 'Unknown': 0}


    for img in imagenames:

        
        # Read the WSI image
        vips_img = Vips.Image.new_from_file(img, level=0)
        vinfo = get_vips_info(vips_img)

        # Get the corresponding json file
        json_file_name = os.path.basename(img).split(".mrxs")[0] + ".json"
        json_file_name = os.path.join(json_path, json_file_name)
        # json_file_list = [json_file_name, "/home/vivek/Datasets/AmyB/amyb_wsi/XE19-010_1_AmyB_1_1.json"]
        # merge_json(json_file_list, "/home/vivek/Datasets/AmyB/amyb_wsi/test.json")
        # json_file_name = os.path.join(os.path.dirname(img), "XE19-010_1_AmyB_1_37894x_177901y_image.png[--series, 0].json")

        # json_file_name = "/home/vivek/Datasets/AmyB/amyb_wsi/test.json"
        
        
        if not exists(json_file_name):
            continue
        
        print("file name : ", json_file_name)

        with open(json_file_name) as f:
            data = json.load(f)
        
        # tile Ids
        for tileId, ele in data.items():
            print("***********************")
            print("tileId", tileId)
            print("Leng", len(ele))
            #print(tileId)
            tileId = tileId.replace("[","")
            tileId= tileId.replace("]","")
            tileX = int(tileId.split(",")[0])
            tileY = int(tileId.split(",")[1])
            tileWidth = 1024
            tileHeight=1024

            tileX = (tileX * tileWidth) + int(vinfo['bounds-x'])
            tileY = (tileY * tileHeight) + int(vinfo['bounds-y'])

            vips_img_crop = vips_img.crop(tileX, tileY,tileWidth, tileHeight)


            vips_img_crop = np.ndarray(buffer=vips_img_crop.write_to_memory(), dtype=np.uint8,
                                shape=(vips_img_crop.height, vips_img_crop.width, vips_img_crop.bands))[..., :3]
            
            x1 = tileX - int(vinfo['bounds-x'])
            x2 = tileX + tileWidth - int(vinfo['bounds-x'])
            y1 = tileY - int(vinfo['bounds-y'])
            y2 = tileY + tileHeight - int(vinfo['bounds-y'])

            # Reset ids for each annotation
            ids = 1

            # Create an Empty mask of size similar to image
            id_mask = np.zeros(ID_MASK_SHAPE, dtype=np.uint8)

            # region_id = 0
            # prev_label = ""
            # i = 0
            # plaque_dict[ele['label']] = plaque_dict[ele['label']] + len(ele['region_attributes'])

            # Different Objects in the same tile
            for region in ele:
                print(img)

                # print(region['label'].keys())
                # # Get tileX and tileY
                # tileX = region['tiles'][0]['tileId'][0]
                # tileY = region['tiles'][0]['tileId'][1]
                # tileWidth = region['tiles'][0]['tileBounds']["WH"][0]
                # tileHeight = region['tiles'][0]['tileBounds']["WH"][1]

                # crop the image
                # get the bound-x and bounds-y, offset as Vips crops the empty spaces. Qupath does not
            
                if 'label' in region:
                    # Check if 'name' key exists in the dictionary under 'label'
                    
                    if 'name' not in region['label'].keys():
                        print("Continue")
                        continue
            
                coords_x, coords_y = zip(*region["region_attributes"][0]['points'])
                coords_x = np.array(coords_x)
                coords_y = np.array(coords_y)

                # Remove overlap annotations
                if len(coords_x[coords_x > x2]) > 0 or len(coords_y[coords_y > y2]) > 0:
                    print('Overlap')
                    continue


                # Translate the coordinates to fit within the image crop
                coords_x = np.mod(coords_x, tileWidth)
                coords_y = np.mod(coords_y, tileHeight)

                # label
                label = region['label']["name"]
                print("label", label)
                ids = int(lablel2id[label])

                print("label", label)
                print("IDs", ids)
            
                # Use polygon2id function to create a mask
                id_mask = polygon2id(ID_MASK_SHAPE, id_mask, ids, coords_y, coords_x)

        

            # Extract FIlename
            # Extract the base name (file name with extension)
            base_name = os.path.basename(img)

            # Remove the extension to get the file name only
            file_name = os.path.splitext(base_name)[0]

            save_img(vips_img_crop, file_name, tileX, tileY, "image")
            save_img(id_mask, file_name, tileX, tileY,"mask")

            if visualize:
                plt.imshow(id_mask)
                plt.show()
            
            id_mask = np.zeros(ID_MASK_SHAPE, dtype=np.uint8)



def merge_json(json_files, json_output_file=None):
    """
    merge_json: a method to return the combined json of a list of json files

    """
    result = list()
    for f1 in json_files:
        with open(f1, 'r') as infile:
            result.extend(json.load(infile))

    with open(json_output_file, 'w') as output_file:
        json.dump(result, output_file)



if __name__ == '__main__':
    result = pyfiglet.figlet_format("Generate Mask", font="slant")
    print(result)

    parser = argparse.ArgumentParser(description='Generate Mask')
    
    parser.add_argument('WSI_path',
                        help='Enter the path where WSI resides')
    parser.add_argument('json_path',
                        help='Enter the path where json annotation resides')
    
    args = parser.parse_args()

    process_json(args.WSI_path, args.json_path)
