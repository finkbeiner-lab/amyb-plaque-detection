"""Generate Mask Script

Annotations(polyong cooridnates) --> Image Mask
Mask Shape is 224 * 224

This script allows the user to generate mask image from
the json annotation file for segmentation

"""
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
lablel2id = {'Core':'50', 'Diffuse':'100',
             'Neuritic':'150', 'CAA': '200', 'Unknown':'0'}

DATASET_PATH = "/mnt/new-nas/work/data/npsad_data/vivek/Datasets/test_amyb_wsi"

def save_img(img, file_name, tileX, tileY, save_dir, label="mask"):
    im = Image.fromarray(img)

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


    # Mask Folder
    mask_save_dir = os.path.join(DATASET_PATH, "labels")
    if not os.path.exists(mask_save_dir):
        os.makedirs(mask_save_dir)

    # Image Folder
    image_save_dir = os.path.join(DATASET_PATH, "images")
    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)


    imagenames = glob.glob(os.path.join(WSI_path, "*.mrxs"))
    imagenames = sorted(imagenames)
    plaque_dict = {'Core': 0, 'Neuritic': 0, 'Diffuse': 0, 'CAA': 0, 'Unknown': 0}

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
        
        for ele in tqdm(data):

            # Reset ids for each annotation
            ids = 1

            # Create an Empty mask of size similar to image
            id_mask = np.zeros(ID_MASK_SHAPE, dtype=np.uint8)

            region_id = 0
            prev_label = ""
            i = 0
            plaque_dict[ele['label']] = plaque_dict[ele['label']] + len(ele['region_attributes'])

            for region in ele['region_attributes']:

                # Get tileX and tileY
                tileX = region['tiles'][0]['tileId'][0]
                tileY = region['tiles'][0]['tileId'][1]
                tileWidth = region['tiles'][0]['tileBounds']["WH"][0]
                tileHeight = region['tiles'][0]['tileBounds']["WH"][1]

                # crop the image
                # get the bound-x and bounds-y, offset as Vips crops the empty spaces. Qupath does not
                tileX = (tileX * tileWidth) + int(vinfo['bounds-x'])
                tileY = (tileY * tileHeight) + int(vinfo['bounds-y'])

                vips_img_crop = vips_img.crop(tileX, tileY,tileWidth, tileHeight)

                # Region Bounds
                regX = region["roiBounds"]["XY"][0]
                regY = region["roiBounds"]["XY"][1]
                regWidth = region["roiBounds"]["WH"][0]
                regHeight = region["roiBounds"]["WH"][1]

                # region_crop = vips_img.crop(regX, regY, tileWidth, tileHeight)
                vips_img_crop = np.ndarray(buffer=vips_img_crop.write_to_memory(), dtype=np.uint8,
                                    shape=(vips_img_crop.height, vips_img_crop.width, vips_img_crop.bands))[..., :3]
                # region_img_crop = np.ndarray(buffer=vips_img_crop.write_to_memory(), dtype=np.uint8,
                #                 shape=(vips_img_crop.height, vips_img_crop.width, vips_img_crop.bands))[..., :3]

                # unpack from [x,y] to [x], [y]
                coords_x, coords_y = zip(*region['points'])

                coords_x = np.array(coords_x)
                coords_y = np.array(coords_y)

                x1 = tileX - int(vinfo['bounds-x'])
                x2 = tileX + tileWidth - int(vinfo['bounds-x'])
                y1 = tileY - int(vinfo['bounds-y'])
                y2 = tileY + tileHeight - int(vinfo['bounds-y'])


                # Remove overlap annotations
                if len(coords_x[coords_x > x2]) > 0 or len(coords_y[coords_y > y2]) > 0:
                    print('Overlap')
                    continue


                # Translate the coordinates to fit within the image crop
                coords_x = np.mod(coords_x, tileWidth)
                coords_y = np.mod(coords_y, tileHeight)



                # label
                label = ele['label']

                if i == 0:
                    ids = int(lablel2id[label])
                elif label == prev_label:
                    ids = int(lablel2id[label])

                # Use polygon2id function to create a mask
                id_mask = polygon2id(ID_MASK_SHAPE, id_mask, ids, coords_y, coords_x)

                # ids = ids + 5

                prev_label = label

                i+=1

                save_img(vips_img_crop, ele['filename'], tileX, tileY, image_save_dir, "image")
                save_img(id_mask, ele['filename'], tileX, tileY, mask_save_dir,"mask")
                id_mask = np.zeros(ID_MASK_SHAPE, dtype=np.uint8)

            if visualize:
                plt.imshow(id_mask)
                plt.show()

        print(plaque_dict)


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
