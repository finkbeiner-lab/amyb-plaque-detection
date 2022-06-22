"""Generate Mask Script

Annotations(polyong cooridnates) --> Image Mask
Mask Shape is 224 * 224

This script allows the user to generate mask image from 
the json annotation file for segmentation

"""
import os
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
import openslide

# Mask size should be same as image size
ID_MASK_SHAPE = (224, 224)

# Color Coding
lablel2id = {'Core':'50', 'Diffused':'100',
             'Neuritic':'150', 'Unknown':'250'}

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

def polygon2mask(image_shape, mask, color, coords_x, coords_y):
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
   
    vertex_row_coords, vertex_col_coords = coords_y, coords_x
    fill_row_coords, fill_col_coords = draw.polygon(
        vertex_row_coords, vertex_col_coords, image_shape)
    
    # Row and col are flipped 
    mask[fill_col_coords, fill_row_coords] = color
    # mask[fill_row_coords, fill_col_coords] = color
    return mask

def cropTiles(tileX, tileY, width, height): {


}

def get_vips_info(vips_img):
    # # Get bounds-x and bounds-y offeset
    vfields = [f.split('.') for f in vips_img.get_fields()]
    vfields = [f for f in vfields if f[0] == 'openslide']
    vfields = dict([('.'.join(k[1:]), vips_img.get('.'.join(k))) for k in vfields])
    
    return vfields


def process_json(json_path, imagename,  save_dir, visualize=False):
    """This function is used to read and process the json files
    and generate save generated masks
    
    Parameters
    -----------
    json_path : path to json file
    save_dir : dir where the generated masks will be saved
    visualize : True , if you want to see the mask generated
    """
    

    # Mask Folder
    mask_save_dir = os.path.join(save_dir, "masks")
    if not os.path.exists(mask_save_dir):
        os.makedirs(mask_save_dir)
    
    # Image Folder
    image_save_dir = os.path.join(save_dir, "images")
    if not os.path.exists(image_save_dir):
        os.makedirs(image_save_dir)

    # Read the WSI image
    vips_img = Vips.Image.new_from_file(imagename, level=0)
    vinfo = get_vips_info(vips_img)

    

    json_files = glob.glob(os.path.join(json_path, "*.json"))
    for json_file in json_files:

        print("file name : ", json_file)
        with open(json_file) as f:
            data = json.load(f)

        for ele in tqdm(data):

            # Reset ids for each annotation
            ids = 1

            # Create an Empty mask of size similar to image
            id_mask = np.zeros(ID_MASK_SHAPE, dtype=np.uint8)

            region_id = 0
            prev_label = ""
            i = 0
           
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

                vips_img = vips_img.crop(tileX, tileY,tileWidth, tileHeight)
                vips_img = np.ndarray(buffer=vips_img.write_to_memory(), dtype=np.uint8, 
                                      shape=(vips_img.height, vips_img.width, vips_img.bands))[..., :3]


                # unpack from [x,y] to [x], [y]
                coords_x, coords_y = zip(*region['points'])
                coords_x = np.array(coords_x)
                coords_y = np.array(coords_y)

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

        
            save_img(vips_img, ele['filename'], tileX, tileY, image_save_dir, "image")
            save_img(id_mask, ele['filename'], tileX, tileY, mask_save_dir,"mask")

                
    
        if visualize:
            plt.imshow(id_mask)
            plt.show()
        # pdb.set_trace()


if __name__ == '__main__':
    result = pyfiglet.figlet_format("Generate Mask", font="slant")
    print(result)

    parser = argparse.ArgumentParser(description='Generate Mask')
    parser.add_argument('json_path',
                        help='Enter the path annotation json file resides')
    parser.add_argument('WSI_path',
                        help='Enter the path where WSI resides')
    parser.add_argument('save_dir',
                        help='Enter the path where you want to save image masks')

    args = parser.parse_args()

    process_json(args.json_path, args.WSI_path, args.save_dir)


