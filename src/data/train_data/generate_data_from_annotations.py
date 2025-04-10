"""
This code takes annotations from Qupath generated json files and create crops and annotation masks for each slide
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
from concurrent.futures import ProcessPoolExecutor, as_completed

# Mask size should be same as image size
ID_MASK_SHAPE = (1024, 1024)

# Color Coding
lablel2id = {'Cored':'50', 'Diffuse':'100',
             'Coarse-Grained':'150', 'CAA': '200', 'Unknown':'0'}

# Assign Dataset_path
DATASET_PATH = "/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/amyb_wsi_v2/interater_data/monika-rater"



def save_img(img, file_name, tileX, tileY, label="mask"):
    """save image - use filename, tileX,tileY in the saved filename

    Args:
        img (np.array): numpy image to save
        file_name (string): filename (slide name)
        tileX (int): tile X coord
        tileY (int): tile Y coord
        label (str, optional): _description_. Defaults to "mask".
    """
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
    """Draw polygon with x and y coords on mask with ids provided

    Args:
        image_shape (np.array): _description_
        mask (numpy array): _description_
        ids (string): _description_
        coords_x (np.array): X coordinates
        coords_y (np.array): Y coordinates

    Returns:
        np.array: mask 
    """
    vertex_row_coords, vertex_col_coords = coords_y, coords_x
    fill_row_coords, fill_col_coords = draw.polygon(
        vertex_row_coords, vertex_col_coords, image_shape)

    # Row and col are flipped
    mask[fill_col_coords, fill_row_coords] = ids
    return masks

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
    return mask



def get_vips_info(vips_img):
    # # Get bounds-x and bounds-y offset
    vfields = [f.split('.') for f in vips_img.get_fields()]
    vfields = [f for f in vfields if f[0] == 'openslide']
    vfields = dict([('.'.join(k[1:]), vips_img.get('.'.join(k))) for k in vfields])Ted@
    return vfields


def process_single_image(img, json_path, visualize=False):
    """
    Processes a single image by extracting tiles and creating masks based on annotations in the corresponding JSON file.

    This function loads an image, extracts tiles based on the tile coordinates provided in a JSON file,
    generates masks for each tile based on region annotations, and saves the resulting images and masks.

    Args:
        img (str): The file path to the image that needs to be processed.
        json_path (str): The directory path where the corresponding JSON file is stored.
        visualize (bool): If set to True, displays the generated mask using matplotlib.

    Returns:
        None: This function does not return any value but saves processed images and masks to disk.
    """
    # Load the image using VIPS with level 0 to minimize memory usage
    vips_img = Vips.Image.new_from_file(img, level=0)
    
    # Extract metadata about the image
    vinfo = get_vips_info(vips_img)

    # Construct the corresponding JSON file path based on the image filename
    json_file_name = os.path.basename(img).split(".mrxs")[0] + ".json"
    json_file_name = os.path.join(json_path, json_file_name)

    # If the JSON file does not exist, exit the function
    if not exists(json_file_name):
        return

    print("Processing JSON file: ", json_file_name)

    # Open and load the JSON data
    with open(json_file_name) as f:
        data = json.load(f)

    # Iterate over each tile in the JSON data
    for tileId, ele in data.items():
        # Clean up tileId by removing square brackets
        tileId = tileId.replace("[", "").replace("]", "")
        
        # Parse the tile coordinates (tileX, tileY)
        tileX = int(tileId.split(",")[0])
        tileY = int(tileId.split(",")[1])
        
        # Define the size of each tile (1024x1024)
        tileWidth = 1024
        tileHeight = 1024

        # Adjust the tile coordinates based on the image's bounding box information
        tileX = (tileX * tileWidth) + int(vinfo['bounds-x'])
        tileY = (tileY * tileHeight) + int(vinfo['bounds-y'])

        # Crop the image using the calculated coordinates and tile size
        vips_img_crop = vips_img.crop(tileX, tileY, tileWidth, tileHeight)

        # Convert the cropped image to a numpy array
        vips_img_crop = np.ndarray(
            buffer=vips_img_crop.write_to_memory(), dtype=np.uint8,
            shape=(vips_img_crop.height, vips_img_crop.width, vips_img_crop.bands)
        )[..., :3]  # Keep only the RGB channels

        # Define the coordinates for the tile's bounding box
        x1 = tileX - int(vinfo['bounds-x'])
        x2 = tileX + tileWidth - int(vinfo['bounds-x'])
        y1 = tileY - int(vinfo['bounds-y'])
        y2 = tileY + tileHeight - int(vinfo['bounds-y'])

        # Initialize the mask (empty initially)
        id_mask = np.zeros(ID_MASK_SHAPE, dtype=np.uint8)

        # Iterate over the regions in the current tile's annotation data
        for region in ele:
            # Skip regions that don't contain a 'label' key
            if 'label' not in region.keys():
                continue

            # Ensure the region has a valid 'name' in its 'label'
            if 'name' not in region['label'].keys():
                print("Region skipped due to missing 'name' in label.")
                continue

            # Extract coordinates of the polygon defining the region
            coords_x, coords_y = zip(*region["region_attributes"][0]['points'])
            coords_x = np.array(coords_x)
            coords_y = np.array(coords_y)

            # Skip regions that overlap with the tile boundaries
            if len(coords_x[coords_x > x2]) > 0 or len(coords_y[coords_y > y2]) > 0:
                print('Region overlaps with tile boundaries. Skipping region.')
                continue

            # Adjust the coordinates to fit within the tile's boundary
            coords_x = np.mod(coords_x, tileWidth)
            coords_y = np.mod(coords_y, tileHeight)

            # Get the label ID for the region
            label = region['label']["name"]
            ids = int(lablel2id[label])

            # Use the polygon2id function to create a mask for the region
            id_mask = polygon2id(ID_MASK_SHAPE, id_mask, ids, coords_y, coords_x)

        # Generate the filename for saving the image and mask
        base_name = os.path.basename(img)
        file_name = os.path.splitext(base_name)[0]
        
        # Save the cropped image and the corresponding mask
        save_img(vips_img_crop, file_name, tileX, tileY, "image")
        save_img(id_mask, file_name, tileX, tileY, "mask")

        # Reset the mask for the next tile
        id_mask = np.zeros(ID_MASK_SHAPE, dtype=np.uint8)

    # If visualize flag is set to True, show the last generated mask using matplotlib
    if visualize:
        plt.imshow(id_mask)
        plt.show()



def process_json(WSI_path, json_path, visualize=False, max_workers=4):
    """This code does multiprocessing by calling process_single_image function on all slides

    Args:
        WSI_path (_type_): WSI file path
        json_path (_type_): Json file path
        visualize (bool, optional): true if need to visualize image. Defaults to False.
        max_workers (int, optional): max no of workers. Defaults to 4.
    """
    imagenames = glob.glob(os.path.join(WSI_path, "*.mrxs"))
    imagenames = sorted(imagenames)
    print(imagenames)
    plaque_dict = {'Cored': 0, 'Coarse-Grained': 0, 'Diffuse': 0, 'CAA': 0, 'Unknown': 0}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_image, img, json_path, visualize) for img in imagenames]
        for future in as_completed(futures):
            future.result()


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

    process_json(args.WSI_path, args.json_path, visualize=False, max_workers=12)