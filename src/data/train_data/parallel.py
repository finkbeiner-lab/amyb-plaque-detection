import os
from os.path import exists
import glob
import json
import numpy as np
from skimage import draw
import matplotlib.pyplot as plt
import argparse
import pyfiglet
from tqdm import tqdm
from PIL import Image
import pyvips as Vips
from concurrent.futures import ProcessPoolExecutor

# Mask size should be same as image size
ID_MASK_SHAPE = (1024, 1024)

# Color Coding
label2id = {'Cored': '50', 'Diffuse': '100', 'Coarse-Grained': '150', 'CAA': '200', 'Unknown': '0'}

DATASET_PATH = "/Volumes/Finkbeiner-Steve/work/data/npsad_data/vivek/Datasets/amyb_wsi"

def save_img(img, file_name, tileX, tileY, label="mask"):
    im = Image.fromarray(img)
    base_name_with_ext = os.path.basename(file_name)
    folder_name = os.path.splitext(base_name_with_ext)[0]
    folder_name = os.path.join(DATASET_PATH, folder_name)

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    save_dir = os.path.join(folder_name, label)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_name = file_name + "_" + str(tileX) + "x" + "_" + str(tileY) + "y" + "_" + label + ".png"
    save_name = os.path.join(save_dir, file_name)
    im.save(save_name)

def polygon2id(image_shape, mask, ids, coords_x, coords_y):
    vertex_row_coords, vertex_col_coords = coords_y, coords_x
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, image_shape)
    mask[fill_col_coords, fill_row_coords] = ids
    return mask

def get_vips_info(vips_img):
    vfields = [f.split('.') for f in vips_img.get_fields()]
    vfields = [f for f in vfields if f[0] == 'openslide']
    vfields = dict([('.'.join(k[1:]), vips_img.get('.'.join(k))) for k in vfields])
    return vfields

def process_single_image(img, json_path, visualize=False):
    vips_img = Vips.Image.new_from_file(img, level=0)
    vinfo = get_vips_info(vips_img)
    json_file_name = os.path.basename(img).split(".mrxs")[0] + ".json"
    json_file_name = os.path.join(json_path, json_file_name)

    if not exists(json_file_name):
        return

    print("Processing file: ", json_file_name)
    with open(json_file_name) as f:
        data = json.load(f)

    plaque_dict = {'Cored': 0, 'Coarse-Grained': 0, 'Diffuse': 0, 'CAA': 0, 'Unknown': 0}
    for ele in tqdm(data):
        ids = 1
        id_mask = np.zeros(ID_MASK_SHAPE, dtype=np.uint8)
        prev_label = ""
        i = 0
        plaque_dict[ele['label']] = plaque_dict[ele['label']] + len(ele['region_attributes'])

        for region in ele['region_attributes']:
            tileX = region['tiles'][0]['tileId'][0]
            tileY = region['tiles'][0]['tileId'][1]
            tileWidth = region['tiles'][0]['tileBounds']["WH"][0]
            tileHeight = region['tiles'][0]['tileBounds']["WH"][1]

            tileX = (tileX * tileWidth) + int(vinfo['bounds-x'])
            tileY = (tileY * tileHeight) + int(vinfo['bounds-y'])

            vips_img_crop = vips_img.crop(tileX, tileY, tileWidth, tileHeight)
            coords_x, coords_y = zip(*region['points'])
            coords_x = np.array(coords_x)
            coords_y = np.array(coords_y)

            x1 = tileX - int(vinfo['bounds-x'])
            x2 = tileX + tileWidth - int(vinfo['bounds-x'])
            y1 = tileY - int(vinfo['bounds-y'])
            y2 = tileY + tileHeight - int(vinfo['bounds-y'])

            if len(coords_x[coords_x > x2]) > 0 or len(coords_y[coords_y > y2]) > 0:
                print('Overlap')
                continue

            coords_x = np.mod(coords_x, tileWidth)
            coords_y = np.mod(coords_y, tileHeight)
            label = ele['label']

            if i == 0:
                ids = int(label2id[label])
            elif label == prev_label:
                ids = int(label2id[label])

            id_mask = polygon2id(ID_MASK_SHAPE, id_mask, ids, coords_y, coords_x)
            prev_label = label
            i += 1

            vips_img_crop = np.ndarray(buffer=vips_img_crop.write_to_memory(), dtype=np.uint8,
                                       shape=(vips_img_crop.height, vips_img_crop.width, vips_img_crop.bands))[..., :3]

            save_img(vips_img_crop, ele['filename'], tileX, tileY, "image")
            save_img(id_mask, ele['filename'], tileX, tileY, "mask")
            id_mask = np.zeros(ID_MASK_SHAPE, dtype=np.uint8)

        if visualize:
            plt.imshow(id_mask)
            plt.show()

    print(plaque_dict)

def process_json(WSI_path, json_path, visualize=False):
    imagenames = glob.glob(os.path.join(WSI_path, "*.mrxs"))
    imagenames = sorted(imagenames)

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_single_image, img, json_path, visualize) for img in imagenames]
        for future in tqdm(futures):
            future.result()

if __name__ == '__main__':
    result = pyfiglet.figlet_format("Generate Mask", font="slant")
    print(result)

    parser = argparse.ArgumentParser(description='Generate Mask')
    parser.add_argument('WSI_path', help='Enter the path where WSI resides')
    parser.add_argument('json_path', help='Enter the path where json annotation resides')
    args = parser.parse_args()

    process_json(args.WSI_path, args.json_path)
