import os
from os.path import exists
os.environ['OMP_NUM_THREADS'] = '1'
import glob
import sys
sys.path.append(os.path.join(os.getcwd(), *tuple(['..'])))
import argparse
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
from torchvision import transforms
import torchvision
import cv2

from models_pytorch_lightning.model_mrcnn_config import _default_mrcnn_config, build_default
from features import build_features
from models_pytorch_lightning.generalized_mask_rcnn_pl import LitMaskRCNN
from features import transforms as T
import torch


class_names = ["Cored","Diffuse","Coarse-Grained","CAA"]
test_config = dict(batch_size = 2,num_classes=4, device_id =0)
model_name= "/home/mahirwar/Desktop/Monika/npsad_data/vivek/runpod_mrcnn_models/yp2mf3i8_epoch=108-step=872.ckpt"
model_config = _default_mrcnn_config(num_classes=1 + test_config['num_classes']).config
model = LitMaskRCNN.load_from_checkpoint(model_name)


device = torch.device('cuda', test_config['device_id'])
model = model.to(device)
model.eval()


ID_MASK_SHAPE = (1024, 1024)

# Color Coding
lablel2id = {'Cored':'50', 'Diffuse':'100',
             'Coarse-Grained':'150', 'CAA': '200', 'Unknown':'0'}

DATASET_PATH = "/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/amyb_wsi"

def get_vips_info(vips_img):
    # # Get bounds-x and bounds-y offset
    vfields = [f.split('.') for f in vips_img.get_fields()]
    vfields = [f for f in vfields if f[0] == 'openslide']
    vfields = dict([('.'.join(k[1:]), vips_img.get('.'.join(k))) for k in vfields])

    return vfields

def prepare_input(image):
    image_float_np = np.float32(image) / 255
    # define the torchvision image transforms
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    input_tensor = transform(image)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_tensor = input_tensor.to(device)
    # Add a batch dimension
    input_tensor = input_tensor.unsqueeze(0)
    return input_tensor, image_float_np


def get_output(outputs, threshold):
    i =0 
    pred_boxes =outputs[i]["boxes"].cpu().detach().numpy()
    pred_class_ids = outputs[i]["labels"].cpu().detach().numpy()
    pred_scores = outputs[i]["scores"].cpu().detach().numpy()
    pred_masks = outputs[i]["masks"][pred_scores>=threshold]
    pred_masks=pred_masks.squeeze(1).permute(2, 1, 0)
    pred_masks = pred_masks.cpu().detach().numpy()
    pred_boxes =pred_boxes[pred_scores>=threshold]
    pred_class_ids = pred_class_ids[pred_scores>=threshold]
    pred_scores = pred_scores[pred_scores>=threshold]
    return pred_scores,pred_class_ids,pred_masks,pred_boxes


def find_intersection(pred_boxes, region_points):
   # Define the points of the first polygon
    polygon1 = np.array(pred_boxes, dtype=np.float32)

    # Define the points of the second polygon
    polygon2 = np.array(region_points, dtype=np.float32)

    # Ensure the polygons are in the correct format
    polygon1 = polygon1.reshape((-1, 1, 2))
    polygon2 = polygon2.reshape((-1, 1, 2))

    # Use intersectConvexConvex to check for intersection
    intersect_area, intersection = cv2.intersectConvexConvex(polygon1, polygon2)

    if intersect_area > 0:
        return 1
    else:
        return 0


json_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/interrater-test-jsons"
img ="/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/amy-def-mfg-images/XE07-049_1_AmyB_1.mrxs"
json_file_name = "XE07-049_1_AmyB_1"
threshold = 0.5



vips_img = Vips.Image.new_from_file(img, level=0)
vinfo = get_vips_info(vips_img)

# Get the corresponding json file
json_file_name = os.path.basename(img).split(".mrxs")[0] + ".json"
json_file_name = os.path.join(json_path, json_file_name)

#if not exists(json_file_name):
#    return

with open(json_file_name) as f:
    data = json.load(f)
    
counter = 2

for tileId, ele in data.items():
    # print("****************", tileId)
    tileId = tileId.replace("[", "")
    tileId = tileId.replace("]", "")
    tileX = int(tileId.split(",")[0])
    tileY = int(tileId.split(",")[1])
    tileWidth = 1024
    tileHeight = 1024

    tileX = (tileX * tileWidth) + int(vinfo['bounds-x'])
    tileY = (tileY * tileHeight) + int(vinfo['bounds-y'])
        
    vips_img_crop = vips_img.crop(tileX, tileY, tileWidth, tileHeight)

    vips_img_crop = np.ndarray(buffer=vips_img_crop.write_to_memory(), dtype=np.uint8,
                                shape=(vips_img_crop.height, vips_img_crop.width, vips_img_crop.bands))[..., :3]

    x1 = tileX - int(vinfo['bounds-x'])
    x2 = tileX + tileWidth - int(vinfo['bounds-x'])
    y1 = tileY - int(vinfo['bounds-y'])
    y2 = tileY + tileHeight - int(vinfo['bounds-y'])
    
    input_tensor, image_float_np = prepare_input(vips_img_crop)
    outputs = model.forward(input_tensor)
    pred_scores,pred_class_ids,pred_masks,pred_boxes=get_output(outputs,threshold)

    pred_boxes_cnt =  [np.int32(np.array([[box[0],box[1]],[box[2],box[3]]])) for box in pred_boxes]
    matches = 0
    #print("pred_boxes", pred_boxes_cnt)
    for region in ele:
        if 'label' in region.keys():
            #print(region['label']['name'])
            #print(region["raterName"])
            #print(region["region_attributes"][0]["points"])
            #print(region["region_attributes"][0]["roiBounds"])
            
            x_bound = region["region_attributes"][0]['tiles'][0]["tileBounds"]['XY'][0]
            y_bound = region["region_attributes"][0]['tiles'][0]["tileBounds"]['XY'][1]
            #x_bound = region["region_attributes"][0]["roiBounds"]['XY'][0]
            #y_bound = region["region_attributes"][0]["roiBounds"]['XY'][1]
            region_points = region["region_attributes"][0]["points"]
            
            region_points = [ [p[0]-x_bound, p[1]-y_bound] for p in region_points]
            #print("region_points",region_points)
            #mean_reg = np.mean(region_points, axis=0)
            #print("mean point", mean_reg)
            for i, box in enumerate(pred_boxes_cnt):
                intresection = find_intersection(pred_boxes, region_points)
                if intresection==1:
                    if class_names[pred_class_ids[i]-1]==region['label']['name']:
                        matches=matches+1
    print(matches)
                        
                #print("pred_box",box)
                #print("Mean point", mean_reg)
                #dist = cv2.pointPolygonTest(box,(int(mean_reg[0]),int(mean_reg[1])),True)
                #if dist>=1:
                #    print("match",region['label']['name'],region["raterName"] )
            #counter=counter-1
            #if counter==1:
            #    break
    #if counter==1:
    #    break