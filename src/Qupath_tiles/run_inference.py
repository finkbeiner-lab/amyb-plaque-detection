import os
import sys
sys.path.insert(0, '../')
#from models.model_mrcnn import _default_mrcnn_config, build_default
import torchvision
import torch
from models_pytorch_lightning.generalized_mask_rcnn_pl import LitMaskRCNN
from timeit import default_timer as timer 
#import concurrent.futures
#from concurrent.futures import ProcessPoolExecutor, as_completed
#import psutil
#import multiprocessing
import numpy as np
import cv2
import glob
import pandas as pd
from skimage.measure import regionprops
from skimage.measure import label, regionprops, regionprops_table
from skimage import data, filters, measure, morphology
from skimage.color import rgb2hed, hed2rgb
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import argparse



class NumpyArrayDataset(Dataset):
    def __init__(self, array_dict):
        """
        Args:
            arrays (list of np.ndarray): List of NumPy arrays.
            keys (list): List of keys corresponding to each array.
        """
        self.arrays = array_dict

    def __len__(self):
        return len(self.arrays)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the item to retrieve.
        
        Returns:
            dict: Contains the 'array' and 'key' for the item at index idx.
        """
        array = self.arrays[idx][1]
        key = self.arrays[idx][0]
        # Define the torchvision image transforms
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        input_tensor = transform(array)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_tensor = input_tensor.to(device)
        image_float_np = array.astype(np.float32) / 255.0
        return {'tensor': input_tensor, 'array':image_float_np, 'key': key}




class ExplainPredictions():
    # TODO fix the visualization flags
    def __init__(self, model, x, y, Image_buffer, detection_threshold):
        self.model = model
        self.x = x
        self.y = y
        self.Image_buffer = Image_buffer
        self.detection_threshold = detection_threshold
        self.class_names = ['Cored', 'Diffuse', 'Coarse-Grained', 'CAA']
        self.class_to_colors = {'Cored': (255, 0, 0), 'Diffuse' : (0, 0, 255), 'Coarse-Grained': (0,255,0), 'CAA':(225, 255, 0)}
        #self.result_save_dir= os.path.join( "/home/mahirwar/Desktop/Monika/npsad_data/vivek/reports/New-Minerva-Data-output", self.model_input_path.split("/")[-1])
        self.colors = np.random.uniform(0, 255, size=(len(self.class_names), 3))
        self.column_names = ["image_name", "region", "region_mask", "label", 
                            "confidence", "brown_pixels", "centroid", 
                            "eccentricity", "area", "equivalent_diameter","mask_present"]

    
    def prepare_input(self, image):
    
        image_float_np = np.float32(image) / 255
        # define the torchvision image transforms
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        input_tensor = transform(image)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_tensor = input_tensor.to(device)
        # Add a batch dimension:
        input_tensor = input_tensor.unsqueeze(0)

        return input_tensor, image_float_np

    def draw_segmentation_map(self, image, masks, boxes, labels):
        alpha = 1
        beta = 0.6
        gamma = 0
        result_masks = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        masks = [mask.squeeze() for mask in masks]

        for i, mask in enumerate(masks):
            color = self.class_to_colors[labels[i]]
            red_map, green_map, blue_map = [np.zeros_like(mask, dtype=np.uint8) for _ in range(3)]        
            red_map[mask == 1], green_map[mask == 1], blue_map[mask == 1] = color
            result_masks[mask == 1] = 255
        return result_masks

    def get_outputs_nms(self, input_tensor,image,img_name,  score_threshold = 0.5, iou_threshold = 0.5):
        #start=timer()
        with torch.no_grad():
            # forward pass of the image through the model
            outputs = self.model(input_tensor)
        #print(timer()-start)
        r= []
        for j in range(len(outputs)):
            boxes = outputs[j]['boxes']
            labels = outputs[j]['labels']
            scores = outputs[j]['scores']
            masks = outputs[j]['masks']
            # Apply score threshold
            keep = scores > score_threshold
            boxes, labels, scores, masks = boxes[keep], labels[keep], scores[keep], masks[keep]
            keep = torchvision.ops.nms(boxes, scores, iou_threshold)
            boxes, labels, scores, masks = boxes[keep], labels[keep], scores[keep], masks[keep]
            scores = list(scores.detach().cpu().numpy())
            masks = list((masks>0.5).detach().cpu().numpy())
            boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]  for i in boxes.detach().cpu()]
            labels = list(labels.detach().cpu().numpy())
            labels = [self.class_names[i-1] for i in labels]
            
            result_masks = self.draw_segmentation_map(image[j], masks, boxes, labels)
            total_brown_pixels = self.get_brown_pixel_cnt(image[j], img_name[j])
            df = self.quantify_plaques(pd.DataFrame(), img_name[j], result_masks, boxes, labels, scores, total_brown_pixels)
            #print(df)
            r.append(df)
        return pd.concat(r, ignore_index=True)
    
    def quantify_plaques(self, df, img_name, result_masks, boxes, labels, scores, total_brown_pixels):
        '''This function takes masks image and generates attributes like plaque count, area, and eccentricity'''
        plaque_counts = {
            "Cored": 0,
            "Coarse-Grained": 0,
            "Diffuse": 0,
            "CAA": 0
        }
        img_x, img_y = img_name.split("_")[0], img_name.split("_")[2]
        for i, label in enumerate(labels):
            if len(boxes) == 0:
                continue
            
            x1, x2 = boxes[i][0][1], boxes[i][1][1]
            y1, y2 = boxes[i][0][0], boxes[i][1][0]
            
            cropped_img_mask = result_masks[x1:x2, y1:y2]
            
            _, bw_img = cv2.threshold(cropped_img_mask, 0, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5, 5), np.uint8)
            closing = cv2.morphologyEx(bw_img, cv2.MORPH_CLOSE, kernel)
            regions = regionprops(closing)
            mask_present = 1 if 0 in np.unique(closing) else 0
            
            #qupath_coord_x = self.x +img_x +  (y1 + y2)//2
            
            qupath_coord_x1 = self.x +img_x +  y1
            qupath_coord_x2 = self.x +img_x +  y2
            qupath_coord_y1 = self.y + img_y + x1
            qupath_coord_y2 = self.y + img_y + x2
            
            #qupath_coord_y = self.y + img_y + (x1 + x2)//2

            for props in regions:
                plaque_counts[label] += 1
                data_record = { 'image_name': img_name,  'label': label, 'confidence': scores[i], 'brown_pixels': total_brown_pixels, 'core': plaque_counts["Cored"], 
                    'coarse_grained': plaque_counts["Coarse-Grained"], 'diffuse': plaque_counts["Diffuse"], 'caa': plaque_counts["CAA"], 'centroid': props.centroid, 
                    'eccentricity': props.eccentricity,  'area': props.area,  'equivalent_diameter': props.equivalent_diameter, 'mask_present': mask_present,
                    'qupath_coord_x1':qupath_coord_x1,'qupath_coord_x2':qupath_coord_x2,  'qupath_coord_y1':qupath_coord_y1,
                    'qupath_coord_y2':qupath_coord_y2}
                df = pd.concat([df, pd.DataFrame.from_records([data_record])], ignore_index=True)

        return df
    
    
    def get_brown_pixel_cnt(self, img, img_name):
        # Separate the stains from the IHC image
        ihc_hed = rgb2hed(img)

        # Create an RGB image for the DAB stain (brown color)
        null_channel = np.zeros_like(ihc_hed[:, :, 0])
        ihc_d = hed2rgb(np.stack((null_channel, null_channel, ihc_hed[:, :, 2]), axis=-1)).astype('float32')

        # Convert to grayscale and apply Gaussian blur
        gray = cv2.cvtColor(ihc_d, cv2.COLOR_RGB2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Count brown pixels (pixels with intensity below 0.35)
        brown_pixel_count = np.sum(gray_blurred < 0.35)
        
        return brown_pixel_count

    def create_dataloader(self, arrays, batch_size=1, shuffle=False, num_workers=0):
        dataset = NumpyArrayDataset(arrays)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return dataloader

    def process_tile(self, batch):
        img_name, input_tensor, image = batch["key"],batch["tensor"],batch["array"]
        df = self.get_outputs_nms(input_tensor,image,img_name, score_threshold = 0.6, iou_threshold = 0.5)
        return df

    
    def generate_results_mpp(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval().to(device)
        #image = cv2.imread(filepath)
        #numpy_arrays = [(str(i), np.random.randn(1024, 1024,3).astype(np.uint8)) for i in range(20)]
        #image = np.random.randn(3072, 3072,3).astype(np.uint8)
        image = np.array(self.Image_buffer)
        height, width, channels = image.shape 
        assert height==3072 
        assert width==3072 
        # Tile size
        tile_size = 1024
        # Loop to create the tiles
        
        tiles = []
        tile_names = []
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                # Crop the image to create a tile
                tile = image[y:y+tile_size, x:x+tile_size]
                tiles.append(tile)
                tile_names.append(str(x)+"_x_"+str(y)+"_y")
            
        numpy_arrays = [(tile_names[i],tiles[i]) for i in range(len(tiles))]
        
        dataloader = self.create_dataloader(numpy_arrays, batch_size=9, shuffle=False, num_workers=0)
        results =[]
        for batch in dataloader:
            df = self.process_tile(batch)
            results.append(df)
        if len(results)>0:
            final_df = pd.concat(results, ignore_index=True)

        return final_df
        
        
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("x", type=int, help="An integer value for x")
    parser.add_argument("y", type=int, help="An integer value for y")
    parser.add_argument("Image_buffer" )
    #parser.add_argument()
    args = parser.parse_args()
      
    model_name = "/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/runpod_mrcnn_models/yp2mf3i8_epoch=108-step=872.ckpt"
    model = LitMaskRCNN.load_from_checkpoint(model_name)
        
    explain = ExplainPredictions(model, args.x, args.y ,args.Image_buffer,  detection_threshold=0.6)
    final_df = explain.generate_results_mpp()
    