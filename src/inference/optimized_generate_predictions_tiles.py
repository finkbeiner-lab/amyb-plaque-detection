import os
import sys
sys.path.insert(0, '../')
#from models.model_mrcnn import _default_mrcnn_config, build_default
import torchvision
import torch
from models_pytorch_lightning.model_mrcnn_config import _default_mrcnn_config, build_default
from features import build_features
from data.test_data import tiling_nosaving
from models_pytorch_lightning.generalized_mask_rcnn_pl import LitMaskRCNN
#from utils.helper_functions import evaluate_metrics, get_outputs, compute_iou, evaluate_mask_rcnn
from features import build_features
from timeit import default_timer as timer 
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
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
import TileArrayDataloader 
from multiprocessing import Pool, cpu_count
#torch.multiprocessing.set_start_method('spawn', force=True)



class ExplainPredictions():
    # TODO fix the visualization flags
    def __init__(self, model, model_input_path, slide_dir, slide_list, tile_dir, detection_threshold):
        self.model = model
        self.model_input_path = model_input_path
        self.slide_dir = slide_dir
        self.slide_list = slide_list
        self.tile_dir = tile_dir
        self.detection_threshold = detection_threshold
        self.class_names = ['Cored', 'Diffuse', 'Coarse-Grained', 'CAA']
        self.class_to_colors = {'Cored': (255, 0, 0), 'Diffuse' : (0, 0, 255), 'Coarse-Grained': (0,255,0), 'CAA':(225, 255, 0)}
        self.result_save_dir= os.path.join( "/home/mahirwar/Desktop/Monika/npsad_data/vivek/reports/New-Minerva-Data-output", self.model_input_path.split("/")[-1])
        self.colors = np.random.uniform(0, 255, size=(len(self.class_names), 3))
        self.column_names = ["image_name", "region", "region_mask", "label", 
                            "confidence", "brown_pixels", "centroid", 
                            "eccentricity", "area", "equivalent_diameter","mask_present"]
        self.quantify_path = ""


    def make_result_dirs(self, folder_name):
        save_path = os.path.join(self.result_save_dir, folder_name)    
        csv_name = folder_name + "_quantify.csv"
        quantify_path = os.path.join(save_path, csv_name)
        self.quantify_path = quantify_path


    
    def prepare_input(self, image):
        # Normalize the image to [0, 1] range
        image_float_np = image.astype(np.float32) / 255.0

        # Define the torchvision image transforms
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])

        # Apply the transform to the image
        input_tensor = transform(image)

        # Move the tensor to the appropriate device (GPU if available)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_tensor = input_tensor.to(device)

        # Add a batch dimension
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

            for props in regions:
                plaque_counts[label] += 1
                data_record = { 'image_name': img_name,  'label': label, 'confidence': scores[i], 'brown_pixels': total_brown_pixels, 'core': plaque_counts["Cored"], 
                    'coarse_grained': plaque_counts["Coarse-Grained"], 'diffuse': plaque_counts["Diffuse"], 'caa': plaque_counts["CAA"], 'centroid': props.centroid, 
                    'eccentricity': props.eccentricity,  'area': props.area,  'equivalent_diameter': props.equivalent_diameter, 'mask_present': mask_present}
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



    def process_tile(self, batch):
        img_name, input_tensor, image = batch["key"],batch["tensor"],batch["array"]
        df = self.get_outputs_nms(input_tensor,image,img_name, score_threshold = 0.6, iou_threshold = 0.5)
        return df


    def process_slide_mpp(self, slide_name):
        start = timer()
        try:
            num_tiles = tiling_nosaving.slide_tile_map(slide_name, self.slide_dir, self.tile_dir, (1024,1024), f=tiling_nosaving.crop_lambda("", slide_name))
            num_tiles= [x for x in num_tiles if (x[1].shape[1]==1024) & (x[1].shape[0]==1024) & (x[1].shape[2]==3)]
            dataloader = self.create_dataloader(num_tiles, batch_size=8, shuffle=False, num_workers=0)
            print("count of tiles: ", len(num_tiles))
            print("All tiles extraction time: ", timer() - start)
            folder_name = os.path.join(self.result_save_dir, slide_name)
            self.make_result_dirs(folder_name)
            results = []
            for batch in dataloader:
                df = self.process_tile(batch)
                results.append(df)
            if len(results)>0:
                final_df = pd.concat(results, ignore_index=True)
                final_df["total_processing_time"] = timer()-start
                final_df.to_csv(self.quantify_path, index=False)
                print("total processing time",  (timer()-start)/60, "sec")
            else:
                print("no output generated")
        except:
            f = open("/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/New-Minerva-Data-Test/filenotrun2.txt", "a")
            f.write(slide_name)
            f.close()
            pass
            
        
        #random_array = np.random.rand(1024, 1024, 3)
        #numpy_arrays = [(str(i), np.random.randn(1024, 1024,3).astype(np.uint8)) for i in range(20)]
        # If you want the values to be integers between 0 and 255 (like an image), you can do this:
        #random_image = (random_array * 255).astype(np.uint8)
        #num_tiles = num_tiles[:1000]

            
        #if len(num_tiles)>5000:
        #    num_workers = 4
        #else:
        #    num_workers = 6
            
        #results = self.parallel_batch_processing(dataloader,num_workers)

       
        

    
    
    def generate_results_mpp(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval().to(device)
        slide_names = self.slide_list
        tile_dirs = glob.glob(os.path.join(self.tile_dir, "*.npy"))
        for slide_name in tqdm(slide_names):
            print("processing slide ", slide_name)
            self.process_slide_mpp(slide_name)
        
        
        
if __name__ == "__main__":
    #try:
    #    torch.multiprocessing.set_start_method('spawn')
    #except RuntimeError:
    #    pass
    tile_dir =  '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/New-Minerva-Data-Test/tiles-npy'
    slide_dir = "/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/New-Minerva-Data/finkbeiner-124208/Images"
    
    #all_files = glob.glob(os.path.join(slide_dir,"*.mrxs"))
    #vips_img_fnames =  [i for i in all_files if i.endswith("_1_AmyB_1.mrxs")] # keeping only AMYB-MFG region images
    #vips_img_fnames =  [i.split("/")[-1].split(".")[0] for i in vips_img_fnames]
    #vips_img_fnames = pd.read_csv("/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/New-Minerva-Data-Test/files_to_run_monika.csv")["slide_name"].values
    vips_img_fnames = pd.read_csv("/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/New-Minerva-Data-Test/files_to_run_vivek.csv")["slide_name"].values
    vips_img_fnames = pd.read_csv("/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/New-Minerva-Data-Test/filenotrun.csv")["WSI_name"].values
    print(vips_img_fnames)
    tile_size = (1024, 1024)
    model_name = "/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/runpod_mrcnn_models/yp2mf3i8_epoch=108-step=872.ckpt"
    model = LitMaskRCNN.load_from_checkpoint(model_name)
    explain = ExplainPredictions(model, model_input_path = model_name, slide_dir= slide_dir, slide_list = vips_img_fnames, tile_dir= tile_dir, 
                                detection_threshold=0.6)
    explain.generate_results_mpp()
    