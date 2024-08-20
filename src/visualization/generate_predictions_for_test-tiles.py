from genericpath import exists
import pdb
from pickletools import uint8
from turtle import pd
import torchvision
import torch


import os
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
# from torchknickknacks import modelutils
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from pytorch_grad_cam.utils.image import show_cam_on_image

import numpy as np
import cv2
import requests
from pytorch_grad_cam.ablation_layer import AblationLayerFasterRCNN
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform
from pytorch_grad_cam.ablation_layer import AblationLayerFasterRCNN
import random
import glob
from skimage.measure import label, regionprops, regionprops_table
from skimage import data, filters, measure, morphology
import pandas as pd
import wandb
from tqdm import tqdm
from skimage import data
from skimage.color import rgb2hed, hed2rgb
import sys
sys.path.insert(0, '../')
#from models.model_mrcnn import _default_mrcnn_config, build_default
from models_pytorch_lightning.model_mrcnn_config import _default_mrcnn_config, build_default
from features import build_features
from data.test_data import tiling_nosaving
from models_pytorch_lightning.generalized_mask_rcnn_pl import LitMaskRCNN
from utils.helper_functions import evaluate_metrics, get_outputs, compute_iou, evaluate_mask_rcnn
from features import build_features
from timeit import default_timer as timer 
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
import multiprocessing


multiprocessing.set_start_method('spawn', force=True)


class ExplainPredictions():
    # TODO fix the visualization flags
    def __init__(self, model, model_input_path, slide_dir, slide_list, tile_dir, detection_threshold, wandb, save_result, ablation_cam, save_thresholds):
        self.model = model
        self.model_input_path = model_input_path
        self.slide_dir = slide_dir
        self.slide_list = slide_list
        self.tile_dir = tile_dir
        self.detection_threshold = detection_threshold
        self.wandb = wandb
        self.save_result = save_result
        self.ablation_cam = ablation_cam
        self.save_thresholds = save_thresholds
        self.class_names = ['Cored', 'Diffuse', 'Coarse-Grained', 'CAA']
        #self.class_to_colors = {'Core': (255, 0, 0), 'Neuritic' : (0, 0, 255), 'Diffuse': (0,255,0), 'CAA':(225, 255, 0)}
        self.class_to_colors = {'Cored': (255, 0, 0), 'Diffuse' : (0, 0, 255), 'Coarse-Grained': (0,255,0), 'CAA':(225, 255, 0)}
        #self.result_save_dir= "/mnt/new-nas/work/data/npsad_data/vivek/reports/figures/"
        self.result_save_dir= os.path.join( "/home/mahirwar/Desktop/Monika/npsad_data/vivek/reports/New-Minerva-Data-output", self.model_input_path.split("/")[-1])
        self.colors = np.random.uniform(0, 255, size=(len(self.class_names), 3))
        self.column_names = ["image_name", "region", "region_mask", "label", 
                            "confidence", "brown_pixels", "centroid", 
                            "eccentricity", "area", "equivalent_diameter","mask_present"]
        self.results_path = ""
        self.masks_path = "" 
        self.detections_path = "" 
        self.ablations_path = ""
        self.quantify_path = ""
      
    def get_brown_pixel_cnt(self, img, img_name):

        # Separate the stains from the IHC image
        ihc_hed = rgb2hed(img)

        # Create an RGB image for each of the stains
        null = np.zeros_like(ihc_hed[:, :, 0])
        ihc_h = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
        ihc_e = hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1))
        ihc_d = hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))
        ihc_d = ihc_d.astype('float32')

        gray = cv2.cvtColor(ihc_d, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        x = gray[gray<0.35]


        if len(x) != 0:
            if self.save_thresholds:
                # Display
                fig, axes = plt.subplots(2, 2, figsize=(7, 6), sharex=True, sharey=True)
                ax = axes.ravel()

                ax[0].imshow(img)
                ax[0].set_title("Original image")

                ax[1].imshow(ihc_h)
                ax[1].set_title("Hematoxylin")

                ax[2].imshow(gray)
                ax[2].set_title("gray")  # Note that there is no Eosin stain in this image

                ax[3].imshow(ihc_d)
                ax[3].set_title("DAB")

                for a in ax.ravel():
                    a.axis('off')

                fig.tight_layout()
            
                
                final_save_path = img_name + "_count_threhold.png"
                final_save_path = os.path.join(self.pixel_count_path, final_save_path)
                fig.savefig(final_save_path)
                plt.close()
            return len(x)
        return 0

        
    def check_memory(self):
        memory_info = psutil.virtual_memory()
        available_memory = memory_info.available / (1024 ** 3)
        return available_memory
    
    
    def get_outputs_nms(self, input_tensor, model, score_threshold = 0.5, iou_threshold = 0.5):
        mask_list = []
        label_list = []
        score_list =[]
        box_list = []
        with torch.no_grad():
            # forward pass of the image through the modle
            outputs = model(input_tensor)
        boxes = outputs[0]['boxes']
        labels = outputs[0]['labels']
        scores = outputs[0]['scores']
        masks = outputs[0]['masks']
        #print(boxes.shape,labels.shape, masks.shape)
        # Apply score threshold
        keep = scores > score_threshold
        boxes, labels, scores, masks = boxes[keep], labels[keep], scores[keep], masks[keep]
        #print(boxes.shape,labels.shape, masks.shape)
        #print(keep)
        keep = torchvision.ops.nms(boxes, scores, iou_threshold)
        boxes, labels, scores, masks = boxes[keep], labels[keep], scores[keep], masks[keep]
        #print(keep)
        #print(boxes.shape,labels.shape, masks.shape)
        scores = list(scores.detach().cpu().numpy())
        #print(scores)
        masks = list((masks>0.5).detach().cpu().numpy())
        #masks = list((masks>0.5).squeeze().detach().cpu().numpy())
        #print(masks.shape)
        boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]  for i in boxes.detach().cpu()]
        labels = list(labels.detach().cpu().numpy())
        labels = [self.class_names[i-1] for i in labels]
        return masks, boxes, labels, scores
            


    def draw_segmentation_map(self, image, masks, boxes, labels):
        alpha = 1 
        beta = 0.6 # transparency for the segmentation map
        gamma = 0 # scalar added to each 
        segmentation_map = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        result_masks = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        masks = [ masks[i].squeeze() for i in range(len(masks))]
        for i in range(len(masks)):

            # TODO fix the color segmentation masks
            red_map = np.zeros_like(masks[i]).astype(np.uint8)
            green_map = np.zeros_like(masks[i]).astype(np.uint8)
            blue_map = np.zeros_like(masks[i]).astype(np.uint8)
            
            # apply a randon color mask to each object
            rect_color = (0,0,0)
            color = self.colors[random.randrange(0, len(self.colors))]
            red_map[masks[i] == 1], green_map[masks[i] == 1], blue_map[masks[i] == 1]  = self.class_to_colors[labels[i]]
            result_masks[masks[i] == 1] = 255
            # combine all the masks into a single image
            # change the format of mask to W,H, C

            # segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
            #convert the original PIL image into NumPy format
            image = np.array(image)
            # convert from RGB to OpenCV BGR format
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # apply mask on the image
            # cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, image)
            # draw the bounding boxes around the objects
            cv2.rectangle(image, boxes[i][0], boxes[i][1], color=rect_color, 
                        thickness=2)
            # Get the centre coords of the rectangle-plaque-detection/src/visualizat
            x1 = boxes[i][0][0]
            y1 = boxes[i][0][1]
            x2 = boxes[i][1][0]
            y2 = boxes[i][1][1]
            x = int((x1 + x2) / 2)
            y = int((y1+y2) / 2)

            
            # put the label text above the objects
            cv2.putText(image , labels[i], (x1, y1-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 
                        thickness=2, lineType=cv2.LINE_AA)
            
            # Convert Back
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, result_masks

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
    
    def predict(self, input_tensor, model, device, detection_threshold):
        outputs = model(input_tensor)
        # i- 1 zero indexing - the model outputs a non zero indexing format ( 1, 2, 3)
        pred_classes = [self.class_names[i-1] for i in outputs[0]['labels'].cpu().numpy()]
        pred_labels = outputs[0]['labels'].cpu().numpy()
        pred_scores = outputs[0]['scores'].detach().cpu().numpy()
        pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
        
        boxes, classes, labels, indices = [], [], [], []
        for index in range(len(pred_scores)):
            if pred_scores[index] >= detection_threshold:
                boxes.append(pred_bboxes[index].astype(np.int32))
                classes.append(pred_classes[index])
                labels.append(pred_labels[index])
                indices.append(index)
        boxes = np.int32(boxes)
        return boxes, classes, labels, indices

    def draw_boxes(self, boxes, labels, classes, image):
        for i, box in enumerate(boxes):
            color = self.colors[labels[i]-1]
            cv2.rectangle(
                image,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                color, 2
            )
            cv2.putText(image, classes[i], (int(box[0]), int(box[1] + 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1,
                        lineType=cv2.LINE_AA)
        return image

    def make_result_dirs(self, folder_name):
        #folder_name = self.wandb.name + "_" + folder_name
        
        save_path = os.path.join(self.result_save_dir, folder_name)
        results_path = os.path.join(save_path, "results")
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        
        detections_path = os.path.join(save_path, "detections")
        if not os.path.exists(detections_path):
            os.makedirs(detections_path)
        
        masks_path = os.path.join(save_path, "masks")
        if not os.path.exists(masks_path):
            os.makedirs(masks_path)
        
        ablations_path = os.path.join(save_path, "ablations")
        if not os.path.exists(ablations_path):
            os.makedirs(ablations_path)
        
        pixel_count_path = os.path.join(save_path, "pixel_count")
        if not os.path.exists(pixel_count_path):
            os.makedirs(pixel_count_path)
        
        csv_name = folder_name + "_quantify.csv"
        quantify_path = os.path.join(save_path, csv_name)

        self.results_path = results_path
        self.masks_path = masks_path
        self.detections_path = detections_path
        self.ablations_path = ablations_path
        self.quantify_path = quantify_path
        self.pixel_count_path = pixel_count_path

    def quantify_plaques(self, df, img_name, result_img, result_masks, boxes, labels, scores, total_brown_pixels):
        '''This function will take masks image and generate attributes like plaque
        count, area, eccentricity'''
        csv_result = []
        for i in range(len(labels)):
            props = {}
            data = {}
            # Here x and y axis are flipped
            total_core_plaques = 0
            total_cg_plaques = 0
            total_diffused_plaques = 0
            total_caa_plaques = 0

            if len(boxes)!= 0:

                x1 = boxes[i][0][1]
                x2 =  boxes[i][1][1]
                y1 = boxes[i][0][0]
                y2 = boxes[i][1][0]

               
                cropped_img = result_img[x1:x2, y1:y2]
                cropped_img_mask = result_masks[x1:x2, y1:y2]
                unique_pixel = np.unique(cropped_img_mask)

                ret, bw_img = cv2.threshold(cropped_img_mask,0,255,cv2.THRESH_BINARY)

                kernel = np.ones((5,5),np.uint8)
                
                # Closing operation Dilation followed by erosion
                closing = cv2.morphologyEx(bw_img, cv2.MORPH_CLOSE, kernel)
                regions = regionprops(closing)
                if 0 in np.unique(closing):
                    mask_present=1
                else:
                    mask_present=0
                for props in regions:
                    if labels[i] == "Cored":
                        total_core_plaques+=1
                    elif labels[i] == "Coarse-Grained":
                        total_cg_plaques+=1
                    elif labels[i] == "Diffuse":
                        total_diffused_plaques+=1
                    elif labels[i] == "CAA":
                        total_caa_plaques+=1
                    
                    data_record = pd.DataFrame.from_records([{ 'image_name': img_name, 'label': labels[i] , 'confidence': scores[i],
                                                               'brown_pixels': total_brown_pixels,
                                                               'core': total_core_plaques, 'coarse_grained': total_cg_plaques, 'diffuse': total_diffused_plaques,
                                                               'caa':total_caa_plaques,
                                                               'centroid': props.centroid, 'eccentricity': props.eccentricity, 
                                                               'area': props.area, 'equivalent_diameter': props.equivalent_diameter, 'mask_present':mask_present}])
                    #wandb_result.append([img_name, wandb.Image(cropped_img), wandb.Image(cropped_img_mask), labels[i], scores[i], 
                    #                     total_brown_pixels, props.centroid, props.eccentricity, props.area, props.equivalent_diameter,mask_present ])

                    df = pd.concat([df, data_record], ignore_index=True)
                   
        
        return df
    
 
    def process_tile(self, img_name, image,total_image_pixels, total_brown_pixels):
        print("processing tiles", img_name )
        total_image_pixels+= image.shape[0] * image.shape[1]
         # Check if image has alpha channel
        if image.shape[2] == 4:
            image = image[:,:, :3]
        start = timer()
        input_tensor, image_float_np = self.prepare_input(image)
        print("tensor creation:", timer()-start)
        start = timer()
        masks, boxes, labels, scores = self.get_outputs_nms(input_tensor, self.model, self.detection_threshold, 0.5)
        print("get output:", timer()-start)
        start = timer()
        result_img, result_masks = self.draw_segmentation_map(image, masks, boxes, labels)
        print("draw segmentation:", timer()-start)
        start = timer()
        total_brown_pixels+= self.get_brown_pixel_cnt(image, img_name)
        print("brown pixel:", timer()-start)
        #df, wandb_result = self.quantify_plaques(df, wandb_result, img_name, result_img, result_masks, boxes, labels, scores, total_brown_pixels)
        start = timer()
        df = self.quantify_plaques(pd.DataFrame(), img_name, result_img, result_masks, boxes, labels, scores, total_brown_pixels)
        print("quantify plaques:", timer()-start)
        return df
    
    def process_slide_mpp(self, slide_name):
        start = timer()
        num_tiles = tiling_nosaving.slide_tile_map(slide_name, self.slide_dir, self.tile_dir, (1024,1024), f=tiling_nosaving.crop_lambda("", slide_name))
        #random_array = np.random.rand(1024, 1024, 3)

        # If you want the values to be integers between 0 and 255 (like an image), you can do this:
        #random_image = (random_array * 255).astype(np.uint8)
        print("count of tiles: ", len(num_tiles))
        print("All tiles extraction time: ", timer() - start)
        folder_name = os.path.join(self.result_save_dir, slide_name)
        self.make_result_dirs(folder_name)
        total_core_plaques = 0
        total_neuritic_plaques = 0
        total_diffused_plaques = 0
        total_brown_pixels = 0
        total_image_pixels = 0
        results = []
        #df = self.process_tile("XE18-045_1_AmyB_1x_90112_y_130048.png",random_image, total_image_pixels, total_brown_pixels)
        
        with ProcessPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(self.process_tile, img_name, image, total_image_pixels, total_brown_pixels): (img_name, image) for img_name, image in num_tiles}
            for future in concurrent.futures.as_completed(futures):
                img_name, image = futures[future]
                try:
                    df = future.result()
                    print(df)
                    #results.append((df, wandb_result))
                    results.append(df)

                except Exception as exc:
                    print(f'{img_name} generated an exception: {exc}')
        
        results.append(df)
        df = pd.concat(results, ignore_index=True)
        df["total_processing_time"] = timer()-start
        df.to_csv(self.quantify_path, index=False)
        
    
    
    def process_slide(self, slide_name):
        start = timer()
        num_tiles = tiling_nosaving.slide_tile_map(slide_name, self.slide_dir, self.tile_dir, (1024,1024), f=tiling_nosaving.crop_lambda("", slide_name))
        
        print("tile extraction time: ", timer() - start)

        folder_name = os.path.join(self.result_save_dir, slide_name)
        self.make_result_dirs(folder_name)

        results = []
        for img_name, image in num_tiles:
            total_core_plaques = 0
            total_neuritic_plaques = 0
            total_diffused_plaques = 0
            total_brown_pixels = 0
            total_image_pixels = 0
            df = self.process_tile(img_name, image,total_image_pixels, total_brown_pixels)
            results.append(df)


        final_df = pd.concat(results, ignore_index=True)
        final_df["total_processing_time"] = timer()-start
        final_df.to_csv(self.quantify_path, index=False)
        print("total_processing_time: ", timer()-start )
        available_memory = self.check_memory()
        print(f"Available memory after processing {slide_name}: {available_memory:.2f} GB")
        
    
    
    
        
    def process_all_slides(self):
        slide_names = self.slide_list
        tile_dirs = glob.glob(os.path.join(self.tile_dir, "*.npy"))
        for slide_name in tqdm(slide_names):
            print("processing slide ", slide_name)
            self.process_slide_mpp(slide_name)
        return True
            

 
 
    def generate_results_mpp(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval().to(device)
        self.process_all_slides()

 
    def generate_results_multi_slides(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval().to(device)
        tile_dirs = glob.glob(os.path.join(self.tile_dir, "*.npy"))
        slide_names = self.slide_list
        results = []
        
        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(self.process_slide, slide_name) for slide_name in slide_names]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing slide: {e}")
        
                 
        
        
 
 
    def generate_results(self):
        # This will help us create a different color for each class
        # Load Trained 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval().to(device)
        tile_dirs = glob.glob(os.path.join(self.tile_dir, "*.npy"))
        #slide_names = ['.'.join(file.split('.')[:-1]) for file in next(os.walk(slide_dir))[2]]
        slide_names = self.slide_list
        out_dir =""
        times= []
        for slide_name in tqdm(slide_names):
            start=timer()
            num_tiles = tiling_nosaving.slide_tile_map(slide_name, slide_dir, tile_dir, tile_size, f=tiling_nosaving.crop_lambda(out_dir, slide_name))           
            print("tile extraction time: ",timer()-start)
            
            folder_name =  os.path.join(self.result_save_dir, slide_name)
            # make all necessary folders
            self.make_result_dirs(folder_name)
            i = 0
            df = pd.DataFrame()
            wandb_result = []
            total_core_plaques = 0
            total_neuritic_plaques = 0
            total_diffused_plaques = 0
            total_brown_pixels = 0
            total_image_pixels = 0
            print(len(num_tiles))
            for img_name, image in num_tiles:
                total_image_pixels+= image.shape[0] * image.shape[1]
                # Check if image has alpha channel
                if image.shape[2] == 4:
                    image = image[:,:, :3]
                timer1 = timer()
                input_tensor, image_float_np = self.prepare_input(image)
                print("prepare input time: ", timer()-timer1)
                timer1 = timer()
                masks, boxes, labels, scores = self.get_outputs_nms(input_tensor, self.model, self.detection_threshold, 0.5)
                print("getting output nms time: ", timer()-timer1)
                #f1_mean, labels_matched,actual_labels,pred_labels, scores =  evaluate_metrics(targets, masks, labels,scores )
                timer1 = timer()
                result_img, result_masks = self.draw_segmentation_map(image, masks, boxes, labels)
                print("draw segmentation time: ", timer()-timer1)
                total_brown_pixels+= self.get_brown_pixel_cnt(image, img_name)
                timer1 = timer()
                df, wandb_result = self.quantify_plaques(df, wandb_result, img_name, result_img, result_masks, boxes, labels, scores, total_brown_pixels)
                print("quantify plaque time: ", timer()-timer1)
                if (self.save_result==True) and (len(boxes)!=0):
                    mask_img_name = img_name +  "_masks.png"
                    mask_save_path = os.path.join(self.masks_path, mask_img_name)

                    # Plot masks
                    cv2.imwrite(mask_save_path, result_masks)

                    # Plot detections
                    detection_img_name = img_name + "_detection.png"
                    detection_save_path = os.path.join(self.detections_path, detection_img_name)
                    bgr_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(detection_save_path, bgr_img)

                    # Plot Results
                    plt.figure(figsize=(10,10))
                    plt.title("Model Prediction")

                    img_array = [result_img, result_masks]

                    for j in range(2):
                        plt.subplot(1, 2, j+1)
                        plt.imshow(img_array[j])
                    
                    result_img_name = img_name +  "_result.png"
                    result_save_path = os.path.join(self.results_path, result_img_name)
                    plt.savefig(result_save_path)
                    plt.close()

                # plt.show()
                if self.ablation_cam:
                    # Ablation CAM
                    boxes, classes, labels, indices = self.predict(input_tensor, model, device, self.detection_threshold)
                    target_layers = [model.backbone]
                    targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=boxes)]
                

                    cam = AblationCAM(model,
                                target_layers, 
                                use_cuda=torch.cuda.is_available(), 
                                reshape_transform=fasterrcnn_reshape_transform,
                                ablation_layer=AblationLayerFasterRCNN())

                    grayscale_cam = cam(input_tensor, targets=targets)
                    # Take the first image in the batch:
                    grayscale_cam = grayscale_cam[0, :]
                    cam_image = show_cam_on_image(image_float_np, grayscale_cam, use_rgb=True)
                    # And lets draw the boxes again:
                    image_with_bounding_boxes = self.draw_boxes(boxes, labels, classes, cam_image)

                    # plt.imshow(image_with_bounding_boxes)
                    plt.figure(figsize=(10,10))
                    plt.title("Ablation Cam")
                    ablation_list = [grayscale_cam, image_with_bounding_boxes]
                    for k in range(2):
                        plt.subplot(1, 2, k+1)
                        plt.imshow(ablation_list[k])


                    if self.save_result:
                        ablation_img_name = img_name +  "_ablation_cam.png"
                        save_path_ablation = os.path.join(self.ablations_path, ablation_img_name)
                        plt.savefig(save_path_ablation)
                i = i + 1
                # plt.show()

            #print("Total area of brown pixel", (total_brown_pixels))
            df["processing_time"] = timer()-start
            df.to_csv(self.quantify_path, index=False)
            print(timer()-start)
            #times.append(timer()-start)
        #time_df = pd.DataFrame({"slide_names":slide_names, "time":times})
        #time_df.to_csv(os.path.join(self.result_save_dir,"slide_processing_time.csv"))
            #test_table = wandb.Table(data=wandb_result, columns=self.column_names)
            # self.wandb.log({'quantifications': test_table})


    
if __name__ == "__main__":
    #tile_dir = '/gladstone/finkbeiner/steve/work/data/npsad_data/slide_masks'
    tile_dir =  '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/New-Minerva-Data-Test/tiles-npy'
    #slide_dir = '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/amy-def-mfg-test'
    slide_dir = "/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/New-Minerva-Data/finkbeiner-124208/Images"
    #all_files = glob.glob(os.path.join(slide_dir,"*.mrxs"))
    #vips_img_fnames =  [i for i in all_files if i.endswith("_1_AmyB_1.mrxs")] # keeping only AMYB-MFG region images
    #vips_img_fnames =  [i.split("/")[-1].split(".")[0] for i in vips_img_fnames]
    vips_img_fnames = pd.read_csv("/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/New-Minerva-Data-Test/files_to_run_vivek.csv")["slide_name"].values
    print(vips_img_fnames)
    
    tile_size = (1024, 1024)

    # watch out for '.DS_Store'
    #slide_names = ['.'.join(file.split('.')[:-1]) for file in next(os.walk(slide_dir))[2]]
    #slide_names = slide_names[3:4]
    model_name = "/home/mahirwar/Desktop/Monika/npsad_data/vivek/runpod_mrcnn_models/yp2mf3i8_epoch=108-step=872.ckpt"
    #slide_names = slide_names[:5]
    test_config = dict(
        batch_size = 1,
        num_classes = 4
    )

    model_config = _default_mrcnn_config(num_classes=1 + test_config['num_classes']).config
    backbone, rpn, roi_heads, transform1 = build_default(model_config, im_size=1024)

    optim_config = dict(
        cls=torch.optim.Adam,
        defaults=dict(lr=0.00001,weight_decay=1e-6) 
    )
    
    model = LitMaskRCNN.load_from_checkpoint(model_name)
    print(model)
    #print(vips_img_fnames[15:25])
    explain = ExplainPredictions(model, model_input_path = model_name, slide_dir= slide_dir, slide_list = vips_img_fnames, tile_dir= tile_dir, 
                                    detection_threshold=0.6, wandb=None, save_result=False, ablation_cam=False, save_thresholds=False)
    explain.generate_results_mpp()