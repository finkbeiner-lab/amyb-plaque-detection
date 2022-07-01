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




class ExplainPredictions():
    
    # TODO fix the visualization flags
    def __init__(self, model_input_path, test_input_path, detection_threshold, wandb, save_result, ablation_cam):
        self.model_input_path = model_input_path
        self.test_input_path = test_input_path
        self.detection_threshold = detection_threshold
        self.wandb = wandb
        self.save_result = save_result
        self.ablation_cam = ablation_cam
        self.class_names = ['Unknown', 'Core', 'Diffuse', 'Neuritic']
        self.class_to_colors = {'Core': (255, 0, 0), 'Neuritic' : (0, 0, 255), 'Diffuse': (0,255,0)}
        self.result_save_dir= "../../reports/figures/"
        self.colors = np.random.uniform(0, 255, size=(len(self.class_names), 3))
        self.masks_path = ""
        self.column_names = ["image_name", "region", "region_mask", "label", "confidence", "centroid", "eccentricity", "area", "equivalent_diameter"]
      
    def get_outputs(self, input_tensor, model, threshold):
        with torch.no_grad():
            # forward pass of the image through the modle
            outputs = model(input_tensor)
        
        # get all the scores
        scores = list(outputs[0]['scores'].detach().cpu().numpy())
        # print("\n scores", max(scores))
        # index of those scores which are above a certain threshold
        thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
        thresholded_preds_count = len(thresholded_preds_inidices)

        scores = scores[:thresholded_preds_count]
        # get the masks
        masks = (outputs[0]['masks']>0.5).squeeze().detach().cpu().numpy()
        # print("masks", masks)
        # discard masks for objects which are below threshold
        masks = masks[:thresholded_preds_count]
        # get the bounding boxes, in (x1, y1), (x2, y2) format
        boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]  for i in outputs[0]['boxes'].detach().cpu()]
        # discard bounding boxes below threshold value
        boxes = boxes[:thresholded_preds_count]
        # get the classes labels
        # print('labels', outputs[0]['labels'])
        labels = [self.class_names[i] for i in outputs[0]['labels']]
        labels = labels[:thresholded_preds_count]

        # [1,1,1, 2, 2, 2, 3, 3]
        return masks, boxes, labels, scores

    def draw_segmentation_map(self, image, masks, boxes, labels):
        alpha = 1 
        beta = 0.6 # transparency for the segmentation map
        gamma = 0 # scalar added to each 
        segmentation_map = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        result_masks = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

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
            # Get the centre coords of the rectangle
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
        pred_classes = [self.class_names[i] for i in outputs[0]['labels'].cpu().numpy()]
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
            color = self.colors[labels[i]]
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

    def make_result_dirs(self):

        results_path = os.path.join(self.result_save_dir, "results")
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        
        masks_path = os.path.join(self.result_save_dir, "masks")
        if not os.path.exists(masks_path):
            os.makedirs(masks_path)
        
        ablations_path = os.path.join(self.result_save_dir, "ablations")
        if not os.path.exists(ablations_path):
            os.makedirs(ablations_path)
        
        return results_path, masks_path, ablations_path

    def quantify_plaques(self, df, wandb_result, img_name, result_img, result_masks, boxes, labels, scores):
        '''This function will take masks image and generate attributes like plaque
        count, area, eccentricity'''

        csv_result = []
        
        for i in range(len(labels)):

            props = {}
            data = {}
            # Here x and y axis are flipped

            if len(boxes)!= 0:

                x1 = boxes[i][0][1]
                x2 =  boxes[i][1][1]
                y1 = boxes[i][0][0]
                y2 = boxes[i][1][0]

               
                cropped_img = result_img[x1:x2, y1:y2]
                cropped_img_mask = result_masks[x1:x2, y1:y2]

                ret, bw_img = cv2.threshold(cropped_img_mask,0,255,cv2.THRESH_BINARY)

                kernel = np.ones((5,5),np.uint8)
                
                # Closing operation Dilation followed by erosion
                closing = cv2.morphologyEx(bw_img, cv2.MORPH_CLOSE, kernel)
                regions = regionprops(closing)

                for props in regions:
                    data_record = pd.DataFrame.from_records([{ 'image_name': img_name, 'label': labels[i] , 'confidence': scores[i], 
                                                               'centroid': props.centroid, 'eccentricity': props.eccentricity, 
                                                               'area': props.area, 'equivalent_diameter': props.equivalent_diameter}])
                    wandb_result.append([img_name, self.wandb.Image(cropped_img), self.wandb.Image(cropped_img_mask), labels[i], scores[i], 
                                         props.centroid, props.eccentricity, props.area, props.equivalent_diameter])

                    df = pd.concat([df, data_record], ignore_index=True)
                   
        
        return df, wandb_result

  
       
        

        
    
    def f(x, w, z):
        print("f")
        for i in x:
            print(i)
        
    def generate_results(self):
        # This will help us create a different color for each class

        results_path, masks_path, ablations_path = self.make_result_dirs()
        quantify_path = os.path.join(self.result_save_dir, "quantify.csv")
        #used downstream by quantify plaques
        self.masks_path = masks_path

        # Load Trained Model
        model = torch.load(self.model_input_path)
    
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval().to(device)

        test_folders = glob.glob(os.path.join(self.test_input_path, "*"))
        
        # Test images from each WSI folder
        for test_folder in tqdm(test_folders):
            images = glob.glob(os.path.join(test_folder, '*.png'))

            i = 0
            df = pd.DataFrame()
            wandb_result = []
        
            for img in tqdm(images):
                img_name = os.path.basename(img).split('.')[0]
        
                image = np.array(Image.open(img))

                # Check if image has alpha channel
                if image.shape[2] == 4:
                    image = image[:,:, :3]

                input_tensor, image_float_np = self.prepare_input(image)
                masks, boxes, labels, scores = self.get_outputs(input_tensor, model, self.detection_threshold)
                
                result_img, result_masks = self.draw_segmentation_map(image, masks, boxes, labels)

                df, wandb_result = self.quantify_plaques(df, wandb_result, img_name, result_img, result_masks, boxes, labels, scores)

                if self.save_result:
                    mask_img_name = img_name +  "_masks.png"
                    mask_save_path = os.path.join(masks_path, mask_img_name)

                    cv2.imwrite(mask_save_path, result_masks)

                    # Plot the result and save
                    plt.figure(figsize=(10,10))
                    plt.title("Model Prediction")

                    img_array = [result_img, result_masks]

                    for j in range(2):
                        plt.subplot(1, 2, j+1)
                        plt.imshow(img_array[j])
                    
                    result_img_name = img_name +  "_result.png"
                    result_save_path = os.path.join(results_path, result_img_name)
                    plt.savefig(result_save_path)
                
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

                    plt.imshow(image_with_bounding_boxes)

                    if self.save_result:
                        ablation_img_name = img_name +  "_ablation_cam.png"
                        save_path_ablation = os.path.join(ablations_path, ablation_img_name)
                        plt.savefig(save_path_ablation)
                i = i + 1
                # plt.show()

            df.to_csv(quantify_path, index=False)
            test_table = self.wandb.Table(data=wandb_result, columns=self.column_names)
            self.wandb.log({'quantifications': test_table})
          
                


            
        
if __name__ == "__main__":

    
    input_path = '/home/vivek/Datasets/AmyB/amyb_wsi/test/images'
    model_input_path = '../../models/mrcnn_model_15.pth'
    with wandb.init(project="nps-ad", id = "17vl5roa", entity="hellovivek", resume="allow"):
        explain = ExplainPredictions(model_input_path = model_input_path, test_input_path=input_path, 
                                     detection_threshold=0.5, wandb=wandb, save_result=False)
        explain.generate_results(ablation_cam=False)
