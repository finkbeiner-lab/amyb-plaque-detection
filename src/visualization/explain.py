from genericpath import exists
import pdb
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
from models.model_mrcnn import _default_mrcnn_config, build_default
from PIL import Image 




class ExplainPredictions():
    
    # TODO fix the visualization flags
    def __init__(self, model, model_input_path, test_input_path, detection_threshold, wandb, save_result, ablation_cam, save_thresholds):
        self.model = model
        self.model_input_path = model_input_path
        self.test_input_path = test_input_path
        self.detection_threshold = detection_threshold
        self.wandb = wandb
        self.save_result = save_result
        self.ablation_cam = ablation_cam
        self.save_thresholds = save_thresholds
        self.class_names = ['Unknown', 'Cored', 'Diffuse', 'Neuritic', 'CAA']
        # self.class_names = ['Unknown', 'Cored', 'Diffuse', 'CAA']
        self.class_to_colors = {'Cored': (255, 0, 0), 'Neuritic' : (0, 0, 255), 'Diffuse': (0,255,0), 'CAA':(225, 255, 0)}
        #TODO change this to nas location later
        self.result_save_dir= "/mnt/new-nas/work/data/npsad_data/vivek/reports/figures"
        self.colors = np.random.uniform(0, 255, size=(len(self.class_names), 3))
        self.column_names = ["image_name", "region", "region_mask", "label", 
                            "confidence", "brown_pixels", "centroid", 
                            "eccentricity", "area", "equivalent_diameter"]
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
        # segmentation_map = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        # result_masks = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        segmentation_map = np.zeros((1024, 1024), dtype=np.uint8)
        result_masks = np.zeros((1024, 1024), dtype=np.uint8)

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
        #TODO: Remove the resize for our internal test data
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(1024),
            torchvision.transforms.ToTensor(),
        ])

        #TODO: change back to transform(image)

        input_tensor = transform(transforms.ToPILImage()(image))
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

    def make_result_dirs(self, folder_name):

    
        # folder_name = "runtest" + "_" + 
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
        
        #TODO remove this folder split
        csv_name = folder_name.split("/")[1] + "_quantify.csv"
        quantify_path = os.path.join(save_path, csv_name)

        self.results_path = results_path
        self.masks_path = masks_path
        self.detections_path = detections_path
        self.ablations_path = ablations_path
        self.quantify_path = quantify_path
        self.pixel_count_path = pixel_count_path

    def quantify_plaques(self, df, wandb_result, img_name, result_img, result_masks, boxes, labels, scores, total_brown_pixels):
        '''This function will take masks image and generate attributes like plaque
        count, area, eccentricity'''

        csv_result = []
        
    
        for i in range(len(labels)):

            props = {}
            data = {}
            # Here x and y axis are flipped
            total_core_plaques = 0
            total_neuritic_plaques = 0
            total_diffused_plaques = 0
            total_caa_plaques = 0


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

                    if labels[i] == "Cored":
                        total_core_plaques+=1
                    elif labels[i] == "Neuritic":
                        total_neuritic_plaques+=1
                    elif labels[i] == "Diffuse":
                        total_diffused_plaques+=1
                    elif labels[i] == "CAA":
                        total_caa_plaques+=1
                    
                    data_record = pd.DataFrame.from_records([{ 'image_name': img_name, 'label': labels[i] , 'confidence': scores[i],
                                                               'brown_pixels': total_brown_pixels,
                                                               'core': total_core_plaques, 'neuritic': total_neuritic_plaques, 'diffuse': total_diffused_plaques,
                                                               'caa': total_caa_plaques, 'centroid': props.centroid, 'eccentricity': props.eccentricity, 
                                                               'area': props.area, 'equivalent_diameter': props.equivalent_diameter}])
                    # TODO Uncomment this later
                    # wandb_result.append([img_name, wandb.Image(cropped_img), wandb.Image(cropped_img_mask), labels[i], scores[i], 
                    #                      total_brown_pixels, props.centroid, props.eccentricity, props.area, props.equivalent_diameter])

                    df = pd.concat([df, data_record], ignore_index=True)
                   
        
        return df, wandb_result
 
    def generate_results(self):
        # This will help us create a different color for each class
        # Load Trained 
        
        self.model.load_state_dict(torch.load(self.model_input_path))
    
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval().to(device)

        test_folders = glob.glob(os.path.join(self.test_input_path, "*"))
        
        # Test images from each WSI folder
        test_folders = sorted(test_folders)

        # Create Run folder
        run_name_folder = self.wandb.name
        if not os.path.exists(run_name_folder):
            os.makedirs(run_name_folder)


        for test_folder in tqdm(test_folders):

            print("\n==========", test_folder)
          
            folder_name = os.path.basename(test_folder)
      

            # if folder_name == match_clinical_list:

            folder_name = os.path.join(run_name_folder, folder_name)


            if folder_name == "labels":
                continue
            
            # make all necessary folders
            self.make_result_dirs(folder_name)
            #TODO change the file extension of .png for internal validation
            images = glob.glob(os.path.join(test_folder, '*.png'))

            i = 0
            df = pd.DataFrame()
            wandb_result = []
            total_core_plaques = 0
            total_neuritic_plaques = 0
            total_diffused_plaques = 0
            total_brown_pixels = 0
            total_image_pixels = 0

            for img in tqdm(images):
                result_img = 0
                img_name = os.path.basename(img).split('.')[0]
        
                image = Image.open(img)
                image = np.array(image.resize((1024, 1024)))
                

                total_image_pixels+= image.shape[0] * image.shape[1]
                # Check if image has alpha channel
                if image.shape[2] == 4:
                    image = image[:,:, :3]

                input_tensor, image_float_np = self.prepare_input(image)
                masks, boxes, labels, scores = self.get_outputs(input_tensor, self.model, self.detection_threshold)

                if len(masks) != 0 and len(boxes) !=0:

                    if np.count_nonzero(masks) == 0:
                        continue
                    
                    # print(" processing------", img_name)
                    
                
                    result_img, result_masks = self.draw_segmentation_map(image, masks, boxes, labels)

                    # total_brown_pixels+= self.get_brown_pixel_cnt(image, img_name)

                    df, wandb_result = self.quantify_plaques(df, wandb_result, img_name, result_img, result_masks, boxes, labels, scores, total_brown_pixels)

                    if self.save_result:
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

                        plt.imshow(grayscale_cam)
                        # ablation_list = [grayscale_cam, image_with_bounding_boxes]
                        # for k in range(2):
                        #     plt.subplot(1, 2, k+1)
                        #     plt.imshow(ablation_list[k])


                        if self.save_result:
                            ablation_img_name = img_name +  "_ablation_cam.png"
                            save_path_ablation = os.path.join(self.ablations_path, ablation_img_name)
                            plt.savefig(save_path_ablation)
                    i = i + 1
                    # plt.show()
                

            if total_image_pixels != 0:
                print("Total area of brown pixel", (total_brown_pixels/ total_image_pixels)*100)
            df.to_csv(self.quantify_path, index=False)
            test_table = wandb.Table(data=wandb_result, columns=self.column_names)
            # self.wandb.log({'quantifications': test_table})   

                
        
if __name__ == "__main__":

    
    # input_path = "/mnt/new-nas/work/data/npsad_data/vivek/Datasets/amyb_wsi/test-patients/images"
    # input_path = "/mnt/new-nas/work/data/npsad_data/vivek/reports/Manuscript"
    # input_path = "/mnt/new-nas/work/data/npsad_data/vivek/Datasets/amyb_wsi/test-missing/images"
    input_path = "/mnt/new-nas/work/data/npsad_data/vivek/Datasets/neuritic/images"

    #input_path = "/mnt/new-nas/work/data/npsad_data/vivek/reports/Manuscript/ablation-cam"
    # input_path = "/mnt/new-nas/work/data/npsad_data/vivek/Datasets/UCDavis-Dataset/tiles/train/"
    # model_input_path = "/mnt/new-nas/work/data/npsad_data/vivek/models/eager-frog-489_mrcnn_model_100.pth"
    model_input_path = "/home/vivek/Projects/amyb-plaque-detection/models/swift-brook-705_mrcnn_model_100.pth"

    test_config = dict(
        batch_size = 2,
        num_classes = 4
    )

    model_config = _default_mrcnn_config(num_classes=1 + test_config['num_classes']).config
    model = build_default(model_config, im_size=1024)

    # Use the Run ID from train_model.py here if you want to add some visualizations after training has been done
    # with wandb.init(project="nps-ad", id = "17vl5roa", entity="hellovivek", resume="allow"):\
     
    
    run = wandb.init(project="nps-ad-vivek",  entity="hellovivek")
    explain = ExplainPredictions(model, model_input_path = model_input_path, test_input_path=input_path, 
                                    detection_threshold=0.75, wandb=run, save_result=True, ablation_cam=True, save_thresholds=False)
    explain.generate_results()
