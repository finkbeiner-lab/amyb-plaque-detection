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


class ExplainPredictions():

    def __init__(self, model_input_path, test_input_path, detection_threshold, ):
        self.model_input_path = model_input_path
        self.test_input_path = test_input_path
        self.detection_threshold = detection_threshold
        self.class_names = ['Unknown', 'Core', 'Diffuse', 'Neuritic']


    def get_outputs(self, input_tensor, model, threshold):
        with torch.no_grad():
            # forward pass of the image through the modle
            outputs = model(input_tensor)
        
        # get all the scores
        scores = list(outputs[0]['scores'].detach().cpu().numpy())
        print("\n scores", max(scores))
        # index of those scores which are above a certain threshold
        thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
        thresholded_preds_count = len(thresholded_preds_inidices)
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
        labels = [coco_names[i] for i in outputs[0]['labels']]

        # [1,1,1, 2, 2, 2, 3, 3]
        return masks, boxes, labels

    def draw_segmentation_map(self, image, masks, boxes, labels):
        alpha = 1 
        beta = 0.6 # transparency for the segmentation map
        gamma = 0 # scalar added to each 
        segmentation_map = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
        for i in range(len(masks)):
            red_map = np.zeros_like(masks[i]).astype(np.uint8)
            green_map = np.zeros_like(masks[i]).astype(np.uint8)
            blue_map = np.zeros_like(masks[i]).astype(np.uint8)
            
            # apply a randon color mask to each object
            rect_color = (0,0,0)
            color = COLORS[random.randrange(0, len(COLORS))]
            red_map[masks[i] == 1], green_map[masks[i] == 1], blue_map[masks[i] == 1]  = color
            # combine all the masks into a single image
            # change the format of mask to W,H, C

            segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
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
        return image, segmentation_map

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




# This will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

# model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model = torch.load('../../models/mrcnn_model_50.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.eval().to(device)


images = glob.glob(os.path.join(input_path, '*.png'))
i = 0
for img in images:

    # Get the input_tensor
    print(img)
    
    image = np.array(Image.open(img))
    input_tensor, image_float_np = prepare_input(image)

    # Run the model and display the detections
    #  boxes, classes, labels, indices = predict(input_tensor, model, device, 0.5)
    # image = draw_boxes(boxes, labels, classes, image)

    masks, boxes, labels = get_outputs(input_tensor, model, THRESHOLD)
    
    # print("\n masks", masks)
    # print("boxes", boxes)
    # result = draw_boxes(labels, coco_names, boxes, labels)
    result_img, result_masks = draw_segmentation_map(image, masks, boxes, labels)

    # visualize the image
    # cv2.imshow('Segmented image', result)
    save_path = "../../reports/figures/result_{no:}.png"

    plt.figure(figsize=(10,10))
    plt.title("Model Prediction")
   
    img_array = [result_img, result_masks]

    for j in range(2):
        plt.subplot(1, 2, j+1)
        plt.imshow(img_array[j])

    plt.savefig(save_path.format(no=i))
    
    # plt.show()

    # cv2.imwrite(save_path.format(no=i), cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    
    
    boxes, classes, labels, indices = predict(input_tensor, model, device, 0.7)
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
    image_with_bounding_boxes = draw_boxes(boxes, labels, classes, cam_image)

    plt.imshow(image_with_bounding_boxes)
    save_path_ablation = "../../reports/figures/ablation_cam_{no:}.png"
    plt.savefig(save_path_ablation.format(no=i))
    i = i + 1
    # # plt.show()

if __name__ == "__main__":

    input_path = '/home/vivek/Datasets/AmyB/amyb_wsi/test/images'
    explain = ExplainPredictions(test_input_path=input_path, detection_threshold=0.75,)


