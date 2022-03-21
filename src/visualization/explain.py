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

THRESHOLD = 0.7

def get_outputs(input_tensor, model, threshold):
    with torch.no_grad():
        # forward pass of the image through the modle
        outputs = model(input_tensor)
    
    # get all the scores
    scores = list(outputs[0]['scores'].detach().cpu().numpy())
    print("\n scores", scores)
    # index of those scores which are above a certain threshold
    thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
    thresholded_preds_count = len(thresholded_preds_inidices)
    # get the masks
    masks = (outputs[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    print("masks", masks)
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

def draw_segmentation_map(image, masks, boxes, labels):
    alpha = 1 
    beta = 0.6 # transparency for the segmentation map
    gamma = 0 # scalar added to each sum
    for i in range(len(masks)):
        red_map = np.zeros_like(masks[i]).astype(np.uint8)
        green_map = np.zeros_like(masks[i]).astype(np.uint8)
        blue_map = np.zeros_like(masks[i]).astype(np.uint8)

        # apply a randon color mask to each object
        color = COLORS[random.randrange(0, len(COLORS))]
        red_map[masks[i] == 1], green_map[masks[i] == 1], blue_map[masks[i] == 1]  = color
        # combine all the masks into a single image
        segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
        #convert the original PIL image into NumPy format
        image = np.array(image)
        # convert from RGN to OpenCV BGR format
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # apply mask on the image
        cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, image)
        # draw the bounding boxes around the objects
        cv2.rectangle(image, boxes[i][0], boxes[i][1], color=color, 
                      thickness=2)
        # put the label text above the objects
        cv2.putText(image , labels[i], (boxes[i][0][0], boxes[i][0][1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 
                    thickness=2, lineType=cv2.LINE_AA)
    
    return image


def predict(input_tensor, model, device, detection_threshold):
    outputs = model(input_tensor)
    # i- 1 zero indexing - the model outputs a non zero indexing format ( 1, 2, 3)
    pred_classes = [coco_names[i-1] for i in outputs[0]['labels'].cpu().numpy()]
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

def draw_boxes(boxes, labels, classes, image):
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
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


def prepare_input(image):
   
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



coco_names = ['Background', 'Core', 'Diffused', 'Neuritic']
# coco_names = ['Core', 'Neuritic', "Unknown"]
# coco_names = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', \
#               'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 
#               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 
#               'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',
#               'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
#               'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
#               'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
#               'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
#               'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
#               'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet',
#               'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
#               'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase',
#               'scissors', 'teddy bear', 'hair drier', 'toothbrush']




input_path = '/home/vivek/Datasets/mask_rcnn/dataset/train/images'

# This will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

# model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model = torch.load('../../models/mrcnn_model_10.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.eval().to(device)


images = glob.glob(os.path.join(input_path, '*.jpg'))
i = 0
for img in images:

    # Get the input_tensor
    print(img)
    
    image = np.array(Image.open(img))
    input_tensor, image_float_np = prepare_input(image)

    # Run the model and display the detections
    # boxes, classes, labels, indices = predict(input_tensor, model, device, 0.5)
    # image = draw_boxes(boxes, labels, classes, image)

    masks, boxes, labels = get_outputs(input_tensor, model, THRESHOLD)
    
    print("\n masks", masks)
    print("boxes", boxes)
    result = draw_segmentation_map(image, masks, boxes, labels)
    # visualize the image
    # cv2.imshow('Segmented image', result)
    save_path = "../../reports/figures/result_{no:}.jpg"
    cv2.imwrite(save_path.format(no=i), cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    i = i + 1
    

# # plt.imshow(image)
# # plt.show()

# pdb.set_trace()

# target_layers = [model.backbone]
# targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=boxes)]

# cam = AblationCAM(model,
#                target_layers, 
#                use_cuda=torch.cuda.is_available(), 
#                reshape_transform=fasterrcnn_reshape_transform,
#                ablation_layer=AblationLayerFasterRCNN())

# grayscale_cam = cam(input_tensor, targets=targets)
# # Take the first image in the batch:
# grayscale_cam = grayscale_cam[0, :]
# cam_image = show_cam_on_image(image_float_np, grayscale_cam, use_rgb=True)
# # And lets draw the boxes again:
# image_with_bounding_boxes = draw_boxes(boxes, labels, classes, cam_image)

# plt.imshow(image_with_bounding_boxes)
# plt.savefig('../../reports/figures/ablation_cam.png')
# plt.show()