import pdb
import torchvision
import torch
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


def predict(input_tensor, model, device, detection_threshold):
    outputs = model(input_tensor)
    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
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



coco_names = ['Core', 'Diffused', 'Neuritic', "Unknown"]
input_path = '/home/vivek/Datasets/mask_rcnn/dataset/train/images/6.jpg'


# This will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

model = torch.load('../../models/mrcnn_model.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.eval().to(device)

# Get the input_tensor
image = np.array(Image.open(input_path))
input_tensor, image_float_np = prepare_input(image)

# Run the model and display the detections
boxes, classes, labels, indices = predict(input_tensor, model, device, 0.9)
image = draw_boxes(boxes, labels, classes, image)

target_layers = [model.backbone]
targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=boxes)]

# cam = EigenCAM(model,
#                target_layers, 
#                use_cuda=torch.cuda.is_available(),
#                reshape_transform=fasterrcnn_reshape_transform)

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
plt.savefig('../../reports/figures/ablation_cam.png')
plt.show()