import torchvision
import torch
from models_pytorch_lightning.generalized_mask_rcnn_pl import LitMaskRCNN
import torchvision.ops.boxes as bops
import torchvision
from torchvision import transforms
import torch
from PIL import Image
import cv2
import numpy as np
import os
import pandas as pd

class_names = ['Cored', 'Diffuse', 'Coarse-Grained', 'CAA']
class_to_colors = {'Cored': (255, 0, 0), 'Diffuse' : (0, 0, 255), 'Coarse-Grained': (0,255,0), 'CAA':(225, 255, 0)}
colors = np.random.uniform(0, 255, size=(len(class_names), 3))  


def get_outputs_nms(input_tensor, model, score_threshold = 0.5, iou_threshold = 0.5):
    #mask_list = []
    #label_list = []
    #score_list =[]
    #box_list = []
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
    boxes = [[int(i[0]), int(i[1]), int(i[2]), int(i[3])]  for i in boxes.detach().cpu()]
    labels = list(labels.detach().cpu().numpy())
    labels = [class_names[i-1] for i in labels]
    return masks, boxes, labels, scores



def prepare_input(image):
    image_float_np = np.float32(image) / 255
    # define the torchvision image transforms
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(1024),
        torchvision.transforms.ToTensor(),
    ])
    input_tensor = transform(transforms.ToPILImage()(image))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_tensor = input_tensor.to(device)
    # Add a batch dimension:
    input_tensor = input_tensor.unsqueeze(0)
    return input_tensor, image_float_np

def draw_boxes(boxes, labels, classes, image):
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
    
def change_to_numpy_array(input_tensor):
    input_tensor_cpu = input_tensor.squeeze(0).cpu()  # Remove batch dimension and move to CPU if necessary
    # Convert to NumPy array
    numpy_array = input_tensor_cpu.permute(1, 2, 0).numpy()  # Convert to HWC format for visualization
    # Scale back to 0-255 range if needed
    numpy_array = (numpy_array * 255).astype(np.uint8)
    return numpy_array
  
    
    
def draw_segmentation_map(image, masks, boxes, labels,scores):
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
        color = colors[random.randrange(0, len(colors))]
        red_map[masks[i] == 1], green_map[masks[i] == 1], blue_map[masks[i] == 1]  = class_to_colors[labels[i]]
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
        cv2.putText(image , labels[i]+str(np.round(scores[i],2)), (x1, y1-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 
                    thickness=2, lineType=cv2.LINE_AA)
        
        # Convert Back
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, result_masks



def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = np.max(b1_x1, b2_x1)
    inter_rect_y1 = np.max(b1_y1, b2_y1)
    inter_rect_x2 = np.min(b1_x2, b2_x2)
    inter_rect_y2 = np.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = np.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * np.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou


"""
def prepare_input(image):
    #image = Image.open(image_path)
    image_float_np = np.float32(image) / 255
    # define the torchvision image transforms
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(1024, interpolation=Image.LANCZOS),
        torchvision.transforms.ToTensor(),
    ])
    #input_tensor = transform(transforms.ToPILImage()(image))
    input_tensor = transform(image)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_tensor = input_tensor.to(device)
    # Add a batch dimension:
    input_tensor = input_tensor.unsqueeze(0)
    return input_tensor, image_float_np



def get_outputs_nms(input_tensor, model, score_threshold = 0.5, iou_threshold = 0.5):
    with torch.no_grad():
        # forward pass of the image through the modle
        outputs = model(input_tensor)
    boxes = outputs[0]['boxes']
    labels = outputs[0]['labels']
    scores = outputs[0]['scores']
    masks = outputs[0]['masks']
    # Apply score threshold
    keep = scores > score_threshold
    boxes, labels, scores, masks = boxes[keep], labels[keep], scores[keep], masks[keep]
    keep = torchvision.ops.nms(boxes, scores, iou_threshold)
    boxes, labels, scores, masks = boxes[keep], labels[keep], scores[keep], masks[keep]
    scores = list(scores.detach().cpu().numpy())
    masks = list((masks>0.5).detach().cpu().numpy())
    boxes = [[int(i[0]), int(i[1]), int(i[2]), int(i[3])]  for i in boxes.detach().cpu()]
    labels = list(labels.detach().cpu().numpy())
    labels = [class_names[i-1] for i in labels]
    return masks, boxes, labels, scores
    
"""