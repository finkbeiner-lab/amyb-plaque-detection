"""Predict on data without training. Sanity check."""

import sys

sys.path.append('../')
sys.path.append('../../')
import numpy as np
import torch
# import wandb


import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from utils.engine import train_one_epoch, evaluate
from utils import utils
from features import build_features, transforms as T
import pdb
import matplotlib.pyplot as plt
from visualization.visualize import visualize_train
from visualization.explain import ExplainPredictions
from data.pipeline import RoboDataset
import cv2

# import albumentations as A


######## WANDB integration begins (mainly) here

optim_config = dict(
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0005,
)
lr_config = dict(
    step_size=3,
    gamma=0.1
)
config = dict(  # TODO: assert no params are overriden
    num_epochs=1,
    batch_size=1,
    **optim_config,
    **lr_config,
    detection_threshold=0.75,
)


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 512
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        # transforms.append(T.RandomPhotometricDistort())
        # transforms.append(T.RandomZoomOut())
        # transforms.append(T.RandomIoUCrop())
        # transforms = A.Compose([A.RandomCrop(width=256, height=256),
        #                        A.HorizontalFlip(p=0.5),
        #                        A.RandomBrightnessContrast(p=0.2)])

        # transforms = torchvision.transforms.Compose([
        # torchvision.transforms.RandomHorizontalFlip(),
        # torchvision.transforms.RandomVerticalFlip(),
        # torchvision.transforms.GaussianBlur([3,3], sigma=(0.1, 2.0))
        # ])
    return T.Compose(transforms)
    # return transforms


def predict(input_tensor, model, device, detection_threshold):
    outputs = model(input_tensor)
    # i- 1 zero indexing - the model outputs a non zero indexing format ( 1, 2, 3)
    pred_classes = [i for i in outputs[0]['labels'].cpu().numpy()]
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


if __name__ == "__main__":

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 1 + 2

    # use our dataset and defined transformations
    dataset = RoboDataset('/mnt/linsley/Shijie_ML/Ms_Tau/dataset/train', get_transform(train=True), istraining=True)
    dataset_test = RoboDataset('/mnt/linsley/Shijie_ML/Ms_Tau/dataset/val', get_transform(train=False),
                               istraining=False)

    # with wandb.init(project="nps-ad", entity="hellovivek", config=config):

    # Get the Id
    # print("\n =======Wandb Run Id========", wandb.util.generate_id())

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.eval()

    imgs, dct = next(iter(data_loader))
    boxes, classes, labels, indices = res = predict(imgs, model, device='cuda',
                                                    detection_threshold=config['detection_threshold'])
    img = imgs[0].cpu().detach().numpy()
    img = np.moveaxis(img, 0, -1)
    img = np.uint8(img)

    for box, cls in zip(boxes, classes):
        x = box[0]
        y = box[1]
        x2 = box[2]
        y2 = box[3]
        if cls == 1:
            channel = 0
        elif cls == 2:
            channel = 2
        else:
            channel = 1
        cv2.rectangle(img[..., channel], (x, y), (x2, y2), (255, 255, 255), 2)
    plt.imshow(img)
    plt.show()
