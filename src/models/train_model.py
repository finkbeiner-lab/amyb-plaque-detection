import sys
sys.path.append('../')
import numpy as np
import torch
import wandb


import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from utils.engine import train_one_epoch, evaluate
from utils import utils
from features import build_features, transforms as T
import pdb
import matplotlib.pyplot as plt
from visualization.visualize import visualize_train


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
config = dict( # TODO: assert no params are overriden
    num_epochs=10,
    **optim_config,
    **lr_config,
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
    return T.Compose(transforms)


# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 1 + 3
# use our dataset and defined transformations



dataset = build_features.AmyBDataset('/home/vivek/Datasets/mask_rcnn/dataset/train', get_transform(train=True))
dataset_test = build_features.AmyBDataset('/home/vivek/Datasets/mask_rcnn/dataset/val', get_transform(train=False))

x = next(iter(dataset))

with wandb.init(project='amyb-plaque-detection', config=config):

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, **optim_config)
    # and a learning rate scheduler
    lr_scheduler = None
    # torch.optim.lr_scheduler.StepLR(optimizer, **lr_config)

    # Test the train data
    # img_no = 0
    # for images, targets in data_loader:
    #     visualize_train(images, targets, wandb, img_no, True)
    #     img_no = img_no + 1
    

   

    for epoch in range(config['num_epochs']):

        train_logs = train_one_epoch(model, optimizer, data_loader, device, epoch, wandb, print_freq=1)
        # update the learning rate
        # lr_scheduler.step()
        # evaluate on the test dataset
        # eval_logs, eval_res = evaluate(model, data_loader_test, device=device)

        # train_logs are instances of utils.engine.MetricLogger,
        #   which stores metrics in train_logs.meters, an instance of
        #   utils.utils.SmoothedValue. The object train_metrics.deque
        #   contains per-batch results of the training, while providing
        #   smoothed sliding-window views of the metrics across training runs
        train_metrics = train_logs.meters
        # eval_metrics = eval_logs.meters

        # eval_res is an isntance of utils.coco_eval.CocoEvaluator,
        #   which stores the the instance ids of images passed to it in
        #   eval_res.img_ids, and processes instances as one of several
        #   eval_res.iou_types. The dictionary eval_res.eval_imgs stores,
        #   with eval_res.iou_types as keys, the results of an evaluation run.
        # eval_imgs = eval_res.eval_imgs
        train_metrics_dict = {k: list(v.deque) for k, v in train_metrics.items() if k!='lr'}
        
        d = train_metrics_dict
        keys, vals = tuple(zip(*d.items()))
        unpacked = [dict(zip(keys, v)) for v in list(zip(*vals))] 

        

        wandb.log({'loss': train_metrics['loss'].global_avg, 
                   'loss_classifier': train_metrics['loss_classifier'].global_avg,
                   'loss_rpn_box_reg': train_metrics['loss_rpn_box_reg'].global_avg,
                   'loss_box_reg': train_metrics['loss_box_reg'].global_avg,
                   'loss_mask': train_metrics['loss_mask'].global_avg,
                   'loss_objectness': train_metrics['loss_objectness'].global_avg,
                    'epoch': epoch,
                    'data_point_details':unpacked})
            
        # wandb.log(train_metrics_dict)

    model_save_name = "../../models/mrcnn_model_{epoch:}.pth"
    torch.save(model, model_save_name.format(epoch=config['num_epochs']))

    print("The Model is Trained!")
    wandb.finish()