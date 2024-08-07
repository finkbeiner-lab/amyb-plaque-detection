import os
import sys
sys.path.append(os.path.join(os.getcwd(), *tuple(['..'])))

from features import build_features
from torchvision import transforms
from features import transforms as T
import torch
from torch import nn, Tensor
import cv2

data_transforms = transforms.Compose([transforms.ToTensor()])
collate_fn = lambda _: tuple(zip(*_)) # one-liner, no need to import


dataset_train_location =  '/workspace/Projects/Amyb_plaque_detection/Datasets/train_2'
#dataset_train_location = '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/amyb_wsi/train_v2'
train_dataset = build_features.AmyBDataset(dataset_train_location, T.Compose([T.ToTensor()]))
train_data_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=16, shuffle=True, num_workers=4,pin_memory=True,
            collate_fn=collate_fn)


def mean_std(loader):
  itr  = next(iter(loader))
  images =torch.stack(itr[0])
  # shape of images = [b,c,w,h]
  mean, std = images.mean([0,2,3]), images.std([0,2,3])
  return mean, std

print(mean_std(train_data_loader))