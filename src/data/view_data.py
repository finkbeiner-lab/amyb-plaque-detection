"""View data from dataloader"""
import sys
sys.path.append('../')

import torch
import numpy as np
from pipeline import RoboDataset
from models.train_model import get_transform
from utils import utils
import matplotlib.pyplot as plt

def view_data():
    dataset = RoboDataset('/mnt/linsley/Shijie_ML/Ms_Tau/dataset/train', get_transform(train=True), istraining=True)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=12, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)
    for _ in range(10):
        imgs, dct = next(iter(data_loader))
        img = imgs[10].cpu().detach().numpy()
        img = np.moveaxis(img, 0,-1)
        masks = dct[10]['masks']
        mask = masks.cpu().detach().numpy()

        print(np.shape(img))
        print(np.shape(mask))
        plt.figure()
        plt.imshow(img)
        for mask in masks:
            plt.figure()
            plt.imshow(mask)
        plt.show()

if __name__=='__main__':
    view_data()