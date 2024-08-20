import os
import sys
sys.path.insert(0, '../')
#from models.model_mrcnn import _default_mrcnn_config, build_default
import torchvision
import torch
from models_pytorch_lightning.model_mrcnn_config import _default_mrcnn_config, build_default
from features import build_features
from data.test_data import tiling_nosaving
from models_pytorch_lightning.generalized_mask_rcnn_pl import LitMaskRCNN
#from utils.helper_functions import evaluate_metrics, get_outputs, compute_iou, evaluate_mask_rcnn
from features import build_features
from timeit import default_timer as timer 
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
#import multiprocessing
import numpy as np
import cv2
import glob
import pandas as pd
from skimage.measure import regionprops
from skimage.measure import label, regionprops, regionprops_table
from skimage import data, filters, measure, morphology
from skimage.color import rgb2hed, hed2rgb
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import TileArrayDataloader 
from multiprocessing import Pool, cpu_count

def create_dataloader(arrays, batch_size=1, shuffle=False, num_workers=0):
    dataset = TileArrayDataloader.NumpyArrayDataset(arrays)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader



if __name__ == "__main__":
    #try:
    #    torch.multiprocessing.set_start_method('spawn')
    #except RuntimeError:
    #    pass
    
    numpy_arrays = [(str(i), np.random.randn(1024, 1024,3).astype(np.uint8)) for i in range(100)]
    print(len(numpy_arrays))
    dataloader = create_dataloader(numpy_arrays, batch_size=8, shuffle=False, num_workers=4)
    c =0
    for batch in dataloader:
        c=c+1
        print("batch_running",c)
        