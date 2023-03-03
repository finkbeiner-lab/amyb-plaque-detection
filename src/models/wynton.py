from typing import Any, Callable, List, Mapping, Optional, Tuple

import os
import sys
__pkg = os.path.abspath(os.path.join(__file__, *('..'.split() * 2)))
if __pkg not in sys.path:
    sys.path.append(__pkg)

import torch
from torch import nn, Tensor

import torchvision

import data.datasets as datasets
import models.rcnn_conf as rcnn_conf
from models.modules.rcnn import RCNN, RCNNTransform
from models.model_utils import train, eval, show



if __name__ == '__main__':
    label_names = 'Core Diffuse Neuritic CAA'.split()
    label_colors = 'red blue green black'.split()

    slide_dir = '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/amyb-def-mfg-test/'
    slide_name_fn = lambda name: os.path.join(slide_dir, f'XE{name}_1_AmyB_1.mrxs')
    json_dir = '/gladstone/finkbeiner/steve/work/data/npsad_data/gennadi/jsons/amyb/'
    json_name_fn = lambda name: os.path.join(json_dir, f'{name}.json')

    tile_size = 1024
    slides_train, slides_test = '09-028 10-033'.split(), '09-063'.split()
    dsets_train, dsets_test = [[VipsJsonDataset(slide_name_fn(slide), json_name_fn(slide), label_names, step=tuple([tile_size // (2 if training else 1)] * 2), size=tuple([tile_size] * 2)) for slide in slide] for slides, training in ((slides_train, True), (slides_test, False))]
    # dset_train, dset_test = [torch.utils.data.ConcatDataset(dsets) for dsets in (dsets_train, dsets_test)]
