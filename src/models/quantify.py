import math
import os
import sys
__pkg = os.path.abspath(os.path.join(__file__, *('..'.split() * 2)))
if __pkg not in sys.path:
    sys.path.append(__pkg)

import torch
from torch import nn, Tensor

import torchvision
from torchvision.transforms import ToTensor, ToPILImage

from models.rcnn_conf import rcnn_conf
from models.model_utils import train, eval, show

from data.json_datasets import VipsEvalDataset



def show_and_eval(model, device, images, thresh=None, label_names=None, label_colors=None):
    results = [dict([(k, v.to(torch.device('cpu'))) for k, v in eval(model, device, image, thresh=thresh, mask_thresh=0.5).items()]) for image in images]
    viz = [show(image, result, label_names=label_names, label_colors=label_colors, pil=False) for image, result in zip(images, results)]

    return ToPILImage()(torchvision.utils.make_grid(viz, nrow=int(math.sqrt(len(images))))), results


if __name__ == '__main__':
    vips_img_dir = '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/amy-def-mfg/'
    label_names = 'Core Diffuse Neuritic CAA'.split()
    label_colors = 'black red green blue'.split()

    vips_img_name = 'XE19-010_1_AmyB_1'
    vips_img_fname = os.path.join(vips_img_dir, f'{vips_img_name}.mrxs')
    weights_fname = '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/models/' + 'vibrant-yogurt-428_mrcnn_model_50.pth'

    dataset = VipsEvalDataset(vips_img_fname)
    # coords = (0, 105000, 20000, 10000)
    # coords = tuple([coord // 1024 for coord in coords])

    centerX, centerY = tuple([((coord - offset) // size) // 2 for coord, size, offset in zip((dataset.vips_img.width, dataset.vips_img.height), dataset.size, dataset.offset)])
    dataset.set_tiles([(x, y) for x in range(centerX + 5, centerX + 10) for y in range(centerY - 5, centerY + 5)])

    device = torch.device('cuda', 0)

    weights = torch.load(weights_fname)
    model = rcnn_conf(pretrained=False, num_classes=5).module()
    model.load_state_dict(weights)
    model.to(device)

    sne = lambda imgs: show_and_eval(model, device, imgs, label_names=label_names, label_colors=label_colors)
