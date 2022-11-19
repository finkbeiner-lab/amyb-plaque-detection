import os
import sys
__pkg = os.path.abspath(os.path.join(__file__, *('..'.split() * 2)))
if __pkg not in sys.path:
    sys.path.append(__pkg)

import torch
from torch import nn, Tensor

import torchvision
from torchvision.transforms import ToTensor, ToPILImage

import models
from models.rcnn_conf import rcnn_v2_conf
from models.model_utils import train, eval, show

import data
from data.json_datasets import VipsDataset


class CombinedVipsDataset(VipsDataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.idxs = list()
        for i, dataset in enumerate(self.datasets):
            self.idxs += [(i, j) for j in range(len(dataset))]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        i, j = self.idxs[idx]
        return self.datasets[i][j]



def get_dataset(slide_name, label_names):
    vips_img_dir = '/gladstone/finkbeiner/steve/work/data/npsad_data/gennadi/tau_ad_mrxs/'
    json_dir = '/home/gryan/projects/qupath/annotations/tau/'

    vips_img_fname = os.path.join(vips_img_dir, f'{slide_name}.mrxs')
    json_fname = os.path.join(json_dir, f'{slide_name}.json')

    return VipsDataset(vips_img_fname, json_fname, label_names=label_names,)


if __name__ == '__main__':
    # vips_img_dir = '/gladstone/finkbeiner/steve/work/data/npsad_data/gennadi/tau_ad_mrxs/'
    # vips_img_name = 'XE16-014_1_Tau_1'
    # vips_img_fname = os.path.join(vips_img_dir, f'{vips_img_name}.mrxs')
    #
    # json_dir = '/home/gryan/projects/qupath/annotations/tau/'
    # json_fname = os.path.join(json_dir, f'{vips_img_name}.json')

    slide_names = 'XE16-014_1_Tau_1 XE16-027_1_Tau_1'.split()
    label_names = 'Pre Mature Ghost'.split()
    label_colors = 'red green blue'.split()

    datasets = [get_dataset(slide_name, label_names) for slide_name in slide_names]
    datasets = [(torch.utils.data.Subset(dataset, list(range(8, len(dataset)))), torch.utils.data.Subset(dataset, list(range(0, 8)))) for dataset in datasets]
    dataset = CombinedVipsDataset([_[0] for _ in datasets])
    dataset_test = CombinedVipsDataset([_[1] for _ in datasets])
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda _: tuple(zip(*_)))

    device = torch.device('cuda', 0)
    epochs = 40
    freq = 5

    model_conf = rcnn_v2_conf(pretrained=True, num_classes=4)
    model = model_conf.module(
        freeze_submodules=['backbone.body.conv1', 'backbone.body.bn1', 'backbone.body.layer1'],
        skip_submodules=['roi_heads.box_predictor', 'roi_heads.mask_predictor.mask_fcn_logits']
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), **dict(lr=2e-4, momentum=9e-2, weight_decay=1e-5,))


    for epoch in range(1, epochs + 1):
        train(model, optimizer, device, loader, epoch=epoch, progress=False,)

        grid = torchvision.utils.make_grid([show(image, eval(model, device, image, thresh=0.5, mask_thresh=0.5,), label_names=label_names, label_colors=label_colors,) for image, _ in dataset_test], nrow=4)
        if epoch % freq == 0 or epoch == epochs:
            ToPILImage()(grid).save(f'/home/gryan/projects/amyb-plaque-detection/reports/eval/{epoch}.png')
