from collections import OrderedDict
import tqdm

import torch
from torch import nn, Tensor

import torchvision


def train(model, optimizer, device, loader, progress=False):
    model.train(True)

    losses = list()
    summary = OrderedDict(list())
    bar = tqdm.tqdm(total=len(loader)) if progress else None

    for step, (images, targets) in enumerate(loader):
        images = [image.to(device) for image in images]
        targets = [dict([(k, v.to(device)) for k, v in target.items()]) for target in targets]

        loss = model.forward(images, targets)
        sum(loss.values()).backward()
        optimizer.step()

        for k, v in loss.items():
            summary.setdefault(k, list()).append(v.item())

        disp = OrderedDict([(k[5:], f'{v.item():.4f}') for k, v in loss.items()])
        if progress:
            bar.set_postfix(disp)
            bar.update()
        else:
            print(f'Step: {step}')
            print('\n'.join([f'  {name}: {val}' for name, val in disp.items()]))
            print()

    summary = OrderedDict([(k[5:], f'{torch.tensor(v).mean().item():.4f}') for k, v in summary.items()])
    if progress:
        bar.set_postfix(summary)
        bar.close()
    else:
        print(f'Epoch:')
        print('\n'.join([f'  {name}: {val}' for name, val in summary.items()]))
        print()


def eval(model, device, image, thresh=None, mask_thresh=None):
    model.train(False)
    out = model.forward([image.to(device)], None)[0]
    if thresh is not None:
        idxs = out['scores'] >= thresh
        out = dict([(k, v[idxs]) for k, v in out.items()])
    if 'masks' in out.keys():
        out['masks'] = (out['masks'].squeeze(1) > (0.5 if mask_thresh is None else mask_thresh)).to(torch.bool)
    return out


def show(image, target):
    image = (image * 255).to(torch.uint8)
    labels = [f'{label.item()}: {target["scores"][i].item():.2f}' if 'scores' in target.keys() else f'{label.item()}' for i, label in enumerate(target['labels'])]
    image = torchvision.utils.draw_bounding_boxes(image, target['boxes'], labels)
    if 'masks' in target.keys():
        image = torchvision.utils.draw_segmentation_masks(image, target['masks'].to(torch.bool), alpha=0.5, colors=(['red'] * target['boxes'].size()[0]))
    return image

def show_stacked_pil(image, targets):
    images = list()
    for target in targets:
        images.append(show(image, target))
    return torchvision.transforms.ToPILImage()(torch.cat(images, dim=2))

# images.append(show(image, eval(model, device, image, 0.5)))
