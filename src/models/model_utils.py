from collections import OrderedDict
import tqdm

import torch
from torch import nn, Tensor

import torchvision


"""
TODO:
  - Add warmup to train function
"""


def train(model, optimizer, device, loader, epoch=None, progress=False):
    def clear(lines, prefix=None):
        if prefix is not None:
            print(prefix)
        print('\n'.join(lines))
        return f'\x1b[{len(lines)}A' + '\n'.join([' ' * len(line) for line in lines]) + f'\x1b[{len(lines)}A'


    model.train(True)

    summary = OrderedDict()
    bar = tqdm.tqdm(total=len(loader)) if progress else None
    clear_str = None

    for step, (images, targets) in enumerate(loader):
        images = [image.to(device) for image in images]
        targets = [dict([(k, v.to(device)) for k, v in target.items()]) for target in targets]

        loss = model.forward(images, targets)
        sum(loss.values()).backward()
        optimizer.step()

        for k, v in loss.items():
            summary.setdefault(k, list()).append(v.item())

        disp = OrderedDict([(k, f'{v.item():.4f}') for k, v in loss.items()])
        if progress:
            bar.set_postfix(disp)
            bar.update()
        else:
            lines = [f'Step: {step + 1}/{len(loader)}'] + [f'  {name}: {val}' for name, val in disp.items()]
            clear_str = clear(lines, clear_str)

    summary = OrderedDict([(k, f'{torch.tensor(v).mean().item():.4f}') for k, v in summary.items()])
    if progress:
        bar.set_postfix(summary)
        bar.close()
    else:
        lines = [f'Epoch{str() if epoch is None else (" " + str(epoch))}:'] + [f'  {name}: {val}' for name, val in summary.items()]
        clear(lines, clear_str)
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


def show(image, target, label_names=None, label_colors=None, masks=True, pil=False):
    image = (image * 255).to(torch.uint8)
    labels = [f'{label}: {target["scores"][i].item():.2f}' if 'scores' in target.keys() else f'{label}' for i, label in enumerate(target['labels'] if label_names is None else [label_names[label - 1] for label in target['labels']])]
    colors = None if label_colors is None else [label_colors[label - 1] for label in target['labels']]
    image = torchvision.utils.draw_bounding_boxes(image, target['boxes'], labels=labels, colors=colors)
    if 'masks' in target.keys() and masks:
        image = torchvision.utils.draw_segmentation_masks(image, target['masks'].to(torch.bool), alpha=0.5, colors=(['red'] * len(target['labels'])))
    if pil:
        return torchvision.transforms.ToPILImage()(image)
    return image
