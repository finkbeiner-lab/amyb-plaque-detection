import torch

import torchvision
from torchvision.transforms import ToTensor, ToPILImage, Compose, RandomHorizontalFlip, RandomVerticalFlip


class _ToTensor(ToTensor):
    def __call__(self, image, target=None):
        return super().__call__(image), target

class _Compose(Compose):
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class _RandomHorizontalFlip(RandomHorizontalFlip):
    def forward(self, image, target=None):
        if torch.rand(1) < self.p:
            image = torchvision.transforms.functional.hflip(image)
            if target is not None:
                if 'boxes' in target.keys():
                    target['boxes'][:, [0, 2]] = torchvision.transforms.functional.get_dimensions(image)[-1] - target['boxes'][:, [2, 0]]
                if 'masks' in target.keys():
                    target['masks'] = target['masks'].flip(-1)
        return image, target

class _RandomVerticalFlip(RandomVerticalFlip):
    def forward(self, image, target=None):
        if torch.rand(1) < self.p:
            image = torchvision.transforms.functional.vflip(image)
            if target is not None:
                if 'boxes' in target.keys():
                    target['boxes'][:, [1, 3]] = torchvision.transforms.functional.get_dimensions(image)[-2] - target['boxes'][:, [3, 1]]
                if 'masks' in target.keys():
                    target['masks'] = target['masks'].flip(-2)
        return image, target
