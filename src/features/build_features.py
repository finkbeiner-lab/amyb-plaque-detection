from typing import Any, Callable, List, Mapping, Optional, Tuple

import os
from matplotlib import image
import numpy as np
from PIL import Image
import pdb

import torch
from torch import nn, Tensor


class AmyBDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "labels"))))

        assert set([len(set(['_'.join('.'.join(s.split('.')[:-1]).split('_')[:-1]) for s in item])) for item in zip(self.imgs, self.masks)]) == {1}
        # print("\nImages Order ", self.imgs)
        # print("\nLabels Order", self.masks)


    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "labels", self.masks[idx])

        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background

        # Palette "P" mode works by creating a mapping table, which corresponds to an 
        # index (between 0 and 255) to a discrete color in a larger 
        # color space (like RGB).
        mask = Image.open(mask_path).convert('P')

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]
        # masks.shape (1, 1024, 1024) first element denotes number of objects
        # (num_objects, height, width)



        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            
            if xmax <= xmin and ymax <=ymin:
                print("degenrate boxes", mask_path)
                print(len(obj_ids))
                break
            boxes.append([xmin, ymin, xmax, ymax])
        

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # labels = torch.tensor(labels, dtype=torch.int64)
        # labels = torch.ones((num_objs,), dtype=torch.int64)

        x = [id // 50 for id in obj_ids]
        labels = torch.tensor(x)
       


        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class DatasetRelabeled(torch.utils.data.Dataset):
    def __init__(self,
        dataset: torch.utils.data.Dataset,
        fn: Callable[[int,], int],
    ) -> None:
        self.dataset = dataset
        self.fn = fn

    def __len__(self) -> int:
        return self.dataset.__len__()

    def __getitem__(self,
        idx: int,
    ) -> Tuple[Tensor, Mapping[str, Tensor]]:
        image, target = self.dataset.__getitem__(idx)
        for i, v in enumerate(target['labels']):
            target['labels'][i] = self.fn(v)
        return image, target


