import os

import torch
import numpy as np
from PIL import Image
import pdb


class AmyBDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        pdb.set_trace()
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "labels"))))
        print("\nImages Order ", self.imgs)
        print("\nLabels Order", self.masks)
       

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "labels", self.masks[idx])
        
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path).convert('P')

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
       
        # labels = np.zeros(4)
        # print(obj_ids)

        # Mapping the Category to one hot vector [0, 1, 0, 0]
        # id = 150
        # for id in obj_ids:
        #     if id >=50 and id < 100:
        #         labels[0] = 1 #Core
        #     elif id >= 100 and id < 150:
        #         labels[1] = 1 # Diffused
        #     elif id >=150 and id < 200:
        #         labels[2] = 1 # Neuritic
                
       
        
        # labels = torch.tensor(labels, dtype=torch.int64)
        # labels = torch.ones((num_objs,), dtype=torch.int64)

        x = [id // 50 for id in obj_ids]
        labels = torch.tensor(x)

        # labels = torch.ones((obj_ids,), dtype=torch.int64)

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
        
        if self.transforms is not None:
            img = self.transforms(img)
            target = self.transforms(target)

        return img, target

    def __len__(self):
        return len(self.imgs)