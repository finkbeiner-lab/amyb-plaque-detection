import os
from matplotlib import image

import torch
import numpy as np
from PIL import Image
import pdb
import glob
import cv2


class AmyBDataset(object):
    """
    Custom dataset class for loading Amyloid Beta-stained images and corresponding masks.

    Args:
        root (str): Root directory containing 'images' and 'labels' subdirectories.
        transforms (callable): A function/transform to apply to the image and target.
    """
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "labels"))))
        # Remove unnecessary files
        if ".DS_Store" in self.imgs:
            self.imgs.remove(".DS_Store")
        if ".DS_Store" in self.masks:
            self.masks.remove(".DS_Store")
        assert set([len(set(['_'.join('.'.join(s.split('.')[:-1]).split('_')[:-1]) for s in item])) for item in zip(self.imgs, self.masks)]) == {1}


    def __getitem__(self, idx):
        """
        Load an image and its corresponding segmentation mask and return 
        the transformed image along with target dictionary for training.

        Args:
            idx (int): Index of the image/mask pair.

        Returns:
            Tuple: (image, target) where image is the PIL image after transformation,
            and target is a dictionary containing bounding boxes, labels, masks,
            image ID, area, and crowd information.
        """
        # Paths to image and mask
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
        mask = mask//50
        # instances are encoded as different colors
        num_labels, mask2 = cv2.connectedComponents(mask)
        masks = [mask2==i for i in range(1,num_labels)]
        # get bounding box coordinates for each mask
        num_objs = num_labels-1
        boxes = []
        areas =[]
        labels = []
        final_masks=[]
        for i in range(num_objs):
            pos = np.where(masks[i]==True)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            label = np.unique(mask[masks[i]])[0]
            if xmax <= xmin and ymax <=ymin:
                print("degenrate boxes", mask_path)
                print("pos",pos)
                print("xmax, xmin, ymax,ymin",xmax, xmin, ymax,ymin)
                
            if ((ymax-ymin)>0) and ((xmax-xmin)>0):
                boxes.append([xmin, ymin, xmax, ymax])
                areas.append([(ymax-ymin)*(xmax-xmin)])
                labels.append(label)
                final_masks.append(masks[i])

                
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        if len(boxes.shape)!=2:
            print(img_path)
            print(mask_path)
            print(labels, np.unique(mask))

        masks = np.array(final_masks, dtype=np.uint8)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = areas
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target

    def __len__(self):
        return len(self.imgs)