"""Load data"""

import torch
from glob import glob
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import src.features.transforms as T
import pdb
from skimage import io, transform
from utils.utils import collate_fn


def random_crop(img, mask, output_size):
    #### Doesn't work, cuts off masks, use src > features > transforms.py
    h, w = img.shape[:2]
    new_h, new_w = output_size
    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)

    img = img[top: top + new_h,
          left: left + new_w]
    mask = mask[top: top + new_h,
           left: left + new_w]
    return img, mask


def rescale(img, mask, output_size):
    # h, w = img.shape[:2]
    new_h, new_w = output_size
    new_h, new_w = int(new_h), int(new_w)
    lbl = np.unique(mask)[-1]  # assume just one label in image
    img = transform.resize(img, (new_h, new_w), preserve_range=True)
    mask = transform.resize(mask, (new_h, new_w), preserve_range=True)
    mask[mask > 0] = lbl
    return np.uint16(img), np.uint8(mask)


class RoboDataset(object):
    def __init__(self, root, transforms, istraining, debug=False):
        self.root = root
        self.istraining = istraining
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        if 'vivek' in self.root:
            self.imgs = list(sorted(glob(os.path.join(root, "images", '*.png'))))

            self.masks = list(sorted(glob(os.path.join(root, "labels", '*.png'))))
        else:
            self.imgs = list(sorted(glob(os.path.join(root, "images", '*Confocal-GFP16*.tif'))))

            self.masks = list(sorted(glob(os.path.join(root, "labels", '*Confocal-GFP16*.tif'))))
        if debug:
            self.imgs = self.imgs[1000:]
            self.masks = self.masks[1000:]
        assert len(self.imgs) == len(self.masks)

        # print(self.imgs)
        self.length = len(self.imgs)
        print('length', self.length)

    def __getitem__(self, idx):
        # idx = 7195
        # 6624
        # 5495
        # 6509
        # 2817
        # 274
        # 5532
        # 7195
        def getmasks(img_path, mask_path):
            # assert img_path.split('/')[-1] in mask_path
            img = Image.open(img_path).convert('RGB')
            # img = Image.open(img_path)
            mask = Image.open(mask_path).convert('P')
            img = np.array(img)
            mask = np.array(mask)
            obj_ids = np.unique(mask)
            obj_ids = obj_ids[1:]
            return img, mask, obj_ids

        def transforms(img, mask, train):
            if train:
                r = np.random.randint(0, 3)
                flip = np.random.random() > 0.5
                up = np.random.random() > 0.5
                img = np.rot90(img, r)
                mask = np.rot90(mask, r)
                if flip:
                    img = np.fliplr(img)
                    mask = np.fliplr(mask)
                if up:
                    img = np.flipud(img)
                    mask = np.flipud(mask)
            return img, mask

        def getboxes_v2(mask, obj_ids):
            masks = mask == obj_ids[:, None, None]
            # print('masks sh', np.shape(masks))

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
            # print('boxes v2', boxes)
            # print('obj_ids v2', obj_ids)
            return boxes, masks

        def getboxes(mask, obj_ids, debug=False):
            """boxes ([N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.

                labels [N]): the class label for each ground-truth box

                masks ([N, H, W]): the segmentation binary masks for each instance"""
            if debug:
                print('max mask', np.max(mask))
                plt.imshow(mask * 255)
                plt.title('mask debug')
                plt.show()
            boxes = []
            masks = np.zeros((1, np.shape(mask)[0], np.shape(mask)[1]), dtype=bool)
            labels = obj_ids
            if len(obj_ids) > 1:
                assert 0, 'not handling multiple object ids per image'
            if np.any(mask > 0):
                # cnts, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                numLabels, encoded = cv2.connectedComponents(mask, connectivity=8, ltype=cv2.CV_16U)
                if debug:
                    plt.imshow(encoded)
                    plt.title('encoded cv2')
                    plt.show()
                if numLabels > 5:
                    numLabels = 5
                masks = np.zeros((numLabels - 1, np.shape(mask)[0], np.shape(mask)[1]), dtype=bool)

                for i in range(1, numLabels):  # skip 0
                    msk = np.uint8((encoded == i) * 255)
                    cnts, hierarchy = cv2.findContours(msk, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    cnt = max(cnts, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(cnt)
                    if w * h > 0:
                        if x + w > np.shape(msk)[0]:
                            x2 = np.shape(msk)[0]
                        else:
                            x2 = x + w
                        if y + h > np.shape(msk)[1]:
                            y2 = np.shape(msk)[0]
                        else:
                            y2 = y + h
                        boxes.append([x, y, x2, y2])
                        masks[i - 1] = msk > 0
                    # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
                    # cnts = cnts[0]
                    # for idx, cnt in enumerate(cnts): # there is one contour
                    #     x, y, w, h = cv2.boundingRect(cnt)
                    #     boxes.append([x, y, x + w, y + h])
                    #     if debug:
                    #         cv2.rectangle(img, (x, y), (x + w, y + h), (36, 255, 12), 2)
                    # if debug:
                    #     plt.imshow(img)
                    #     plt.show()
                labels = [obj_ids[0]] * len(masks)
                assert len(labels) == len(masks), f'{len(labels)}, {len(masks)}'
                assert len(labels) == len(boxes)
                # print('labels', labels)
            else:
                assert 0, 'no mask values'

            return masks, boxes, labels

        def combine_imgs(img, img2, rand):
            """Add part of img2 to img1"""
            img = np.float32(img)
            img2 = np.float32(img2)
            res = img + img2 * rand
            return res / np.max(res)

        def remove_degenerate_boxes(boxes):
            child = []
            for i, box in enumerate(boxes):
                for j, b in enumerate(boxes):
                    if i != j:
                        if box[0] >= b[0] and box[2] <= b[2] and box[1] >= b[1] and box[3] <= b[3]:
                            # box is child of b
                            child.append(i)
            remove = np.unique(child)
            return [i for i in range(len(boxes)) if i not in remove]

        # load images and masks

        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "labels", self.masks[idx])
        img, mask, obj_ids = getmasks(img_path, mask_path)
        # the dataset is balanced, get random index and use it for augmentation
        random_idx = np.random.randint(0, len(self.imgs))
        img_path_aug = os.path.join(self.root, "images", self.imgs[random_idx])
        mask_path_aug = os.path.join(self.root, "labels", self.masks[random_idx])
        img2, mask2, obj_ids2 = getmasks(img_path_aug, mask_path_aug)

        # while len(obj_ids) == 0:
        #     idx -= 1
        #     img_path = os.path.join(self.root, "images", self.imgs[idx])
        #     mask_path = os.path.join(self.root, "labels", self.masks[idx])
        #     img, mask, obj_ids = getmasks(img_path, mask_path)

        # print(idx)
        # augmentation
        img, mask = transforms(img, mask, self.istraining)
        img2, mask2 = transforms(img2, mask2, self.istraining)
        masks, boxes, obj_ids = getboxes(mask, obj_ids)
        masks2, boxes2, obj_ids2 = getboxes(mask2, obj_ids2)
        # boxes, masks = getboxes_v2(mask, obj_ids)
        # _boxes2, _masks2 = getboxes_v2(mask2, obj_ids2)
        # got boxes

        rand = np.random.random()

        if rand > .25:
            merged_masks = np.vstack((masks, masks2))
            img = combine_imgs(img, img2, rand)
            merged_boxes = boxes + boxes2
            merged_obj_ids = obj_ids + obj_ids2
        else:
            img = np.float32(img) / np.max(img)
            merged_masks = masks
            merged_boxes = boxes
            merged_obj_ids = obj_ids
        merged_masks = np.array(merged_masks, dtype=np.uint8)
        merged_obj_ids = np.array(merged_obj_ids)
        merged_boxes = np.array(merged_boxes)
        # print('lengths', len(merged_obj_ids), len(merged_masks), len(merged_boxes))
        if 'vivek' in self.root:
            merged_obj_ids = [x // 50 for x in merged_obj_ids]
        else:
            if len(merged_boxes) > 1:
                keep = remove_degenerate_boxes(merged_boxes)
                merged_boxes = merged_boxes[keep]
                merged_obj_ids = merged_obj_ids[keep]
                merged_masks = merged_masks[keep]
        if np.shape(img)[-1] == 3:
            img = np.moveaxis(img, -1, 0)

        boxes = torch.as_tensor(merged_boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(merged_obj_ids, dtype=torch.int64)  # todo: variable
        masks = torch.as_tensor(np.array(merged_masks, dtype=np.uint8), dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) else torch.tensor([0, 0, 1, 1])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(merged_obj_ids),), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "masks": masks, "image_id": image_id, "area": area,
                  "iscrowd": iscrowd}
        if len(np.shape(img)) != 3:
            img = np.array([img, img, img])
        # else:
        # if 'vivek' in self.root:
        #     img = np.moveaxis(img, -1, 0)  # channels first
        img = torch.from_numpy(np.ascontiguousarray(img))
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_transform(train):
    if train:
        transforms = [
            # T.ToTensor(),
                      T.RandomIoUCrop(),
                      T.RandomPhotometricDistort(contrast=(0.5, 1.5), saturation=(1, 1), hue=(0, 0), brightness=(.8, 1.2),
                                                 p=0.5)
                       ]
    else:
        transforms = [T.ToTensor()]
    return T.Compose(transforms)


# class AmyBDataset(object):
#     def __init__(self, root, transforms):
#         self.root = root
#         self.transforms = transforms
#         # load all image files, sorting them to
#         # ensure that they are aligned
#         self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
#         self.masks = list(sorted(os.listdir(os.path.join(root, "labels"))))
#         assert len(self.imgs) == len(self.masks)
#         print(self.imgs)
#
#     def __getitem__(self, idx):
#         # load images and masks
#         img_path = os.path.join(self.root, "images", self.imgs[idx])
#         mask_path = os.path.join(self.root, "labels", self.masks[idx])
#
#         img = Image.open(img_path).convert("RGB")
#         # note that we haven't converted the mask to RGB,
#         # because each color corresponds to a different instance
#         # with 0 being background
#         mask = Image.open(mask_path).convert('P')
#
#         mask = np.array(mask)
#         # instances are encoded as different colors
#         obj_ids = np.unique(mask)
#         # first id is the background, so remove it
#         obj_ids = obj_ids[1:]
#
#         # split the color-encoded mask into a set
#         # of binary masks
#         masks = mask == obj_ids[:, None, None]
#
#         # get bounding box coordinates for each mask
#         num_objs = len(obj_ids)
#         boxes = []
#         for i in range(num_objs):
#             pos = np.where(masks[i])
#             xmin = np.min(pos[1])
#             xmax = np.max(pos[1])
#             ymin = np.min(pos[0])
#             ymax = np.max(pos[0])
#             boxes.append([xmin, ymin, xmax, ymax])
#
#         boxes = torch.as_tensor(boxes, dtype=torch.float32)
#         # there is only one class
#         labels = torch.ones((num_objs,), dtype=torch.int64)
#         masks = torch.as_tensor(masks, dtype=torch.uint8)
#
#         image_id = torch.tensor([idx])
#         area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
#         # suppose all instances are not crowd
#         iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
#
#         target = {"boxes": boxes, "labels": labels, "masks": masks, "image_id": image_id, "area": area,
#                   "iscrowd": iscrowd}
#
#         if self.transforms is not None:
#             img, target = self.transforms(img, target)
#
#         return img, target
#
#     def __len__(self):
#         return len(self.imgs)


if __name__ == '__main__':
    import utils.utils as utils

    # view data
    dataset = RoboDataset('/mnt/linsley/Shijie_ML/Ms_Tau/dataset/train_vivek', get_transform(train=True), istraining=True, debug=False)
    # dataset = HistoDataset('/mnt/linsley/Shijie_ML/Ms_Tau/dataset/train', get_transform(), istraining=True)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)
    for i, (images, targets) in enumerate(data_loader):
        if i > 10:
            break
        # images, targets = next(iter(data_loader))
        images = list(image.detach().cpu().numpy() for image in images)
        targets = [{k: v for k, v in t.items()} for t in targets]
        boxes = (targets[0]['boxes'].detach().cpu().numpy())
        classes = (targets[0]['labels'].detach().cpu().numpy())
        # img = np.uint8(np.moveaxis(images[0], 0, -1))
        # img = np.uint8(images[0])
        if np.shape(images[0])[0] == 3:
            img = np.moveaxis(images[0] * 255, 0, -1)

        else:
            img = images[0] * 255
        img = np.ascontiguousarray(np.uint8(img))

        for box, cls in zip(boxes, classes):
            x = int(box[0])
            y = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            if cls == 1:
                channel = 0
            elif cls == 2:
                channel = 2
            else:
                channel = 1
            cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 2)
            # cv2.rectangle(img[...,channel], (x, y), (x2, y2), (255, 255, 12), 2)
        print(boxes)
        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        # print(targets[0]['boxes'])
        # print(targets[0]['labels'])
        for mask in targets[0]['masks']:
            msk = mask.detach().cpu().numpy()
            print(np.max(msk))
            plt.figure()
            plt.imshow(msk)
        plt.figure()
        plt.imshow(img)
        plt.show()
    print('done')

'''361
[[ 93.   0. 300. 201.]
 [227.  35. 240.  49.]
 [239.  77. 300. 126.]
 [140. 113. 157. 127.]
 [ 83. 115.  99. 132.]
 [198. 122. 236. 139.]] overlap'''
