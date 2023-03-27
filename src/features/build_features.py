import os
from glob import glob
from matplotlib import image

import torch
import numpy as np
import pandas as pd
from PIL import Image
import features.transforms as T
import pdb


class AmyBDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "labels"))))

        assert set([len(set(['_'.join('.'.join(s.split('.')[:-1]).split('_')[:-1]) for s in item])) for item in
                    zip(self.imgs, self.masks)]) == {1}
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

            if xmax <= xmin and ymax <= ymin:
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


class InstanceDataset(torch.utils.data.Dataset):
    """Gather and process data for mitophagy

    000000-0-000000000
    RowColumn-fieldofview-timeplanechannel

    Well identities:
    NO TF DMSO: 007001 (G1), 008001 (H1)
    NO TF OA: 007012 (G12), 008012 (H12)
    SCR DMSO: 001001 (A1), 002001 (B1)
    SCR OA: 001012 (A12), 002012 (B12)
    PINK1 DMSO: 003001 (C1), 004001 (D1)
    PINK1 OA: 003012 (C12), 004012 (D12)
    CLN3 OA: 001010 (A10), 002010 (B10)
    """

    def __init__(self, df, morphology_channel, label_dict=None, transforms=None):
        self.df = df
        self.morphology_channel = morphology_channel
        self.transforms = transforms
        self.img_groups = self.df.groupby(by=['imrow', 'imcol', 'fov', 'time', 'plane'])
        self.idx = list(self.img_groups)
        _label_dict = {(7, 1): 1, (8, 1): 1,
                       (7, 12): 2, (8, 12): 2,
                       (1, 1): 3, (2, 1): 3,
                       (1, 12): 4, (2, 12): 4,
                       (3, 1): 5, (4, 1): 5,
                       (3, 12): 6, (4, 12): 6,
                       (1, 10): 7, (2, 10): 7
                       }
        self.label_dict = label_dict if label_dict is not None else _label_dict

        # todo: drop groups if no mask, insufficient channels

    def __getitem__(self, idx):
        # load images and masks
        key = self.idx[idx][0]
        group = self.img_groups.get_group(key)
        group = group.sort_values(by='channel')
        mask_path, row, col = group.loc[group.channel == self.morphology_channel, ['mask', 'imrow', 'imcol']].values[0]
        assert mask_path != 'null'
        # todo: verify mask path exists
        img_paths = group.file.tolist()
        imgs = []
        for img_path in img_paths:
            _im = Image.open(img_path).convert('L')
            imgs.append(_im)
        img = Image.merge("RGB", imgs)

        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
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

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # class labels
        labels = torch.ones((num_objs,), dtype=torch.int64) * self.label_dict[(row, col)]
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
        return len(self.img_groups)


class Process:
    def __init__(self, datadir, imagedir, maskdir=None):
        self.datadir = datadir
        self.imagedir = imagedir
        self.maskdir = maskdir

    def make_dataframe(self, morphology_channel):
        files = glob(os.path.join(self.imagedir, '*.tif'))
        # Make dataframe
        d = {'file': [], 'filename': [], 'imrow': [], 'imcol': [], 'fov': [], 'time': [], 'plane': [], 'channel': [],
             'mask': []}
        for f in files:
            d['file'].append(f)
            filename = f.split('/')[-1]
            d['filename'].append(filename)
            d['imrow'].append(self.get_row(filename))
            d['imcol'].append(self.get_col(filename))
            d['fov'].append(self.get_fov(filename))
            d['time'].append(self.get_time(filename))
            d['plane'].append(self.get_plane(filename))
            channel = self.get_channel(filename)
            d['channel'].append(channel)
            if self.maskdir is not None and channel == morphology_channel:
                stem = filename.split('.')[0]
                mfiles = glob(os.path.join(self.maskdir, stem + '*.tif'))
                assert len(mfiles) <= 1
                if len(mfiles) == 0:
                    d['mask'].append('null')
                else:
                    d['mask'].append(mfiles[0])
            else:
                d['mask'].append('null')
        df = pd.DataFrame(d)
        df.to_csv(os.path.join(self.datadir, 'data.csv'))
        return df

    def split_train_val_test(self, df, split=[.8, .1, .1]):
        """
        Split to train val test by groups
        :param df:
        :return:
        """
        img_groups = df.groupby(by=['imrow', 'imcol', 'fov', 'time', 'plane'])
        a = np.arange(img_groups.ngroups)
        np.random.shuffle(a)
        train_df = df[img_groups.ngroup().isin(a[:int(len(img_groups) * split[0])])]
        df = df.drop(train_df.index)

        img_groups = df.groupby(by=['imrow', 'imcol', 'fov', 'time', 'plane'])
        a = np.arange(img_groups.ngroups)
        np.random.shuffle(a)
        test_df = df[img_groups.ngroup().isin(a[:int(len(img_groups) * split[1] / (1 - split[0]))])]
        val_df = df.drop(test_df.index)
        return train_df, val_df, test_df

    def select_df(self, df, fovs=[], planes=[], channels=[], rows=None, cols=None):
        if rows is not None and cols is not None:
            df = df[(df.fov.isin(fovs)) & (df.plane.isin(planes)) & (df.channel.isin(channels)) &
                    (df.imrow.isin(rows)) & (df.imcol.isin(cols))]
        else:
            df = df[(df.fov.isin(fovs)) & (df.plane.isin(planes)) & (df.channel.isin(channels))]
        return df

    def get_row(self, filename):
        row = int(filename[:3])
        return row

    def get_col(self, filename):
        col = int(filename[3:6])
        return col

    def get_fov(self, filename):
        return int(filename[7])

    def get_time(self, filename):
        return int(filename[9:12])

    def get_plane(self, filename):
        return int(filename[12:15])

    def get_channel(self, filename):
        return int(filename[15:18])


def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default=r'/gladstone/finkbeiner/lab/MITOPHAGY/')
    parser.add_argument('--imagedir', default=r'/gladstone/finkbeiner/lab/MITOPHAGY/Images for MitophAIgy Test')
    parser.add_argument('--maskdir', default=r'/gladstone/finkbeiner/lab/MITOPHAGY/Masks')
    args = parser.parse_args()
    print('ARGS: ', args)
    collate_fn = lambda _: tuple(zip(*_))  # one-liner, no need to import

    Proc = Process(args.datadir, args.imagedir, args.maskdir)
    df = Proc.make_dataframe(morphology_channel=3)
    dataset = InstanceDataset(df, morphology_channel=3, transforms=get_transform(train=False))
    data_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=collate_fn)
    for img, target in data_loader:
        print(len(img), img[0].size())
