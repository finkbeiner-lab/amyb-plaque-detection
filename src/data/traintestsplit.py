"""Train test split dataset
Take exp > wells > crops, masks > wells > crops and shuffle to train val test > exp > wells > crops"""

import os
from glob import glob
import shutil
import numpy as np
import random
from tqdm import tqdm


class DataSorter:
    def __init__(self, homedir, posdir, posmaskdir, negdir, negmaskdir):
        self.homedir = homedir
        self.poscrops = sorted(glob(os.path.join(posdir, '*', '*Confocal-GFP16*.tif')))
        self.posmasks = sorted(glob(os.path.join(posmaskdir, '*', '*Confocal-GFP16*.tif')))
        self.negcrops = sorted(glob(os.path.join(negdir, '*', '*Confocal-GFP16*.tif')))
        self.negmasks = sorted(glob(os.path.join(negmaskdir, '*', '*Confocal-GFP16*.tif')))
        self.datadir = os.path.join(homedir, 'dataset')


    def shuffle_data(self):
        """Shuffle data, crops and masks together."""
        # pidx = [i for i in range(len(self.poscrops))]
        # nidx = [i for i in range(len(self.negcrops))]
        random.seed(11)
        random.shuffle(self.poscrops)
        random.seed(11)
        random.shuffle(self.posmasks)
        random.seed(121)
        random.shuffle(self.negcrops)
        random.seed(121)
        random.shuffle(self.negmasks)
        plen = len(self.poscrops)
        nlen = len(self.negcrops)
        # self.poscrops = self.poscrops[pidx]
        # self.posmasks = self.posmasks[pidx]
        # self.negcrops = self.negcrops[nidx]
        # self.negmasks = self.negmasks[nidx]

        # cutoff data to be same length
        print('Cutting off data')
        if plen > nlen:
            self.poscrops = self.poscrops[:nlen]
            self.posmasks = self.posmasks[:nlen]
        elif plen < nlen:
            self.negcrops = self.negcrops[:plen]
            self.negmasks = self.negmasks[:plen]
        # to train, val, test
        length = len(self.poscrops)

        crops = self.poscrops + self.negcrops
        masks = self.posmasks + self.negmasks
        random.seed(121)
        random.shuffle(crops)
        random.seed(121)
        random.shuffle(masks)

        traindir = os.path.join(self.datadir, 'train')
        valdir = os.path.join(self.datadir, 'val')
        testdir = os.path.join(self.datadir, 'test')
        self.traincrops = list(crops[:int(length * .7)]) + list(crops[:int(length * .7)])
        self.trainlabels = list(masks[:int(length * .7)]) + list(masks[:int(length * .7)])
        self.valcrops = list(crops[int(length * .7):int(length * .85)]) + list(
            crops[int(length * .7):int(length * .85)])
        self.vallabels = list(masks[int(length * .7):int(length * .85)]) + list(
            masks[int(length * .7):int(length * .85)])
        self.testcrops = list(crops[int(length * .85):]) + list(crops[int(length * .85):])
        self.testlabels = list(masks[int(length * .85):]) + list(masks[int(length * .85):])
        if not os.path.exists(traindir):
            os.makedirs(os.path.join(traindir, 'images'))
            os.makedirs(os.path.join(traindir, 'labels'))
            os.makedirs(os.path.join(valdir, 'images'))
            os.makedirs(os.path.join(valdir, 'labels'))
            os.makedirs(os.path.join(testdir, 'images'))
            os.makedirs(os.path.join(testdir, 'labels'))

        def save(imgs, lbls, savedir):
            for img, lbl in tqdm(zip(imgs, lbls)):
                imgdst = os.path.join(savedir, 'images', img.split('/')[-1])
                maskdst = os.path.join(savedir, 'labels', lbl.split('/')[-1])
                shutil.copyfile(img, imgdst)
                shutil.copyfile(lbl, maskdst)

        print('Saving train')
        save(self.traincrops, self.trainlabels, traindir)
        print('Saving val')
        save(self.valcrops, self.vallabels, valdir)
        print('Saving test')
        save(self.testcrops, self.testlabels, testdir)
        print('Finished!')


if __name__ == '__main__':
    homedir = '/mnt/linsley/Shijie_ML/Ms_Tau'
    posdir = '/mnt/linsley/Shijie_ML/Ms_Tau/P301S-Tau'
    posmaskdir = '/mnt/linsley/Shijie_ML/Ms_Tau/P301S-Tau_Label'
    negdir = '/mnt/linsley/Shijie_ML/Ms_Tau/WT-Tau'
    negmaskdir = '/mnt/linsley/Shijie_ML/Ms_Tau/WT-Tau_Label'
    DS = DataSorter(homedir, posdir=posdir, posmaskdir=posmaskdir, negdir=negdir, negmaskdir=negmaskdir)
    DS.shuffle_data()
