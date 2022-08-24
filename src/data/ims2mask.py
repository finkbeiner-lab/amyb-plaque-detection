"""Take crops of images and use otsu to get corresponding masks
todo: is picking up neurites and cell debris"""

import cv2
import imageio
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm


class Mask:
    def __init__(self, cropdir, savedir, lbl, plotbool=True):
        """
        lbl is a global label, doesn't handle multiple labels in image at the moment. Lbl is based on directory.
        """
        self.cropdir = cropdir
        self.savedir = savedir
        self.lbl = lbl
        self.plotbool = plotbool
        # dir > wells > crops
        self.imgs = glob(os.path.join(self.cropdir, '*', '*.tif'))
        self.ecn = 0.1
        self.small = 50
        self.large = 2500

    def get_cells_by_contours(self, contours):
        contours_kept = []
        for cnt in contours:
            if len(cnt) > 0 and cv2.contourArea(cnt) > self.small and cv2.contourArea(cnt) < self.large:
                (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
                ecc = np.sqrt(1 - ((MA) ** 2 / (ma) ** 2))
                print('area', cv2.contourArea(cnt))
                print('eccentricity', ecc)
                if ecc >= self.ecn:
                    contours_kept.append(cnt)
        return contours_kept

    def get_cells_by_hough(self, img, debug=True):
        def pixelVal(pix, r1, s1, r2, s2):
            if (0 <= pix and pix <= r1):
                return (s1 / r1) * pix
            elif (r1 < pix and pix <= r2):
                return ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1
            else:
                return ((255 - s2) / (255 - r2)) * (pix - r2) + s2

            # Define parameters.

        r1 = 70
        s1 = 0
        r2 = 200
        s2 = 255
        gray = np.uint8(img / np.max(img) * 255)
        blurM = cv2.medianBlur(gray, 5)
        pixelVal_vec = np.vectorize(pixelVal)
        contrast_stretched_blurM = pixelVal_vec(blurM, r1, s1, r2, s2)

        processed = np.uint8(contrast_stretched_blurM)

        # morphological operations
        kernel = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(processed, kernel, iterations=1)
        closing = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)

        # Adaptive thresholding on mean and gaussian filter
        th2 = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                    cv2.THRESH_BINARY, 11, 2)
        th3 = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                    cv2.THRESH_BINARY, 11, 2)
        # Otsu's thresholding
        ret4, th4 = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if debug:
            plt.figure()
            plt.title('gray')
            plt.imshow(gray)

            plt.figure()
            plt.title('blurM')
            plt.imshow(blurM)

            plt.figure()
            plt.title('processed')
            plt.imshow(processed)

            plt.figure()
            plt.title('th2')
            plt.imshow(th2)

            plt.figure()
            plt.title('th3')
            plt.imshow(th3)

            plt.figure()
            plt.title('th4')
            plt.imshow(th4)
        # Initialize the list
        Cell_count, x_count, y_count = [], [], []

        # read original image, to display the circle and center detection
        # display = cv2.imread("D:/Projects / ImageProcessing / DA1 / sample1 / cellOrig.png")

        # hough transform with modified circular parameters
        circles = cv2.HoughCircles(processed, cv2.HOUGH_GRADIENT, 1.2, 20,
                                   param1=200, param2=15, minRadius=1, maxRadius=20)

        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")

            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in circles:
                cv2.circle(gray, (x, y), r, (255, 255, 0), 2)
                cv2.rectangle(gray, (x - 2, y - 2),
                              (x + 2, y + 2), (0, 128, 255), -1)
                Cell_count.append(r)
                x_count.append(x)
                y_count.append(y)
            # show the output image
            plt.figure()
            plt.title('hough')
            plt.imshow(gray)
            # cv2.imshow("gray", display)
            # cv2.waitKey(0)
        plt.show()

    def make_mask(self):
        for f in self.imgs:
            well = f.split('/')[-2]
            name = f.split('/')[-1]
            savename = 'MASK_' + name
            wellfolder = os.path.join(self.savedir, well)
            savepath = os.path.join(wellfolder, savename)

            img = imageio.v2.imread(f)
            img = np.uint8(np.float32(img) / np.max(img) * 255)
            ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_16U)
            contours, hierarchy = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # self.get_cells_by_hough(img)
            # contours = self.get_cells_by_contours(contours)
            # labels = np.zeros_like(img, dtype=np.uint8)
            # cv2.drawContours(labels, contours, -1, 255, -1)
            # print(np.unique(labels))
            # print(stats[-1])
            # areaidx = np.argwhere(stats[:, -1] > 40)
            # for i in range(numLabels):
            #     if i not in areaidx:
            #         labels[labels == i] = 0
            labels[labels > 0] = 1
            kernel = np.ones((5, 5), np.uint8)
            labels = cv2.dilate(np.uint8(labels), kernel, iterations=1)
            labels = cv2.erode(labels, np.ones((3, 3), np.uint8), iterations=1)
            labels[labels > 0] = self.lbl
            if self.plotbool:
                plt.figure()
                plt.imshow(img)
                plt.figure()
                plt.imshow(labels)
                plt.show()
            if not os.path.exists(wellfolder):
                os.makedirs(wellfolder)
            imageio.v2.imwrite(savepath, labels)
        print(f'Saved masks to {self.savedir}')


class DeleteBlankMasks:
    def __init__(self, root):
        self.crops = list(sorted(glob(os.path.join(root, "images", '*Confocal-GFP16*.tif'))))
        # self.masks = list(sorted(glob(os.path.join(root, "**", '*.tif'))))
        self.masks = list(sorted(glob(os.path.join(root, "labels", '*Confocal-GFP16*.tif'))))

    def rename_masks(self):
        for m in tqdm(self.masks):
            parts = m.split('/')
            name = parts[-1].split('_MAS')[0]
            name = 'MASK_' + name + '.tif'
            parts[-1] = name
            dst = '/'.join(parts)
            os.rename(m, dst)

    def delete_blanks(self):
        for c, m in zip(self.crops, self.masks):
            mask = imageio.v3.imread(m)
            if np.all(mask == 0):
                print('delete')
                print(c, m)
                assert c.split('/')[-1].split('.t')[0] in m.split('/')[-1]
                os.remove(c)
                os.remove(m)


if __name__ == '__main__':
    cropdirs = ['/mnt/linsley/Shijie_ML/Ms_Tau/P301S-Tau', '/mnt/linsley/Shijie_ML/Ms_Tau/WT-Tau']
    savedirs = ['/mnt/linsley/Shijie_ML/Ms_Tau/P301S-Tau_Labelx', '/mnt/linsley/Shijie_ML/Ms_Tau/WT-Tau_Labelx']
    # for lbl, (cropdir, savedir) in enumerate(zip(cropdirs, savedirs)):
    #     Msk = Mask(cropdir, savedir, lbl + 1)
    #     Msk.make_mask()

    datadirs = ['/mnt/linsley/Shijie_ML/Ms_Tau/dataset/train',
                '/mnt/linsley/Shijie_ML/Ms_Tau/dataset/val',
                '/mnt/linsley/Shijie_ML/Ms_Tau/dataset/test']
    for datadir in datadirs:
        Delt = DeleteBlankMasks(root=datadir)
        # Delt.delete_blanks()
        Delt.rename_masks()
