import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import imageio

# read original image
f='/mnt/linsley/Shijie_ML/Ms_Tau/P301S-Tau/A7/PID20220505_2022-0505-MsNeuron-RGEDI-Tau-TauP301S_T0_0.0-0_A7_0_Confocal-GFP16_0_0_1_BGs_MN_ALIGNED_1.tif'
savedir = '/mnt/linsley/Shijie_ML/Ms_Tau'
gray = imageio.v2.imread(f)

# convert to gray scale image
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = np.uint8(gray / np.max(gray) * 255)
plt.figure()
plt.title('gray')

plt.imshow(gray)

# cv2.imwrite('gray.png', gray)

# apply median filter for smoothing
blurM = cv2.medianBlur(gray, 5)
plt.figure()
plt.title('blurM')
plt.imshow(blurM)
# cv2.imwrite('blurM.png', blurM)

# apply gaussian filter for smoothing
blurG = cv2.GaussianBlur(gray, (9, 9), 0)
plt.figure()
plt.title('blurG')

plt.imshow(blurG)

# cv2.imwrite('blurG.png', blurG)

# histogram equalization
histoNorm = cv2.equalizeHist(gray)
plt.figure()
plt.title('histoNorm')
plt.imshow(histoNorm)

# cv2.imwrite('histoNorm.png', histoNorm)

# create a CLAHE object for
# Contrast Limited Adaptive Histogram Equalization (CLAHE)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
claheNorm = clahe.apply(gray)
plt.figure()
plt.title('clahe')
plt.imshow(claheNorm)

# cv2.imwrite('claheNorm.png', claheNorm)


# contrast stretching
# Function to map each intensity level to output intensity level.
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

# Vectorize the function to apply it to each value in the Numpy array.
pixelVal_vec = np.vectorize(pixelVal)

# Apply contrast stretching.
contrast_stretched = pixelVal_vec(gray, r1, s1, r2, s2)
plt.figure()
plt.title('contrast_stretched')
plt.imshow(contrast_stretched)

contrast_stretched_blurM = pixelVal_vec(blurM, r1, s1, r2, s2)
plt.figure()
plt.title('contrast_stretched_blurM')

plt.imshow(contrast_stretched_blurM)


# cv2.imwrite('contrast_stretch.png', contrast_stretched)
# cv2.imwrite('contrast_stretch_blurM.png',
#             contrast_stretched_blurM)

# edge detection using canny edge detector
edge = cv2.Canny(gray, 100, 200)
# cv2.imwrite('edge.png', edge)

edgeG = cv2.Canny(blurG, 100, 200)
# cv2.imwrite('edgeG.png', edgeG)

edgeM = cv2.Canny(blurM, 100, 200)
# cv2.imwrite('edgeM.png', edgeM)
plt.show()