"""
An example for histological images color normalization based on the adaptive color deconvolution as described in the paper:
https://github.com/Zhengyushan/adaptive_color_deconvolution

Yushan Zheng, Zhiguo Jiang, Haopeng Zhang, Fengying Xie, Jun Shi, and Chenghai Xue.
Adaptive Color Deconvolution for Histological WSI Normalization.
Computer Methods and Programs in Biomedicine, v170 (2019) pp.107-120.

"""
import os
import cv2
import numpy as np
from glob import glob
from stain_normalizer import StainNormalizer
#from stain_normalizer import StainNormalizer
import imageio.v3 as iio
import matplotlib.pyplot as plt
import torch
import torchmetrics
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import multiscale_structural_similarity_index_measure as msssim
from glob import glob
import pickle

def compute_metrics(img_path, img_names):
    psnr_list = []
    ssim_list = []
    msssim_list = []
    for img_name in img_names:
        orig_img = os.path.join(img_path,img_name+"_origin.jpg")
        norm_img = os.path.join(img_path,img_name+"_norm.jpg")
        orig_img_load = iio.imread(orig_img)
        orig = torch.torch.from_numpy(orig_img_load.reshape((3, 1024, 1024))).float()
        norm_img_load = iio.imread(norm_img)
        norm = torch.torch.from_numpy(norm_img_load.reshape((3, 1024, 1024))).float()
        psnr_val = psnr(norm, orig)
        ssim_val = ssim(norm[None, :], orig[None, :])
        msssim_val = msssim(norm[None, :], orig[None, :])
        psnr_list.append(psnr_val.cpu().numpy())
        ssim_list.append(ssim_val.cpu().numpy())
        msssim_list.append(msssim_val.cpu().numpy())
    print("PSNR - ","mean: ", np.mean(psnr_list), "max: ", np.max(psnr_list), "min: ", np.min(psnr_list), "std: ", np.std(psnr_list))
    print("SSIM - ","mean: ", np.mean(ssim_list), "max: ", np.max(ssim_list), "min: ", np.min(ssim_list), "std: ", np.std(ssim_list))
    print("MSSSIM - ","mean: ", np.mean(msssim_list), "max: ", np.max(msssim_list), "min: ", np.min(msssim_list), "std: ", np.std(msssim_list))
    return psnr_list, ssim_list, msssim_list

def plot_rgb_hist(img_path):
    plant_seedling = iio.imread(img_path)

    # display the image
    fig, ax = plt.subplots()
    ax.imshow(plant_seedling)

    # tuple to select colors of each channel line
    colors = ("red", "green", "blue")

    # create the histogram plot, with three lines, one for
    # each color
    fig, ax = plt.subplots()
    ax.set_xlim([0, 256])
    for channel_id, color in enumerate(colors):
        histogram, bin_edges = np.histogram(
            plant_seedling[:, :, channel_id], bins=256, range=(0, 256)
        )
        ax.plot(bin_edges[0:-1], histogram, color=color)

    ax.set_title("Color Histogram")
    ax.set_xlabel("Color value")
    ax.set_ylabel("Pixel count")
# disable GPU
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#source_image_dir = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/WSI_Normalization_techniques/StainNet/data"
source_image_dir = "/home/mahirwar/Desktop/Monika/npsad_data/vivek/Datasets/amyb_wsi"
#'data/images'
#template_dir = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/WSI_Normalization_techniques/StainNet/test"
#template_dir1 = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/WSI_Normalization_techniques/StainNet/data/11_063_CG_aSyn_x200.svs"
#template_dir2 = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/WSI_Normalization_techniques/StainNet/data/12_007_CG_aSyn_x200.svs"
#template_dir3 = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/WSI_Normalization_techniques/StainNet/data/PD001_Syn1_CG.svs"
#template_dir4 = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/WSI_Normalization_techniques/StainNet/data/PD067_Syn1_CG.svs"
#'data/template'
#img_list =["11_063_CG_aSyn_x200.svs","12_007_CG_aSyn_x200.svs","PD001_Syn1_CG.svs","PD067_Syn1_CG.svs"]
img_list =["XE12-010_1_AmyB_1","XE17-010_1_AmyB_1","XE11-025_1_AmyB_1","XE07-049_1_AmyB_1"]
#"XE18-004_1_AmyB_1",
template_list = []
temp_dir = "/home/mahirwar/Desktop/Monika/npsad_data/vivek/Datasets/amyb_wsi"

for img in img_list:
    template_list.extend(glob(os.path.join(temp_dir,img,"image","*.png")))
    #template_list.extend(glob(os.path.join(temp_dir,img,"images","*.png")))

result_dir = '/home/mahirwar/Desktop/Monika/npsad_data/vivek/Datasets/amyb_wsi_normalized'

# load template images
#template_list = os.listdir(template_dir)
#template_list1 = glob(os.path.join(template_dir1,"*/*.png"))
#template_list2 = glob(os.path.join(template_dir2,"*/*.png"))
#template_list3 = glob(os.path.join(template_dir3,"*/*.png"))
#template_list4 = glob(os.path.join(template_dir4,"*/*.png"))
#template_list = template_list1+template_list2+template_list3+template_list4
#print(len(template_list1))
if ".DS_Store" in template_list:
    template_list.remove(".DS_Store")
#template_list.remove(".DS_Store")
#print(len(template_list))
#temp_images = np.asarray([cv2.imread(os.path.join(template_dir, name)) for name in template_list])
temp_images = np.asarray([cv2.imread(name) for name in template_list])

# extract the stain parameters of the template slide
normalizer = StainNormalizer()
normalizer.fit(temp_images[:,:,:,[2,1,0]]) #BGR2RGB

with open('/home/mahirwar/Desktop/Monika/npsad_data/vivek/Datasets/amyb_wsi_normalized/amyb_normalizer.pkl', 'wb') as f:
    pickle.dump(normalizer, f, pickle.HIGHEST_PROTOCOL)
#print(normalizer)

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# normalization
#slide_list = os.listdir(source_image_dir)

slide_list =  ["XE09-063_1_AmyB_1","XE14-004_1_AmyB_1","XE19-037_1_AmyB_1","XE18-003_1_AmyB_1","XE18-066_1_AmyB_1","XE15-022_1_AmyB_1"]
slide_list =  glob(os.path.join(source_image_dir,"XE*"))
slide_list = [ x.split("/")[-1] for x in slide_list]
"""
slide_list = ["XE07-064_1_AmyB_1",
"XE15-039_1_AmyB_1",
"XE17-039_1_AmyB_1",
"XE15-022_1_AmyB_1",
"XE10-045_1_AmyB_1",
"XE19-028_1_AmyB_1",
"XE19-010_1_AmyB_1",
"XE10-053_1_AmyB_1",
"XE09-063_1_AmyB_1",
"XE12-010_1_AmyB_1"]
"""
if ".DS_Store" in slide_list:
    slide_list.remove(".DS_Store")
#slide_list.remove("screenshots")
region = "image"
#slide_list = ["screenshots"]
for s in slide_list:
    print('normalize slide', s)
    slide_dir = os.path.join(source_image_dir, s, region)
    image_list = os.listdir(slide_dir)
    if ".DS_Store" in image_list:
        image_list.remove(".DS_Store")
    images = np.asarray([cv2.imread(os.path.join(slide_dir, name)) for name in image_list])
    print(len(images))
    print(images[0].shape)
    ## color transform
    results = normalizer.transform(images[:,:,:,[2,1,0]]) #BGR2RGB
    print(len(results))
    if not os.path.exists(os.path.join(result_dir,s, region)):
        os.makedirs(os.path.join(result_dir,s, region))
        
    for result, img_name in zip(results, image_list):
        #cv2.imwrite(os.path.join(result_dir, s , images[i])
        cv2.imwrite(os.path.join(result_dir, s, region, img_name), result[:,:,[2,1,0]]) #RGB2BGR

    ## h&e decomposition
    #he_channels = normalizer.he_decomposition(images[:,:,:,[2,1,0]], od_output=True) #BGR2RGB
    # debug display
    #for i, result in enumerate(he_channels):
    #    cv2.imwrite(os.path.join(result_dir, s + '_{}_h.jpg'.format(i)), result[:,:,0]*128)
    #    cv2.imwrite(os.path.join(result_dir, s + '_{}_e.jpg'.format(i)), result[:,:,1]*128)

"""
image_list = os.listdir(source_image_dir)
images = np.asarray([cv2.imread(os.path.join(source_image_dir, name)) for name in image_list])
## color transform
results = normalizer.transform(images[:,:,:,[2,1,0]]) #BGR2RGB
# display
for i, result in enumerate(results):
    cv2.imwrite(os.path.join(result_dir, '_{}_origin.jpg'.format(i)), images[i])
    cv2.imwrite(os.path.join(result_dir, '_{}_norm.jpg'.format(i)) , result[:,:,[2,1,0]]) #RGB2BGR

## h&e decomposition
he_channels = normalizer.he_decomposition(images[:,:,:,[2,1,0]], od_output=True) #BGR2RGB    
for i, result in enumerate(he_channels):
    cv2.imwrite(os.path.join(result_dir, '_{}_h.jpg'.format(i)), result[:,:,0]*128)
    cv2.imwrite(os.path.join(result_dir, '_{}_e.jpg'.format(i)), result[:,:,1]*128)
"""

#img_path = os.path.join(result_dir,region)
#l = glob(os.path.join(img_path,"*.jpg"))
#img_names = ["_".join(x.split("/")[-1].split("_")[:-1]) for x in l]
#psnr_grey_test, ssim_grey_test, msssim_grey_test = compute_metrics(img_path, img_names)


