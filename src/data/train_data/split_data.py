
#Preprocessing
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from numpy import expand_dims
from shutil import copyfile


#Visualization
from tqdm import tqdm
from tqdm.auto import trange
from PIL import Image
from colorama import Fore


import glob
import pdb
import os
import random
import time
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool
import shutil
import albumentations as A
import cv2
import numpy as np

__author__ = 'Vivek Gopal Ramaswamy'


class SplitData:
    """
    This is a class for splitting the data into train, test and val

    """

    def __init__(self, dataset_base_dir, data_aug, aug_value):
        self.dataset_base_dir = dataset_base_dir
        self.data_aug = data_aug
        self.aug_value = aug_value
        self.cores = multiprocessing.cpu_count()
        self.image_input = "images"
        self.label_input = "labels"
        self.transforms  = A.Compose([
                            A.VerticalFlip(p=0.5),
                            A.HorizontalFlip(p=0.5),
                            A.Blur(blur_limit=3),
                            A.OpticalDistortion(),
                            A.HueSaturationValue(),
                            A.RandomRotate90(),
                            A.RandomBrightnessContrast(p=0.2),
                        ])



    def generate_split_dirs(self):
        '''Generate Proper Directory strucute with labels under train test
        and val'''

        labeldirs = ['images', 'labels']
        subdirs = ['train', 'test', 'val']

        print("\nSplitting Directory ...")

        for labldir in labeldirs:
            newdir = os.path.join(self.dataset_base_dir, labldir)
            if not os.path.exists(newdir):
                os.makedirs(newdir)
                print("Directory '%s' created" %newdir)

        for subdir in subdirs: # To generate T and F folder under train test and val
            for labldir in labeldirs:
                newdir = os.path.join(self.dataset_base_dir, subdir, labldir)
                if not os.path.exists(newdir):
                    os.makedirs(newdir)
                print("Directory '%s' created" %newdir)

    def split_dataset(self, image_filenames, label_filenames, dst_directory):
        '''This Fn will split the dataset into train test and val for
        contents in images and labels folder'''

        # Step 1 : Sort the filenames
        image_filenames.sort()
        label_filenames.sort()

        # Step 2 : Shuffle the filenames in the same order for images and labels

        temp = list(zip(image_filenames, label_filenames))
        random.seed(230)# shuffles the ordering of filenames
        #(deterministic given the chosen seed)
        random.shuffle(temp)
        image_filenames, label_filenames = zip(*temp)

        # Step 3 : Create 80 - 10 - 10 Split
        split_1 = int(0.8 * len(image_filenames))
        split_2 = int(1 * len(image_filenames))

        split_3 = int(0.8 * len(label_filenames))
        split_4 = int(1 * len(label_filenames))

        print("\n Splitting dataset into Train and Val for  Images ...")

        data_dict = dict(
            train=dict(
                images=image_filenames[:split_1],
                labels=label_filenames[:split_1],
            ),
            val=dict(
                images=image_filenames[split_1:split_2],
                labels=label_filenames[split_1:split_2],
            )
        )

        train_filenames_images = image_filenames[:split_1]
        val_filenames_images = image_filenames[split_1:split_2]

        print("\n\nSplitting dataset into Train and Val for Labels ...")

        train_filenames_labels = label_filenames[:split_3]
        val_filenames_labels = label_filenames[split_3:split_4]


        # Step 4 : Copy the split contents to folders
        #Parallel Process - Images and Labels
        [self.copy_split_files_to_dataset(data_dict[fold][x], dst_directory, f'{fold}/{x}') for x in 'images labels'.split() for fold in 'train val'.split()]


    def copy_split_files_to_dataset(self, filenames, dst_directory, dst_type):
        '''
        This Fn will copy all the file belonging to true and false classes to
        respective T and F folders

        Parameters:
        filenames -- the source filenames for True or False classes
        dst_directory -- the destination directory where the files will be copied
        to
        dst_type -- describes the destination of the files controlled by the
        split type ex: train/T or val/F
        '''
        for src_file in tqdm(filenames, bar_format="{l_bar}%s{bar}%s{r_bar}" %
                             (Fore.BLUE, Fore.RESET)):
            filename = os.path.basename(src_file)
            dst = os.path.join(dst_directory, dst_type, filename)
            copyfile(src_file, dst)

    def check_distribution(self, name, image_filenames, label_filenames):
        '''This Fn will check for the distribution of Images and labels specified by
        the source name'''
        assert len(image_filenames) == len(label_filenames)
        print("\n========Check Distribution==========")
        print("\nSource : ", name)
        print('\nTotal Images :', len(image_filenames))
        print('\nTotal Labels :', len(label_filenames))
        print("\n====================================")

    def get_randimages_dataug(self, total_imgs, image_filenames, label_filenames):
        '''
        This Fn generates random files from the original dataset, which will
        be used to perform data augmentation

        Parameters:
        image_filenames -- list of filenames of images
        label_filenames -- list of filenames of labels
        total_imgs -- how many images needs to be augmented

        Return:
        list of random image and label files for performing data augmentation
        '''
        image_filenames.sort()
        label_filenames.sort()
        random_image_file = []
        random_label_file = []

        for i in trange(total_imgs):
            random.seed(i)
            random_image_file.append(random.choice(image_filenames))
            random.seed(i)
            random_label_file.append(random.choice(label_filenames))

        return [random_image_file, random_label_file]

    def upsample_dataset(self, random_img_filenames, rand_label_filenames, variations):
        '''
        This Fn will upsample the images by performing data augmentation

        DataAugmentation includes vertical and horizontal flips

        Parameters:
        random_filenames -- the random filenames of the images for performing data augmentation
        file_type -- if it belongs to images or labels
        variations -- how many variations you need from each image for
        generating your data augmented sample

        '''
        i = 0
        aug_img_files = []
        aug_mask_files = []
        # random.seed(500)

        # Make dir where tha augmented file will reside
        aug_img_dir = os.path.join(self.dataset_base_dir, "augmented_images")
        if not os.path.exists(aug_img_dir):
                os.makedirs(aug_img_dir)
                print("Augmented Directory '%s' created" %aug_img_dir)
        
        aug_mask_dir = os.path.join(self.dataset_base_dir, "augmented_labels")
        if not os.path.exists(aug_mask_dir):
                os.makedirs(aug_mask_dir)
                print("Augmented Directory '%s' created" %aug_mask_dir)


        print("\nData Augmentation in Progress ...")
        total_imgs = len(random_img_filenames)

        for i in trange(total_imgs):

            # load the image
            img = Image.open(random_img_filenames[i]).convert("RGB")
            img = np.array(img)
            mask = Image.open(rand_label_filenames[i]).convert('P')
            mask = np.array(mask)

            for j in range(variations):
            
                transformed = self.transforms(image=img, mask=mask)

                transformed_img = transformed["image"]
                transformed_img = Image.fromarray(transformed_img)


                #To rename the file with prefix A_
                filename = os.path.basename(random_img_filenames[i])
                filepath = os.path.dirname(random_img_filenames[i])

                aug_file_name = "A_" + str(i) + "_" + str(j) + "_" + filename
                new_file = os.path.join(self.dataset_base_dir, "augmented_images",
                                        aug_file_name)
                transformed_img.save(new_file)
                aug_img_files.append(new_file)

                transformed_mask = transformed["mask"]
                transformed_mask = Image.fromarray(transformed_mask)

                #To rename the file with prefix A_
                filename = os.path.basename(rand_label_filenames[i])
                filepath = os.path.dirname(rand_label_filenames[i])
                aug_file_name = "A_" + str(i) + "_" + str(j) + "_" + filename
                new_file = os.path.join(self.dataset_base_dir, "augmented_labels",
                                        aug_file_name)
                transformed_mask.save(new_file)
                aug_mask_files.append(new_file)
                
        return aug_img_files, aug_mask_files

    def preprocess_dataset(self, image_filenames, label_filenames):
        '''
        This Fn arranges the dataset into different images and lables folder
        and performs the following on the dataset
        - data augmentation
        - splits dataset into train test and val folders

        Parameters:
        - image_filenames - images are accessed from remote box folder
        - label_filenames - labels are accessed from remote box folder

       '''

        image_filenames.sort()
        label_filenames.sort()

        # split the images into train test and val
        self.check_distribution("original_data", image_filenames, label_filenames)
        # self.split_dataset(image_filenames, label_filenames, self.dataset_base_dir)

        # Step 3 Data Augmentation Block
        if self.data_aug:
            rand_image_filenames, rand_label_filenames = self.get_randimages_dataug(self.aug_value,
                                                                                    image_filenames,
                                                                                    label_filenames)
            augmented_image_files, augmented_label_files = self.upsample_dataset(rand_image_filenames, rand_label_filenames, 3)
          
            self.check_distribution("augmented_data", augmented_image_files, augmented_label_files)
            print("\n Total Data :", len(augmented_image_files) + len(image_filenames))

            destination_path = os.path.join(self.dataset_base_dir, "images")

            self.copy_split_files_to_dataset("images", augmented_image_files, destination_path)

            destination_path = os.path.join(self.dataset_base_dir, "labels")

            #  copying files from Augmented folder to images and labels

            self.copy_split_files_to_dataset("labels", augmented_label_files, destination_path)

            # self.split_dataset(augmented_image_files, augmented_label_files, self.dataset_base_dir)

        # self.visualization.check_images(augmented_image_files, 2)
        # self.visualization.check_images(augmented_label_files, 2)




    def prepare_dataset(self):
        '''This Fn does performs all the necessary actions to prepare the dataset
        for training the model'''

        # Extracting Image File Names
        images_input = os.path.join(self.dataset_base_dir, self.image_input)
        image_path = os.path.join(images_input, '*.png')
        image_filenames = glob.glob(image_path)

        # Extracting labels File Names
        label_input = os.path.join(self.dataset_base_dir, self.label_input)
        label_path = os.path.join(label_input, '*.png')
        label_filenames = glob.glob(label_path)

        assert len(image_filenames) != 0 and len(label_filenames) != 0

        # self.generate_split_dirs()
        self.preprocess_dataset(image_filenames, label_filenames)


#

if __name__ == "__main__":
    #TODO Run this for train and val separately - since its patient specific split
    # Run merge file before running this file
    base_WSI_path = '/mnt/new-nas/work/data/npsad_data/vivek/Datasets/amyb_wsi/val'
    split_data = SplitData(base_WSI_path, True, 200)
    split_data.prepare_dataset()
