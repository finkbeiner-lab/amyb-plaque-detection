
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
from pathos.multiprocessing import ProcessingPool as Pool
import shutil

__author__ = 'Vivek Gopal Ramaswamy'


class SplitData:
    """
    This is a class for splitting the data into train, test and val

    """

    def __init__(self, dataset_base_dir):
        self.dataset_base_dir = dataset_base_dir
        self.data_aug = False
        self.cores = multiprocessing.cpu_count()
        self.image_input = "images"
        self.label_input = "masks"
        
    
    def generate_split_dirs(self):
        '''Generate Proper Directory strucute with lables under train test
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
        split_2 = int(0.9 * len(image_filenames))

        split_3 = int(0.8 * len(label_filenames))
        split_4 = int(0.9 * len(label_filenames))

        print("\n\nSplitting dataset into Train,Test and Val for  Images ...")
        train_filenames_images = image_filenames[:split_1]
        test_filenames_images = image_filenames[split_2:]
        val_filenames_images = image_filenames[split_1:split_2]

        print("\n\nSplitting dataset into Train,Test and Val for Labels ...")
        train_filenames_labels = label_filenames[:split_3]
        test_filenames_labels = label_filenames[split_4:]
        val_filenames_labels = label_filenames[split_3:split_4]

        # Step 4 : Copy the split contents to folders
        #Parallel Process - Images and Labels
        pool = Pool(self.cores)
        pool.map(self.copy_split_files_to_dataset, [train_filenames_images,
                                                    test_filenames_images,
                                                    val_filenames_images,
                                                    train_filenames_labels,
                                                    test_filenames_labels,
                                                    val_filenames_labels],
                 [dst_directory, dst_directory, dst_directory, dst_directory,
                  dst_directory, dst_directory],
                 [os.path.join('train', 'images'), os.path.join('test', 'images'),
                  os.path.join('val', 'images'), os.path.join('train', 'labels'),
                  os.path.join('test', 'labels'), os.path.join('val', 'labels')])
        pool.close()
        pool.clear()

        print("\ntrain images :", len(train_filenames_images))
        print("test images : ", len(test_filenames_images))
        print("val images : ", len(val_filenames_images))

        print("\ntrain labels :", len(train_filenames_labels))
        print("test labels : ", len(test_filenames_labels))
        print("val labels : ", len(val_filenames_labels))
    
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
      
        pdb.set_trace()
        image_filenames.sort()
        label_filenames.sort()

        # split the images into train test and val
        self.check_distribution("original_data", image_filenames, label_filenames)
        self.split_dataset(image_filenames, label_filenames, self.dataset_base_dir)

        # Step 3 Data Augmentation Block
        if self.data_aug:
            rand_image_filenames, rand_label_filenames = self.get_randimages_dataug(self.aug_value, 
                                                                                    image_filenames,
                                                                                    label_filenames)
            augmented_image_files = self.upsample_dataset(rand_image_filenames, "augmented_images", 2)
            augmented_label_files = self.upsample_dataset(rand_label_filenames, "augmented_labels", 2)
            
            
            self.check_distribution("augmented_data", augmented_image_files, augmented_label_files)
            print("\n Total Data :", len(augmented_image_files) + len(image_filenames))
        
            self.split_dataset(augmented_image_files, augmented_label_files, self.dataset_base_dir)
        
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

        self.generate_split_dirs()
        self.preprocess_dataset(image_filenames, label_filenames)

    
if __name__ == "__main__":
    split_data = SplitData("/home/vivek/Datasets/AmyB/amyb_wsi/")
    split_data.prepare_dataset()
       
    
    
