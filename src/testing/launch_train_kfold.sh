#! /bin/bash

python train_kfold.py --base_dir '/gladstone/finkbeiner/steve/work/data/npsad_data/monika/' --dataset_train_location '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/amyb_wsi/train/train1/' --dataset_test_location '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/amyb_wsi/val/val1/' --k_list '3 5 7 9' --repeat 5