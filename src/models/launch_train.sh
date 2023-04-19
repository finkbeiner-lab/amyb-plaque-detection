#! /bin/bash
# Ge amd split
python train.py '/mnt/new-nas/work/data/npsad_data/vivek/''/mnt/new-nas/work/data/npsad_data/vivek/Datasets/amyb_wsi/train' '/mnt/new-nas/work/data/npsad_data/vivek/Datasets/amyb_wsi/test'

# K fold
python generate data -flag - 1/0
gen_kfold.py