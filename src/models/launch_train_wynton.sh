#! /usr/bin/env bash
#$ -S /bin/bash  # run job as a Bash shell [IMPORTANT]
#$ -cwd          # run job in the current working directory

## GPU Monitoring
gpuprof=$(dcgmi group -c mygpus -a $SGE_GPU | awk '{print $10}')
dcgmi stats -g $gpuprof -e
dcgmi stats -g $gpuprof -s $JOB_ID

export WANDB_API_KEY=1538da3c15965f957d5e5aa41bbbd1b7cb520768
wandb login $WANDB_API_KEY
wandb offline
#module load CBI scl-devtoolset/6
module load CBI miniconda3-py39
conda activate amyb
python train.py /gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/amyb_wsi /gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/amyb_wsi/train-short/ /gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/amyb_wsi/train-short/

echo "The model is running on node $HOSTNAME"

#wandb sync /wynton/home/finkbeiner/vgramas/Projects/amyb-plaque-detection/src/wandb_offline_logs

## End Script for GPU Monitoring
dcgmi stats -g $gpuprof -x $JOB_ID
dcgmi stats -g $gpuprof -v -j $JOB_ID
dcgmi group -d $gpuprof
