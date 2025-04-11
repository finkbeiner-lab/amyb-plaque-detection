# AmyB-plaque-detection

==============================

  

### Generalizable Prediction of Alzheimer Disease Pathologies with a Scalable Annotation Tool and an High-Accuracy Model


## Software Dependencies
1.  Qupath = 0.5.0

## Workflow Instructions


### Qupath Project Setup
Each WSI (Whole Slide Image) is opened as a project in Qupath. All the annotations are done for the specific WSI within the project.  All the projects are stored in /home/user/Qupath/bin/. For a WSI XE10-045_1_AmyB_1.mrxs , you should create a project named XE10-045_1_AmyB_1 in Qupath

### Qupath Annotations
Install our Qupath ./src/annotationTool for annotation


### Convert Qupath Annotations to Json
The annotations from Qupath are conveted to custom json format which is used downstream to generate training data. 
Use ./src/qupath_scripts/export_annotation_training_scripts 

### Generate crops and masks for training 
The annotation json are processed to save crops and masks using ./src/data/training_data/generate_data_from_annotations.py, and further spliting 
and augmentation is performed with ./src/data/training_data/Data_split_and_augmentation.ipynb. The mean and std is computed over train data using 
./src/data/training_data/compute_mean_std_amyb.ipynb

### Train model
Mask RCNN Model is trained using pytorch lightning - ./src/models. The model learns both plaque object and it's mask

### Generating inference on unseen whole slide images
The trained Mask RCNN model is made to run on unseen slides by iterating through crops. Thus, prediction on a slide contains detected plaque object positions, masks and labels. Further, features are extracted from these detected plaques. 
1. Internal Dataset : The inference is generated for our dataset with these codes - ./src/inference/internal_dataset. The slides are preprocessed to subtract background and then crops containing tissue area are run through the model.
2. External Dataset: Used two external datasets and their ground truths for measuring model's perfomance ./src/inference/external_dataset

### Clinical Correlation Analysis
The inference features such as count of each type of plaques, size, area generated for internal dataset is then combined with the clinical metadata. 
Analysis is performed to find the correlation - .src/clinical_correlation_analysis

### Interrater Study
Prepare interrater annotation jsons following these scripts - ./src/qupath_scripts/interrater_study_scripts
Compute pairwise Cohen's kappa score between raters - ./src/interrater_study/pairwise_kappa_NPs.ipynb
Measure performance of interrater study with model predictions - ./src/interrater_study/Performance_metric_NPs_consensus.ipynb