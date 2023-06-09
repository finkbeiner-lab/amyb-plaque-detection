# AmyB-plaque-detection

==============================

  

### Deep learning model to detect and quantify plaques from histopathology Images

 


## Software Dependencies
1.  Qupath >= 0.3.0

## Workflow Instructions


### Qupath Project Setup
Each WSI (Whole Slide Image) is opened as a project in Qupath. All the annotations are done for the specific WSI within the project.  All the projects are stored in /home/user/Qupath/bin/. For a WSI XE10-045_1_AmyB_1.mrxs , you should create a project named XE10-045_1_AmyB_1 in Qupath

### Qupath Annotations
ToDO: Gennadi add instructions to use qupath annotations

### Convert Qupath Annotations to Json
The annotations from Qupath are conveted to custome json format which is used downstream to generate training and test data. For the structure please refer the sample json file
