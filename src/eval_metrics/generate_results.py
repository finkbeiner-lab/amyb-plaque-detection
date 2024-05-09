import os
import sys
import sys
sys.path.insert(0, '../')
from visualization import explain
from models.model_mrcnn import _default_mrcnn_config, build_default
from PIL import Image 
from glob import glob
import random
import wandb
random.seed(0)


input_test_files = "/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/amyb_wsi/test-patients/images"
input_test_files_list = glob(os.path.join(input_test_files,"*"))

samples = random.sample(input_test_files_list, 1)

model_input_path = "/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/models/eager-frog-489_mrcnn_model_100.pth"

test_config = dict(
    batch_size = 1,
    num_classes = 3
)

model_config = _default_mrcnn_config(num_classes=1 + test_config['num_classes']).config
model = build_default(model_config, im_size=1024)

# Use the Run ID from train_model.py here if you want to add some visualizations after training has been done
# with wandb.init(project="nps-ad", id = "17vl5roa", entity="hellovivek", resume="allow"):\
    

#run = wandb.init(project="nps-ad-vivek",  entity="hellovivek")
run = ""
explain_object = explain.ExplainPredictions(model, model_input_path = model_input_path, test_input_path=samples, 
                                detection_threshold=0.2, wandb=run, save_result=True, ablation_cam=False, save_thresholds=True)
explain_object.generate_results()
