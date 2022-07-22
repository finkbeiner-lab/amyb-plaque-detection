import sys
sys.path.append('../')
import pyfiglet
import argparse
from visualization.explain import ExplainPredictions
import numpy as np
import torch
import wandb
import pdb


if __name__ == "__main__":

    result = pyfiglet.figlet_format("Test Model", font="slant")
    parser = argparse.ArgumentParser(description="Model testing")

    parser.add_argument('trained_model_path', help='Enter the path to the  trained model resides')
    parser.add_argument('test_WSI_input', help='Enter the path to the test WSI images') 
    parser.add_argument('detection_threshold', help='Enter the detection threshold Ex- 0.75')   

 
    args = parser.parse_args()

    with wandb.init(project="nps-ad", entity="hellovivek"):
        explain = ExplainPredictions(model_input_path=args.trained_model_path , test_input_path=args.test_WSI_input, 
                                    detection_threshold=float(args.detection_threshold),  wandb=wandb, save_result=True ,ablation_cam=False, save_thresholds=False)
        explain.generate_results()


