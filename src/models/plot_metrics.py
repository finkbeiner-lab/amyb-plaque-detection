"""
This code is used to generate performance metrics such as confusion matrix, ROC curve, PR curve of training dataset
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
import glob
import sys
sys.path.append(os.path.join(os.getcwd(), *tuple(['..'])))
import argparse

from typing import Callable, Dict, List, Optional, Set
from collections import OrderedDict
import pdb
import numpy as np
import torch
from models_pytorch_lightning.model_mrcnn_config import _default_mrcnn_config, build_default
from features import build_features
from models_pytorch_lightning.generalized_mask_rcnn_pl import LitMaskRCNN
from utils.helper_functions import evaluate_metrics, get_outputs, compute_iou, evaluate_mask_rcnn
from features import transforms as T
import torchvision
import pandas as pd
from lightning.pytorch.callbacks import ModelCheckpoint
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import pdb

def plot_roc_curve(all_df, save_path):
    """
    Plots and saves the Receiver Operating Characteristic (ROC) curve for multiple classes.

    Parameters:
    -----------
    all_df : pandas.DataFrame
        A dataframe containing true labels for each class and the predicted scores. 
        It must contain one column per class with true binary labels (0 or 1) and a 
        'pred_score' column with prediction probabilities or scores.
    
    save_path : str
        The path where the interactive ROC plot (as an HTML file) will be saved.

    Notes:
    ------
    This function assumes the presence of global variables:
    - class_names: list of class names corresponding to columns in `all_df`
    - colors: dictionary mapping class names to colors for the plot
    """
    
    # Initialize plotly figure
    fig = go.Figure()

    # Add a diagonal reference line representing random classifier performance
    fig.add_shape(
        type='line',
        line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    
    # Plot ROC curve for each class
    for class_name in class_names:
        # Compute False Positive Rate (FPR) and True Positive Rate (TPR)
        fpr, tpr, _ = roc_curve(all_df[class_name], all_df["pred_score"])
        # Add ROC curve trace to the figure
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=class_name,
            mode='lines',
            line=dict(color=colors[class_name], width=2)
        ))
    
    # Update axis labels, title, and layout properties
    fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=700,
        height=700,
        plot_bgcolor='white',
        title="Receiver Operating Characteristic Curve"
    )

    # Improve axis visibility with mirrored lines and light grid
    fig.update_xaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    fig.update_yaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')

    # Save the plot as an interactive HTML file
    fig.write_html(save_path)



def plot_pr_curve(all_df, save_path):
    """
    Plots and saves the Precision-Recall (PR) curve for multiple classes.

    Parameters:
    -----------
    all_df : pandas.DataFrame
        A dataframe containing true labels for each class and the predicted scores.
        It must contain one column per class with true binary labels (0 or 1), and a 
        'pred_score' column with prediction probabilities or scores.
    
    save_path : str
        The path where the interactive PR plot (as an HTML file) will be saved.

    Notes:
    ------
    This function assumes the presence of global variables:
    - class_names: list of class names corresponding to columns in `all_df`
    - colors: dictionary mapping class names to colors for the plot
    """
    
    # Initialize an empty Plotly figure
    fig = go.Figure()

    # Plot PR curve for each class
    for class_name in class_names:
        # Compute Precision and Recall values
        pr, rc, _ = precision_recall_curve(all_df[class_name], all_df["pred_score"])

        # Optional: Compute the proportion of positive samples (for reference)
        rp = (all_df[class_name]).sum() / len(all_df)

        # Add PR curve trace to the figure
        fig.add_trace(go.Scatter(
            x=rc, y=pr,
            name=class_name,
            mode='lines',
            line=dict(color=colors[class_name], width=2)
        ))
    
    # Update axis labels, title, and layout properties
    fig.update_layout(
        xaxis_title='Recall',
        yaxis_title='Precision',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=700,
        height=700,
        plot_bgcolor='white',
        title="Precision-Recall Curve"
    )

    # Improve axis visibility with mirrored lines and light grid
    fig.update_xaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')
    fig.update_yaxes(mirror=True, ticks='outside', showline=True, linecolor='black', gridcolor='lightgrey')

    # Save the plot as an interactive HTML file
    fig.write_html(save_path)




def plot_confusion_matrix(conf_mat, save_path):
    """
    Plots and saves a confusion matrix using matplotlib with custom styling.

    Parameters:
    -----------
    conf_mat : numpy.ndarray
        A 2D array representing the confusion matrix, typically of shape (n_classes, n_classes).
    
    save_path : str
        The path where the plotted confusion matrix figure (as a PDF file) will be saved.

    Notes:
    ------
    This function assumes the presence of a global variable:
    - class_names: a list of class names for axis labeling.

    The function displays the confusion matrix with formatted values, and uses a blue color map by default.
    """
    
    # Set the title font size and update global font size for matplotlib
    title_size = 16
    plt.rcParams.update({'font.size': 16})

    # Labels to be used for the x and y axes
    display_labels = class_names

    # Display options for the confusion matrix
    colorbar = False  # Whether to show the colorbar alongside the plot
    cmap = "Blues"    # Colormap for the confusion matrix. Alternatives: "Greens", "Oranges", etc.
    values_format = ".3f"  # Format for displaying cell values (up to 3 decimal places)

    # Create a matplotlib figure and axes
    f, ax = plt.subplots(1, 1, figsize=(10, 16))

    # Plot the confusion matrix with values
    ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=display_labels).plot(
        include_values=True,
        cmap=cmap,
        ax=ax,
        colorbar=colorbar,
        values_format=values_format
    )

    # Optional: Clean up or customize axis ticks and labels (currently commented out)
    # ax.xaxis.set_ticklabels(['', '', '', ''])
    # ax.set_xlabel('')
    # ax.tick_params(axis='x', which='both')

    # Set a suptitle for the entire figure
    f.suptitle("Multiple Confusion Matrices", size=title_size, y=0.93)

    # Display the plot in the notebook or script output
    plt.show()

    # Save the figure to the specified path in high quality
    f.savefig(save_path, bbox_inches='tight')



if __name__ == "__main__":
    
    OUTPUT_DIR = "/home/mahirwar/Desktop/Monika/npsad_data/vivek/reports/test-metrics"
    model_name= "/home/mahirwar/Desktop/Monika/npsad_data/vivek/runpod_mrcnn_models/yp2mf3i8_epoch=108-step=872.ckpt"
    dataset_test_location = '/home/mahirwar/Desktop/Monika/npsad_data/vivek/Datasets/amyb_wsi_v2/test
    colors={"Cored":"royalblue", "Diffuse":"firebrick","Coarse-Grained":"orange","CAA":"green"}
    class_names = ["Cored","Diffuse","Coarse-Grained","CAA"]

    # Load model
    test_config = dict(
        batch_size = 1,
        num_classes=4,
        device_id =0
    )
    optim_config = dict(
        cls=torch.optim.Adam,
        defaults=dict(lr=0.00001,weight_decay=1e-6) 
    )
    model = LitMaskRCNN.load_from_checkpoint(model_name)
        device = torch.device('cuda', test_config['device_id'])
    model = model.to(device)
    model.eval()


    #initialize lists and paths
    f1_list =[]
    label_matched_list =[]
    actual_labels_list = []
    pred_labels_list = []
    score_list = []
    output_path = os.path.join(OUTPUT_DIR,model_name.split("/")[-1])
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    collate_fn=lambda x: tuple(zip(*x))
        #exp_name = run.name

    # generate predictions for test data
    test_folders = glob.glob(os.path.join(dataset_test_location, "*"))
    print("test_folders",test_folders)
    df = pd.DataFrame()
    for test_folder in test_folders:
        test_dataset = build_features.AmyBDataset(os.path.join(dataset_test_location, test_folder), T.Compose([T.ToTensor()]))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_config['batch_size'], shuffle=False, num_workers=4, collate_fn=collate_fn)
        for i, (images, targets) in enumerate(test_loader):
            images = [image.to(device) for image in images]
            targets = [dict([(k, v.to(device)) for k, v in target.items()]) for target in targets]
            outputs = model.forward(images, targets)
            masks, labels, scores ,_ = get_outputs(outputs, 0.5)
            f1_mean, labels_matched,actual_labels,pred_labels, scores =  evaluate_metrics(targets, masks, labels,scores,0.5 )
            f1_list.extend(f1_mean)
            label_matched_list.extend(labels_matched)
            actual_labels_list.extend(actual_labels)
            pred_labels_list.extend(pred_labels)
            score_list.extend(scores)

    all_df = pd.DataFrame({"f1_score":f1_list, "matched_labels":label_matched_list, "actual_labels":actual_labels_list, "pred_labels":pred_labels_list,
    "pred_score":score_list})

    print(np.mean(all_df["f1_score"].values), np.mean(all_df["matched_labels"].values))

    print(np.unique(all_df["actual_labels"].values))
    print(np.unique(all_df["pred_labels"].values))

    all_df["Cored"] = np.where(all_df["actual_labels"]==1, 1, 0)
    all_df["Diffuse"] = np.where(all_df["actual_labels"]==2, 1, 0)
    all_df["Coarse-Grained"] = np.where(all_df["actual_labels"]==3, 1, 0)
    all_df["CAA"] = np.where(all_df["actual_labels"]==4, 1, 0)


    for test_folder in test_folders:
        test_dataset = build_features.AmyBDataset(os.path.join(dataset_test_location, test_folder), T.Compose([T.ToTensor()]))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_config['batch_size'], shuffle=False, num_workers=4, collate_fn=collate_fn)
        for threshold in [0.25,0.5,0.75,0.9,0.95]:
            for i, (images, targets) in enumerate(test_loader):
                images = [image.to(device) for image in images]
                targets = [dict([(k, v.to(device)) for k, v in target.items()]) for target in targets]
                outputs = model.forward(images, targets)
                #print(outputs)
                masks, labels, scores = get_outputs(outputs, threshold)
                f1_mean, labels_matched,actual_labels,pred_labels, scores =  evaluate_metrics(targets, masks, labels,scores )
                print(f1_mean, labels_matched)
                f1_list.extend(f1_mean)
                label_matched_list.extend(labels_matched)
                actual_labels_list.extend(actual_labels)
                pred_labels_list.extend(pred_labels)
                score_list.extend(scores)
            precision, recall, fscore, support = precision_recall_fscore_support(actual_labels_list,pred_labels_list)
            temp = pd.DataFrame({"class":["Cored", "Diffuse","Coarse-Grained","CAA"], "precision":list(precision), "recall":list(recall), "fscore":fscore, "support":support} )
            temp["threshold"]= threshold
            if len(df)==0:
                df = temp
            else:
                df = pd.concat([df,temp], ignore_index=True)       
    df.to_csv("/home/mahirwar/Desktop/Monika/npsad_data/vivek/reports/test-metrics"+model_name.split("/")[-3]+"_eval_metrics.csv")   
   
    
    ## To plot ROC curve
    roc_save_path = os.path.join(output_path,"ROC_Curve.html" )
    plot_roc_curve(all_df, roc_save_path)

    ## To plot PR curve
    pr_save_path = os.path.join(output_path,"PR_Curve.html" )
    plot_pr_curve(all_df,pr_save_path)
    
    
    ## To plot confusion matrix 
    conf_mat_save_path = os.path.join(output_path,"conf_mat.png" )
    conf_mat = confusion_matrix(all_df["actual_labels"], all_df["pred_labels"])
    print(conf_mat)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat,
                                display_labels=class_names)
    disp.plot()
    plot_confusion_matrix(conf_mat, conf_mat_save_path)
    pd.DataFrame(conf_mat ).to_csv((os.path.join(output_path,"confusion_matrix.csv")))

    # used for testing
    #precision, recall, fscore, support = precision_recall_fscore_support(all_df["actual_labels"], all_df["pred_labels"])
    #temp = pd.DataFrame({"class":["Cored", "Diffuse","Coarse-Grained","CAA"], "precision":list(precision), "recall":list(recall), "fscore":fscore, "support":support} )
    #temp.to_csv(os.path.join(output_path,"prec_recall.csv"))