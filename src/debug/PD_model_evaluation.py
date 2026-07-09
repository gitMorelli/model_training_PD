import tarfile
import time
import io
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import os
import pandas as pd
import torch.nn as nn
import time
import webdataset as wds
import glob
from tqdm import tqdm
import torch.optim as optim
from torchvision import models
import torchvision.utils as vutils
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics.classification import MulticlassRecall
import random
import shutil
#from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import Callback
import re
import signal
import sys
import json
# 4. Compute and display metrics using scikit-learn
from sklearn.metrics import classification_report, confusion_matrix

from src.utils.data_loading_utils import melt_df, prepare_exclusion_sets_PD, load_grid_dict, prepare_loaders_PD
from src.utils.data_loading_utils import prepare_handedness_dataset, prepare_handedness_dataset_all, generate_exclusion_set_val
from src.utils.model_utils import SimpleMockModel, CustomBinaryCNN, CustomMLP, TiledJoinedModels
from src.utils.model_utils import get_model, test_output, get_classification_head, JoinedModels, unfreeze_layers
from src.utils.visualization import debug_images_dataset
from src.utils.image_processing import ResizeLongestSide, get_augmentation_transform, get_transforms
from src.utils.training_utils import ModelPD

SOURCE_PATH = "/mnt/beegfs02/scratch/a_morelli/model_training/PD/"
exp_params = {
    'list_of_ids_paths': "/home/a_morelli/datasets/id_lists/final_data_for_training.parquet",
    'data_folder': "final_png_whitebg",
    'grid_dict_path': "/mnt/beegfs02/scratch/a_morelli/datasets/PD_data_h5.pkl",
    'predict_on_train': False, #if True the model will be evaluated on the training set as well

    #experiment parameters
    'data_modality': ['X_crop']+['digit_full','digit_crop']+['digit' for _ in range(3)]+['text_full','text_crop']+ ['text' for _ in range(3)],# 'X', 'text', 'digit', 'all' (all returns 3x3x224x224 elements instead of 3x224x224)
    #or list e.g. ['digit_full','digit_crop','digit','digit','digit'] for 5 tiles
    'num_tiles': 3,
    'use_grid': True,
    'use_balanced_weights': True,
    'balancing_factor': 1, #even if float is converted to int with int(balancing_factor), balancing_factor controls for each case-control group are kept 
    'balanced_data': True, #note that this and balace_validation are independent
    'balance_validation': False, #if True the validation set is balanced, if False it is not balanced
    'majority_class_id': 0, 
    'threshold_num': 1,
    'num_classes': 1, #1 for BCE loss, 2 for crossentropy
    'filter_missing': 'last_q', #'all', 'last_q' #if all remove only ids with grid_pattern=0000..00 13 times, 
    #if 'last_q' with the first last_q equal to 0
    'censor_time': -1, #0, -1 (if keep all) or a positive value
    'filter_modality': 'digit', #None, 'digit', 'text', 'X' (if None keep all)

    #model definition
    'model':"resnet18", #'swin_s' #'resnet18', 'custom_cnn', 'resnet34_layer1','resnet34_layer2','resnet34_layer3', 'resnet34', 'resnet50'
#clip-vit-large-patch14, clip-vit-large-patch14-inter
    'custom_pre_trained_weights': os.path.join(
    '/mnt/beegfs02/scratch/a_morelli/model_training/pre_trained_models/mnist',
    'resnet18/checkpoints/best-resnet18-mnist-epoch=05-val_loss=0.0181.ckpt'
), #None, see options below
    'model_structure': 'SequenceQuestionnaireModel',

    #Transforms definitions
    'custom_transform': 'pad_resize_normalize', #None, #if not None overrides the transform defined for the model with ta custom one
    'norm_mu': [0.06040578708052635, 0.06040578708052635, 0.06040578708052635],
    'norm_std': [0.23823712766170502, 0.23823712766170502, 0.23823712766170502],
    'apply_augmentation': None, #None, 'random_crop_half' ; if data_modality is a list the transform for each view mode will be determined
    #in the code based on the view name
    'invert_color':True,
    
    #Training params definition
    'lr_backbone': 1e-4,
    'lr_classifier_head': 1e-3,
    'lr_scheduling': 'cosine', #'cosine' # 'cosine', 'step', None
    'batch_size': 4,
    'num_epochs': 100,
    'patience': 50,
    'eta_min_cosine': 1e-6,
    'weight_decay': 1e-2, #0.05 (swi) #1e-2 (resnet)
    'warmup_fraction': 0.05,   # ~5% of total steps as warmup
    'input_size': 224,
    'layers_to_unfreeze': ['all','classifier'], #Update it for every model
    'seed': 42,
}
if isinstance(exp_params['data_modality'],list):
    exp_params['num_tiles'] = len(exp_params['data_modality'])
huggingface_transform=True if exp_params['model'] in ['clip-vit-large-patch14-un', 'clip-vit-large-patch14-inter'] else False
exp_params['huggingface_transform'] = huggingface_transform

#PATHS
SOURCE_PATTERN = os.path.join(SOURCE_PATH,exp_params['data_folder'])
SHARD_PATTERN_val = os.path.join(SOURCE_PATTERN,"val/worker*_shard-*.tar")
SHARD_PATTERN_train = os.path.join(SOURCE_PATTERN,"train/worker*_shard-*.tar")
#Checkpoint paths
RESULTS_PATH = os.path.join(SOURCE_PATH,f"{exp_params['model']}_model_results")
CHECKPOINT_PATH = os.path.join(RESULTS_PATH, "checkpoints")
checkpoint_to_load='v_6/best-epoch=20-val_loss=0.64.ckpt'#best-epoch=55-val_loss=0.91.ckpt'#best.ckpt , None last.ckpt

VERBOSE = False
CLASS_COL = 'diag_park_final1_quest'


def main():
    args = get_args()
    worker = args.num_workers
    prefetch_factor = 2 if worker > 0 else None

    #fix all the seeds for reproducibility 
    torch.manual_seed(exp_params['seed'])
    random.seed(exp_params['seed'])
    #with lightning 
    L.seed_everything(exp_params['seed'], workers=True)

    #load grid_files for selecting chunks from the images during the dataloading
    grid_dict = load_grid_dict(exp_params)

    csv_data = pd.read_parquet(exp_params['list_of_ids_paths'])

    #exclude controls from the training if i want to reduce the asimmetry of the dataset (for example if i want to have a 1:1 ratio between cases and controls)
    exclusion_set, val_exclusion_set, num_0, num_1 = prepare_exclusion_sets_PD(exp_params,verbose=VERBOSE,class_col=CLASS_COL)

    _,transform = get_model(name=exp_params['model'], pretrained=True)
    transform = get_transforms(exp_params, transform)
    
    train_loader,val_loader,_,_= prepare_loaders_PD(worker,prefetch_factor,exp_params,exclusion_set,val_exclusion_set, grid_dict, transform, 
                                                    SHARD_PATTERN_train=SHARD_PATTERN_train, SHARD_PATTERN_val=SHARD_PATTERN_val)
    
    # 1. Gather predictions using the best checkpoint saved during training
    # Setting ckpt_path="best" tells Lightning to automatically find your top model
    ckpt_path=os.path.join(CHECKPOINT_PATH,checkpoint_to_load) 
    lit_model = ModelPD.load_from_checkpoint(ckpt_path, write_log=False)
    tb_logger=False
    # 4. Initialize Trainer and Fit
    trainer = L.Trainer(
        max_epochs=1,
        logger = tb_logger,
        accelerator="auto"                # Automatically selects GPU/CPU/MPu
    )

    outputs = trainer.predict(lit_model, dataloaders=val_loader)# ckpt_path=os.path.join(CHECKPOINT_PATH,"best.ckpt"))
    results_df, all_probs, all_preds, all_labels = get_result_df(outputs)
    
    print(f"Evaluating model on validation set using checkpoint: {ckpt_path}")
    analyze_results(all_preds, all_labels, results_df)
    

    if exp_params['predict_on_train']:
        outputs = trainer.predict(lit_model, dataloaders=train_loader)# ckpt_path=os.path.join(CHECKPOINT_PATH,"best.ckpt"))
        results_df_train, all_probs, all_preds, all_labels = get_result_df(outputs)
        analyze_results(all_preds, all_labels, results_df, split="train")

        #concatenate the result dataframes
        results_complete = pd.concat([results_df, results_df_train], ignore_index=True)
        results_df = results_complete.copy()
    
    store_results(csv_data, results_df, ckpt_path,exp_params)

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="I/O Benchmark for Multi-Tar Dataset")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for DataLoader")
    parser.add_argument("--batches_to_test", type=int, default=50, help="Number of batches to process for benchmark")
    return parser.parse_args()


def get_result_df(outputs):
    # 2. Concatenate all batch outputs into unified tensors
    all_probs = torch.cat([batch["probs"] for batch in outputs])
    all_preds = torch.cat([batch["preds"] for batch in outputs])
    all_labels = torch.cat([batch["labels"] for batch in outputs])
    all_subjects = [sid for batch in outputs for sid in batch["subject_ids"]]

    #create a dataframe with the subject id, questionnaire, true label, predicted label and probabilities
    results_df = pd.DataFrame({
        "subject_id": all_subjects,
        "true_label": all_labels.numpy(),
        "predicted_label": all_preds.numpy(),
        "probability_0": all_probs[:, 0].numpy(),
        "probability_1": all_probs[:, 1].numpy() 
    })
    return results_df, all_probs, all_preds, all_labels

def analyze_results(all_preds, all_labels, results_df,split="validation"):
    # 3. Convert to numpy arrays for statistics calculation
    y_pred = all_preds.numpy()
    y_true = all_labels.numpy()
    
    print(f"\n================ {split.upper()} STATISTICS ================")
    print("\n--- Classification Report ---")
    # Adjust target_names to match your two classes if needed
    print(classification_report(y_true, y_pred, target_names=["healthy", "PD"]))
    
    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_true, y_pred))
    print("=======================================================")    

    
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(results_df.head(10))

def store_results(csv_data, results_df, ckpt_path,params):
    merged_df = pd.merge(csv_data, results_df, on='ident_projet', how='left')
    #check for duplicate rows
    if merged_df.duplicated(subset=['ident_projet']).any():
        print("Warning: There are duplicate rows in the merged dataframe based on 'ident_projet'.")
    else:
        print("No duplicate rows found in the merged dataframe based on 'ident_projet'.")
    #save the merged dataframe in a csv file
    merged_df.to_csv(os.path.join(os.path.dirname(ckpt_path), f"predictions.csv"), index=False)
    #save params dict as predictions_metadata.json
    with open(os.path.join(os.path.dirname(ckpt_path), f"predictions_metadata.json"), 'w') as f:
        json.dump(params, f, indent=4)

if __name__ == "__main__":
    main()
    