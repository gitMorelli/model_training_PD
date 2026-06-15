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
# 4. Compute and display metrics using scikit-learn
from sklearn.metrics import classification_report, confusion_matrix

from src.utils.data_loading_utils import MultiTarSequenceDataset, InMemoryWdsDataset, melt_df
from src.utils.data_loading_utils import prepare_handedness_dataset, prepare_handedness_dataset_all, generate_exclusion_set_val
from src.utils.model_utils import SimpleMockModel, CustomBinaryCNN, CustomMLP, TiledJoinedModels
from src.utils.model_utils import get_model, test_output, get_classification_head, JoinedModels, unfreeze_layers
from src.utils.visualization import debug_images_dataset
from src.utils.image_processing import ResizeLongestSide
from src.utils.training_utils import LitModel

#PATHS
SOURCE_PATH = "/mnt/beegfs02/scratch/a_morelli/model_training/handedness/"
LIST_OF_IDS_HANDEDNESS_PATH = "/home/a_morelli/datasets/id_lists/handedness_model_ids_all_qs.csv"
data_folder = "all_no_grids_png_whitebg" 
#data_folder = "all_full_sentences_png_whitebg" 
SOURCE_PATTERN = os.path.join(SOURCE_PATH,data_folder)
SHARD_PATTERN_val = os.path.join(SOURCE_PATTERN,"val/worker*_shard-*.tar")
SHARD_PATTERN_train = os.path.join(SOURCE_PATTERN,"train/worker*_shard-*.tar")

#QUESTIONNAIRES_TO_INCLUDE_HANDEDNESS = ['5']
QUESTIONNAIRES_TO_INCLUDE_HANDEDNESS = [str(q) for q in range(1,14)]

#Model definition
MODEL = 'resnet50' #'swin_s' #'resnet18', 'custom_cnn', 'resnet34_layer1','resnet34_layer2','resnet34_layer3', 'resnet34', 'resnet50'
#clip-vit-large-patch14, clip-vit-large-patch14-inter
huggingface_transform=True if MODEL in ['clip-vit-large-patch14-un', 'clip-vit-large-patch14-inter'] else False
transform_override = True #if true overrides the transform defined for the model with ta custom one
CLASSIFICATION_HEAD = 'linear' #'MLPClassifier1'#'MLPClassifier1' # 'linear', 'regularized_linear', 'MLPClassifier1'
PARAMS = {
    'dropout': 0.2,
    'hidden_sizes': [32],
    'with_input_norm': 'batch_norm'
}
batch_size = 32
input_size = 224

EXPERIMENT_NAME = f"{MODEL}_{data_folder}"
RESULTS_PATH = os.path.join(SOURCE_PATH,f"{MODEL}_model_results")
CHECKPOINT_PATH = os.path.join(RESULTS_PATH, "checkpoints")
checkpoint_to_load='v_10/best-epoch=55-val_loss=0.91.ckpt'#best-epoch=55-val_loss=0.91.ckpt'#best.ckpt , None last.ckpt
DEBUG_IMGS = False
SEED=42
DATA_MODALITY = 'all' # 'X', 'text', 'digit', 'all' (all returns 3x3x224x224 elements instead of 3x224x224)
NUM_tiles = 1

BALANCED_DATA = True
USE_BALANCED_WEIGHTS = False
BALANCING_FACTOR = 1
MAJORITY_CLASS_ID = 0
THRESHOLD_NUM = 1
PREDICT_ON_TRAIN = True

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
    all_subjects = [sid for batch in outputs for sid in batch["subject_id"]]
    all_questionnaires = [q for batch in outputs for q in batch["questionnaire"]]

    #create a dataframe with the subject id, questionnaire, true label, predicted label and probabilities
    results_df = pd.DataFrame({
        "subject_id": all_subjects,
        "questionnaire": all_questionnaires,
        "true_label": all_labels.numpy(),
        "predicted_label": all_preds.numpy(),
        "probability_0": all_probs[:, 0].numpy(),
        "probability_1": all_probs[:, 1].numpy() 
    })
    return results_df, all_probs, all_preds, all_labels

custom_transform = T.Compose(
        [
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.06040578708052635, 0.06040578708052635, 0.06040578708052635], 
                        std=[0.23823712766170502, 0.23823712766170502, 0.23823712766170502]),
        ]
    )
AUGMENTATION_TRANSFORM = T.Compose([
                #ResizeLongestSide(448),
                T.RandomCrop(
                    112, 
                    pad_if_needed=True, 
                    padding_mode='constant', 
                    fill=(255,255,255) # <-- White fill for RGB PIL images
                )
            ])



def main():
    args = get_args()
    worker = args.num_workers
    prefetch_factor = 2
    decode_approach = 'pil'
    load_in_memory = False
    split_workers = True
    transform = None
    num_classes=1 #1 for BCE loss, 2 for crossentropy
    exclusion_set = set() # you can add here the ids you want to exclude from the dataset (for example because they are corrupted or for debugging purposes)
    val_exclusion_set = set()
    apply_augmentation = True
    invert_color=True

    #fix all the seeds for reproducibility 
    torch.manual_seed(SEED)
    random.seed(SEED)
    #with lightning 
    L.seed_everything(SEED, workers=True)

    if NUM_tiles > 1 and DATA_MODALITY == 'all':
        print("Warning: Data modality = 'all' and NUM_tiles>1 are incompatible ")
        return 


    if apply_augmentation:
        #add a random crop transform without resizing
        augmentation_transform = AUGMENTATION_TRANSFORM
    else:
        augmentation_transform = None

    if DATA_MODALITY == 'all':
        selection_modality = 'text' 
    else:
        selection_modality = DATA_MODALITY 
    csv_data = pd.read_csv(LIST_OF_IDS_HANDEDNESS_PATH)
    print(csv_data.head(2))
    
    csv_data, num_less_than_1_rows = melt_df(csv_data, modality=selection_modality, threshold=THRESHOLD_NUM)
    print(csv_data.head(2))
    print('Debug less than 1: ',len(num_less_than_1_rows))
    
    val_exclusion_set.update(num_less_than_1_rows)
    exclusion_set.update(num_less_than_1_rows)

    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu = torch.cuda.get_device_name(device_id)
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')
        print(f"--- GPU DIAGNOSTICS ---")
        print(f"Active GPU: {gpu}")
        print(f"CUDA_VISIBLE_DEVICES: {visible}")


    backbone,transform = get_model(name=MODEL, pretrained=True)
    if transform_override:
        transform = custom_transform
    #check on which device the backbone is
    #print(f"Backbone device: {next(backbone.parameters()).device}")
    out=test_output(input_size, backbone)
    #print(out)
    in_features = out.shape[1]
    in_features*= 3 if DATA_MODALITY=='all' else 1
    in_features*= NUM_tiles if NUM_tiles > 1 else 1
    classificaton_head = get_classification_head(name=CLASSIFICATION_HEAD,in_features=in_features,num_classes=num_classes,**PARAMS)
    if DATA_MODALITY == 'all' or NUM_tiles>1:
        model = TiledJoinedModels(backbone, classificaton_head) #it gets the dimension by itself
    else:
        model = JoinedModels(backbone, classificaton_head)

    
    #print(f"Number of dataset samples (train): {len(train_dataset)}")
    if BALANCED_DATA:
        val_exclusion_set.update( generate_exclusion_set_val(csv_data, data_modality=selection_modality,
                                                    majority_class_id=MAJORITY_CLASS_ID, balancing_factor=BALANCING_FACTOR, 
                                                    label_col='lateralite', id_col='ident_projet', split='val') )
        if PREDICT_ON_TRAIN:
            exclusion_set.update( generate_exclusion_set_val(csv_data, data_modality=selection_modality,
                                                        majority_class_id=MAJORITY_CLASS_ID, balancing_factor=BALANCING_FACTOR, 
                                                        label_col='lateralite', id_col='ident_projet', split='train') )
    unique_ids_in_val = csv_data[csv_data['split'] == 'val']['ident_projet'].unique()
    unique_ids_in_train = csv_data[csv_data['split'] == 'train']['ident_projet'].unique()
    print(f"Debug -> exclusion set length (val): {len(val_exclusion_set)}; Remaining samples in val dataset: {len(unique_ids_in_val) - len(val_exclusion_set)}")
    print(f"Debug -> exclusion set length (train): {len(exclusion_set)}; Remaining samples in train dataset: {len(unique_ids_in_train) - len(exclusion_set)}")
    return 
    val_dataset = prepare_handedness_dataset_all(SHARD_PATTERN_val, decode_approach=decode_approach, load_in_memory=load_in_memory,
                                                    split_workers=split_workers, batch_size=batch_size,
                                                    transform=transform, modality=DATA_MODALITY, exclusion_set=val_exclusion_set, 
                                                    huggingface_transform=huggingface_transform, augmentation_transform=augmentation_transform,
                                                    invert_color=invert_color, n_views=NUM_tiles)
    if PREDICT_ON_TRAIN:
        train_dataset = prepare_handedness_dataset_all(SHARD_PATTERN_train, decode_approach=decode_approach, load_in_memory=load_in_memory,
                                                        split_workers=split_workers, batch_size=batch_size,
                                                        transform=transform, modality=DATA_MODALITY, exclusion_set=exclusion_set, 
                                                        huggingface_transform=huggingface_transform, augmentation_transform=augmentation_transform,
                                                        invert_color=invert_color, n_views=NUM_tiles)
    
    
    if DEBUG_IMGS:
        # Iterate through the first few items to see what labels are coming out
        for i, (img, label, id,q,mode) in enumerate(val_dataset):
            print(f"Sample {i}: Label {label}, ID {id}, Q {q}, Mode {mode}")
            if i > 10: break
        #sample N data points at random from the train dataset, save them in an image with the corresponding label
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        n_stacked=NUM_tiles
        if DATA_MODALITY == 'all':
            n_stacked = 3
        debug_images_dataset(val_dataset, output_path="data/anteprima_dataset.png", num_immagini=16, mean=None, std=None, n_stacked=n_stacked)
    
    val_loader = DataLoader(
        val_dataset, 
        num_workers=worker, 
        batch_size=None, 
        prefetch_factor=prefetch_factor, # Tells workers to queue up batches in advance (set to none if 0 workers)
        pin_memory=True
    )

    if PREDICT_ON_TRAIN:
        train_loader = DataLoader(
            train_dataset, 
            num_workers=worker, 
            batch_size=None, 
            prefetch_factor=prefetch_factor, # Tells workers to queue up batches in advance (set to none if 0 workers)
            pin_memory=True
        )

    if DATA_MODALITY == 'all':
        example_input_array = torch.randn(1, 3,3, 224, 224)  # For visualizing the graph in TensorBoard
    elif NUM_tiles > 1:
        example_input_array = torch.randn(1, NUM_tiles, 3, 224, 224)
    else:
        example_input_array = torch.randn(1, 3, 224, 224)
    lit_model = LitModel(write_log=None,model=model, num_0=1, num_1=1, num_classes=num_classes, example_input_array=example_input_array)

    tb_logger=False

    # 4. Initialize Trainer and Fit
    trainer = L.Trainer(
        max_epochs=1,
        logger = tb_logger,
        accelerator="auto"                # Automatically selects GPU/CPU/MPu
    )
    
    print("\n--- Starting Validation Evaluation ---")

    # 1. Gather predictions using the best checkpoint saved during training
    # Setting ckpt_path="best" tells Lightning to automatically find your top model
    ckpt_path=os.path.join(CHECKPOINT_PATH,checkpoint_to_load) 
    # 3. Load the checkpoint file (weights are skipped, only reading metadata)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    print(f"Checkpoint loaded from: {ckpt_path}")
    # 4. Extract the exact epoch
    best_epoch = checkpoint["epoch"]
    print(f"The best model was saved at epoch: {best_epoch}")
    outputs = trainer.predict(lit_model, dataloaders=val_loader,ckpt_path = ckpt_path)# ckpt_path=os.path.join(CHECKPOINT_PATH,"best.ckpt"))
    results_df, all_probs, all_preds, all_labels = get_result_df(outputs)

    #show 10 example outputs and the corresponding expected label
    print("\n--- Sample Predictions ---")
    for i in range(10):
        print(f"Probs: {all_probs[i].tolist()} | Predicted: {all_preds[i].item()} | True Label: {all_labels[i].item()}")
    
    # 3. Convert to numpy arrays for statistics calculation
    y_pred = all_preds.numpy()
    y_true = all_labels.numpy()
    
    print("\n================ VALIDATION STATISTICS ================")
    print("\n--- Classification Report ---")
    # Adjust target_names to match your two classes if needed
    print(classification_report(y_true, y_pred, target_names=["Right", "Left"]))
    
    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(y_true, y_pred))
    print("=======================================================")

    #Make a classification report for each of the questionnaires included in the dataset
    print("\n--- Classification Report by Questionnaire ---")
    for q in sorted(results_df['questionnaire'].unique()):
        subset = results_df[results_df['questionnaire'] == q]
        y_pred_q = subset['predicted_label'].values
        y_true_q = subset['true_label'].values
        print(f"\nQuestionnaire {q}:")
        print(classification_report(y_true_q, y_pred_q, target_names=["Right", "Left"]))
    

    # create a new column subject_id_questionnaire by concatenating subject_id and questionnaire
    q = results_df['questionnaire'].astype(str).str.removeprefix('q')
    results_df['ident_projet'] = results_df['subject_id'].astype(str) + '_' + q
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(results_df.head(10))

    if PREDICT_ON_TRAIN:
        outputs = trainer.predict(lit_model, dataloaders=train_loader,ckpt_path = ckpt_path)# ckpt_path=os.path.join(CHECKPOINT_PATH,"best.ckpt"))
        results_df_train, all_probs, all_preds, all_labels = get_result_df(outputs)
        q = results_df_train['questionnaire'].astype(str).str.removeprefix('q')
        results_df_train['ident_projet'] = results_df_train['subject_id'].astype(str) + '_' + q

        #concatenate the result dataframes
        results_complete = pd.concat([results_df, results_df_train], ignore_index=True)
        results_df = results_complete.copy()
    merged_df = pd.merge(csv_data, results_df, on='ident_projet', how='left')
    #check for duplicate rows
    if merged_df.duplicated(subset=['ident_projet']).any():
        print("Warning: There are duplicate rows in the merged dataframe based on 'ident_projet'.")
    else:
        print("No duplicate rows found in the merged dataframe based on 'ident_projet'.")
    #save the merged dataframe in a csv file
    merged_df.to_csv(os.path.join(os.path.dirname(ckpt_path), f"predictions.csv"), index=False)

if __name__ == "__main__":
    main()
    