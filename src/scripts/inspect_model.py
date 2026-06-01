import tarfile
import time
import io
import torch
from PIL import Image
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
import random
import shutil
#from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import Callback

from src.utils.data_loading_utils import MultiTarSequenceDataset, InMemoryWdsDataset
from src.utils.model_utils import SimpleMockModel, CustomBinaryCNN, CustomMLP
from src.utils.model_utils import get_model, test_output, get_classification_head, JoinedModels, unfreeze_layers

def main():
    model_name ='resnet18'
    classifier_name='linear'
    output_path=f"data/model_structures/{model_name}_structure.txt"
    if not os.path.exists(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    backbone,transform = get_model(name=model_name, pretrained=True)
    out=test_output(224, backbone)
    in_features = out.shape[1]
    classificaton_head = get_classification_head(name=classifier_name,in_features=in_features)
    model = JoinedModels(backbone, classificaton_head)
    # Scriviamo la struttura del modello su file
    # The improved logging method
    with open(output_path, 'w') as f:
        f.write(f"Output_shape: {in_features}\n\n")
        f.write("--- Model Parameter Names for Freezing/Unfreezing ---\n\n")
        for name, param in model.named_parameters():
            # This writes the EXACT name, its current freeze state, and its shape
            f.write(f"Name: {name:<40} | Trainable: {param.requires_grad} | Shape: {list(param.shape)}\n")
        f.write("\n--- Model Structure ---\n\n")
        f.write(str(model))
    

if __name__ == "__main__":
    main()