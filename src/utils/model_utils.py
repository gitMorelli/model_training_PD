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


class SimpleMockModel(nn.Module):
    def __init__(self,seq_length=13*3):
        super().__init__()
        self.conv = nn.Conv2d(seq_length*3, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10) # 10 output classes
        self.seq_length = seq_length

    def forward(self, x):
        # x shape: [Batch, 6, 3, H, W]
        batch_size = x.shape[0]
        channels = x.shape[2] # Extract the channel dimension (3)
        # Flatten the time and channel dimensions: [Batch, 18, H, W]
        x = x.view(batch_size, -1, x.shape[3], x.shape[4]) 
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(batch_size, -1)
        return self.fc(x)