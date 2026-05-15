import os
from pathlib import Path
import numpy as np
import cv2
from time import perf_counter
import time
import tarfile
import shutil


def recreate_dir(path, retries=5, delay=0.1):
    if os.path.exists(path):
        shutil.rmtree(path)
    
    for i in range(retries):
        try:
            os.makedirs(path)
            return # Success!
        except OSError:
            if i < retries - 1:
                time.sleep(delay)
            else:
                raise # Re-raise the error if it still fails after retries