import contextlib
import glob
import os
from pathlib import Path
import sys
import numpy as np
import cv2
from time import perf_counter
import time
import tarfile
import shutil
import h5py


class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()

@contextlib.contextmanager
def tee_stdout(path, mode="w"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        old = sys.stdout
        sys.stdout = _Tee(old, f)
        try:
            yield
        finally:
            sys.stdout = old

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


def get_id_data_from_h5_file(file_path, target_id):
    """
    Retrieves data only for a specific ID.
    Returns: {q_name: {class_key: (scalar, array)}} or None if ID not found.
    """
    # Ensure ID is a string to match the H5 group naming convention
    target_id = str(target_id)
    id_data = {}

    with h5py.File(file_path, 'r') as f:
        # Check if the ID exists in the file
        if target_id not in f:
            print(f"ID {target_id} not found in {file_path}")
            return None
        
        id_grp = f[target_id]
        
        # Iterate through the questionnaires for this specific ID
        for q_name in id_grp.keys():
            id_data[q_name] = {}
            q_grp = id_grp[q_name]
            
            # Iterate through the datasets (classes) in the questionnaire
            for class_key in q_grp.keys():
                dset = q_grp[class_key]
                
                # Extract the array and the scalar attribute
                array_val = dset[:]
                scalar_val = dset.attrs.get('scalar_value')
                
                id_data[q_name][class_key] = (scalar_val, array_val)
                
    return id_data

def get_last_best_checkpoint(checkpoint_dir,version):
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir,f"v_{version}", "*best*.ckpt"))
    checkpoint_to_load = max(checkpoint_files, key=os.path.getctime) if checkpoint_files else None
    checkpoint_name = os.path.basename(checkpoint_to_load) if checkpoint_to_load else None
    checkpoint_to_load=f"v_{version}/{checkpoint_name}" if checkpoint_name else None
    print(f"Checkpoint to load: {checkpoint_name}", flush=True)
    return checkpoint_to_load