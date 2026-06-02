import os
from pathlib import Path
import numpy as np
import cv2
from time import perf_counter
import time
import tarfile
import shutil
import h5py


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