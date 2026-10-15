import os
import sys
import h5py
import pickle
import numpy as np

#Loading the grid_file data
def pre_load_grid_data(h5_filepath,csv_data):
    '''
    this function takes a csv with the samples to consider and for each 
    saves the grid file in a csv (load grids for all modalities)
    '''
    import time
    start = time.time()
    unique_subjects = csv_data['ident_projet'].unique()
    
    full_dict=get_ids_data_from_h5_file_list(h5_filepath, unique_subjects)
        
    end = time.time()
    print(f"Preloaded grid data for {len(unique_subjects)} subjects in {end - start:.2f} seconds.")
    #get the memory used by the full_dict in MBs
    total_size = sys.getsizeof(full_dict) / (1024 * 1024)
    print(f"Total memory used by preloaded grid data: {total_size:.2f} MB")
    return full_dict
def get_ids_data_from_h5_file_list(file_path, target_ids):
    """
    Retrieves data for a list of IDs in a single file open.
    Returns: {id: {q_name: {class_key: (scalar, array)}}}
    IDs not found in the file are omitted (and reported).
    """
    target_ids = [str(i) for i in target_ids]
    results = {}

    with h5py.File(file_path, 'r') as f:
        for i,target_id in enumerate(target_ids):
            if target_id not in f:
                print(f"ID {target_id} not found in {file_path}")
                continue

            id_grp = f[target_id]
            id_data = {}

            for q_name in id_grp.keys():
                q_grp = id_grp[q_name]
                id_data[q_name] = {
                    class_key: (
                        q_grp[class_key].attrs.get('scalar_value'),
                        q_grp[class_key][()],          # [()] reads the full array
                    )
                    for class_key in q_grp.keys()
                }

            results[target_id] = id_data
            if i%100 == 0:
                print(f"Processed {i+1}/{len(target_ids)} IDs from {file_path}", flush=True)
                total_size = sys.getsizeof(results) / (1024 * 1024)
                print(f"Total memory used until now: {total_size:.2f} MB", flush=True)

    return results
def load_grid_dict_old(exp_params):
    if exp_params['use_grid']:
        """Load the grid dictionary from a pickle file."""
        with open(exp_params['grid_dict_path'], "rb") as f:
            grid_dict = pickle.load(f)
        print("Grid dictionary loaded successfully. Number of entries:", len(grid_dict))
        return grid_dict
    else:
        print("Grid usage is disabled. No grid dictionary will be loaded.")
        return None
def load_grid_dict(exp_params):
    if exp_params['use_grid']:
        base = exp_params['grid_dict_path']  # wherever you saved step 1 
        packed = {
            "data":          np.load(os.path.join(base, "grid_data.npy"), mmap_mode='r'),
            "offsets":       np.load(os.path.join(base, "grid_offsets.npy")),
            "scalars":       np.load(os.path.join(base, "grid_scalars.npy")),
            "entry_of_code": np.load(os.path.join(base, "grid_entry_of_code.npy"), mmap_mode='r'),
        }
        with open(os.path.join(base, "grid_vocabs.pkl"), "rb") as f:
            packed["vocabs"] = pickle.load(f)
        return packed
    else:
        print("Grid usage is disabled. No grid dictionary will be loaded.")
        return None
def grid_lookup(packed, id_, q_name, class_key):
    v = packed["vocabs"]
    code = (v["id"][id_] * v["NQ"] + v["q"][q_name]) * v["NC"] + v["c"][class_key]
    i = packed["entry_of_code"][code]
    if i < 0:
        raise KeyError((id_, q_name, class_key))
    lo, hi = packed["offsets"][i], packed["offsets"][i + 1]
    return packed["scalars"][i], packed["data"][:, lo:hi]
