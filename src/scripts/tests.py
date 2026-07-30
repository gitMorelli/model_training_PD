from __future__ import annotations
import tarfile
import glob
import os
import json
import pandas as pd
import psutil
from PIL import Image
import io
import math
import numpy as np
import cv2
import re
import pickle
import time


from src.utils.data_loading_utils import pre_load_grid_data,prepare_pre_training

def test_read_json(tar_path="/mnt/beegfs01/scratch/a_morelli/extraction/final/data/id_A0C5I2D5.tar"):

    scale_tolerance = 0.1 
    templates_path="/home/a_morelli/datasets/others/template_sizes.json"
    #open the template info
    template_data = json.load(open(templates_path))
    with tarfile.open(tar_path, 'r') as old_tar:
        members = old_tar.getmembers()
        for m in members:
            if m.isfile() and m.name.endswith('.json'):
                #print("Extracting JSON file: ", m.name) #-> id_A0C5I2D5/q10/detections.json
                questionnaire = m.name.split("/")[1][1:] 
                print("Processing questionanire: ", questionnaire) 
                
                f = old_tar.extractfile(m)
                data = json.load(f)
                #print(list(data[0].keys())) #page_number, dimensions, detections
                #get the pages extracted with their dimension
                rescaling_factors = []
                n_images = len(data)
                for page in data:
                    if questionnaire=='8' and page['page_number'] in [1,2]:
                        print("Skipping page ", page['page_number'], " of questionnaire 8 due to known issues with template dimensions.")
                        n_images = len(data) - 2
                        continue
                    page_number = page['page_number']
                    #print(page_number)
                    template_dim = template_data[questionnaire][str(page_number)]
                    #print(template_dim)
                    page_dim = page['dimensions']
                    #print(f"Extracted Dimension: {page_dim}")
                    print(f"Page: {page_number} | Expected Dimension: {template_dim} | Extracted Dimension: {page_dim}")
                    scale_x = page_dim[0]/template_dim[0]
                    scale_y = page_dim[1]/template_dim[1]
                    if abs(scale_x-1)>scale_tolerance or abs(scale_y-1)>scale_tolerance:
                        print(f"Page {page_number} has a scale factor outside the tolerance: (x: {scale_x:.2f}, y: {scale_y:.2f})")
                        rescaling_factors.append((page_number, 1/scale_x, 1/scale_y))
                '''if n_images>0 and len(rescaling_factors)>=int(n_images/2):
                    return True, rescaling_factors
                else:
                    return False, rescaling_factors'''

def test_read_templates(output_path="/home/a_morelli/datasets/others"):
    templates_path="/home/a_morelli/temporary_data/test_parallel_censoring/test_parallelization/current_template"
    templates_files=glob.glob(os.path.join(templates_path, "*.json"))
    template_filenames = [os.path.basename(f) for f in templates_files]
    template_filenames = sorted(template_filenames)
    #remove filenames that contain the letter v
    template_filenames = [f for f in template_filenames if 'v' not in f]
    templates_files = [os.path.join(templates_path, f) for f in template_filenames]
    print("Available templates: ", template_filenames)
    template_sizes = {}
    for template_file in templates_files:
        print("Processing template: ", os.path.basename(template_file))
        #get the list of expected pages and their dimensions
        page_list = get_page_list(json.load(open(template_file)))
        print("Page list from template :", page_list)
        questionnaire = os.path.basename(template_file).split("_")[1].split(".")[0]
        template_pages = {}
        for page in page_list:
            dim = get_page_dimensions(json.load(open(template_file)), page)
            template_pages[page] = dim
            print(f"Page: {page} | Expected Dimension: {dim}")
        template_sizes[questionnaire] = template_pages
    # save the template sizes in a json file
    with open(os.path.join(output_path, "template_sizes.json"), "w") as f:
        json.dump(template_sizes, f, indent=4)
    

def extract_page_number(image_path):
    """
    Estrae il numero X dal nome del file che termina in page_X.png
    """
    match = re.search(r'page_(\d+)\.png$', image_path)
    if match:
        return int(match.group(1))
    return None

def get_page_list(json_data):
    """
    Restituisce una lista ordinata dei numeri di pagina presenti nel JSON.
    """
    page_numbers = []
    for entry in json_data:
        image_path = entry.get('image', '')
        p_num = extract_page_number(image_path)
        if p_num is not None:
            page_numbers.append(p_num)
    return sorted(page_numbers)

def get_page_dimensions(json_data, target_page_number):
    """
    Retrieves the (width, height) of the specified page.
    Returns a tuple (width, height) or None if not found.
    """
    for entry in json_data:
        # Check if this entry corresponds to the target page
        if extract_page_number(entry.get('image', '')) == target_page_number:
            
            # The dimensions are stored inside the 'label' list items
            labels = entry.get('label', [])
            
            if labels:
                # We assume the page dimensions are the same for all labels on that page,
                # so we take them from the first one.
                first_label = labels[0]
                width = first_label.get('original_width')
                height = first_label.get('original_height')
                return width, height
            else:
                # Label list is empty, cannot determine dimensions from this schema
                return None
                
    return None


def get_h5_data():
    id = "A0C5I2D5"
    h5_path = "/mnt/beegfs01/scratch/a_morelli/extraction/final/results_aggregated/final_aggregated_data.h5"
    id_data = get_id_data_from_h5_file(h5_path, id)
    print(f"Data for ID {id}:")
    for q_name, classes in id_data.items():
        print(f"  Questionnaire: {q_name}")
        for class_key, (scalar_val, array_val) in classes.items():
            print(f"    Class: {class_key} | Scalar Value: {scalar_val} | Array Shape: {array_val.shape}")
    #save an example image for a specific questionnaire and modality
    array_val = id_data['q6']['number'][1] #get the array value for q6 number class
    num_tiles = id_data['q6']['number'][0]
    img = save_class_image('q6','number','/home/a_morelli/vscode_projects/model_training/data',array_val,id)
    coords = get_tiles(img,array_val,num_tiles)
    #get the width and height of the first tile
    w = coords[0][0][2]-coords[0][0][0]
    h = coords[0][0][3]-coords[0][0][1]
    print(f"First tile width: {w} | Tile height: {h}")
    img_processed = recolor_border_via_profiles(img, coords)
    print(f"Image size is {img.shape} and number of tiles is {num_tiles} -> average tile size is {img.shape[0]/math.sqrt(num_tiles)}x{img.shape[1]/math.sqrt(num_tiles)}")
    #save the processed image
    save_image_path = os.path.join('/home/a_morelli/vscode_projects/model_training/data', f"{id}_q6_number_processed.png")
    cv2.imwrite(save_image_path, img_processed)

def save_class_image(q_name, class_key, subfolder,array_val,id):
    folder_path_id = os.path.join('/mnt/beegfs01/scratch/a_morelli/extraction/final/data', f"id_{id}")
    if not os.path.exists(folder_path_id+'.tar'):
        print(f"Tar file not found for ID {id} at expected path: {folder_path_id+'.tar'}")
        return

    with tarfile.open(folder_path_id+'.tar', 'r') as tar:
        #print the filenames in the tar to check the structure
        '''for member in tar.getmembers():
            print(member.name)'''
        image_path_in_tar = os.path.join(f"id_{id}", f"{q_name}",f"{class_key}.png")
        if image_path_in_tar in tar.getnames():
            image_file = tar.extractfile(image_path_in_tar)
            if image_file is not None:
                image_data = image_file.read()
                img = Image.open(io.BytesIO(image_data)).convert("RGB")
                img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                #draw the coordinates on the image
                img_with_coords = draw_coordinates_on_image(img_cv2, array_val)
                #save the image
                save_image_path = os.path.join(subfolder, f"{id}_{q_name}_{class_key}_with_coords.png")
                cv2.imwrite(save_image_path, img_with_coords)
            else:
                print(f"Could not extract {image_path_in_tar} from tar archive {folder_path_id+'.tar'}")
        else:
            print(f"Image {image_path_in_tar} not found in tar archive {folder_path_id+'.tar'}")
    return img_cv2

def draw_coordinates_on_image(image, coords, color=(0, 255, 0), thickness=1):
    """
    Draws vertical and horizontal lines based on a (2, N) array.
    
    Args:
        image (np.ndarray): The input image.
        coords (np.ndarray): Array of shape (2, N). 
                             coords[0, :] contains x-coordinates.
                             coords[1, :] contains y-coordinates.
        color (tuple): BGR color of the lines.
        thickness (int): Thickness of the lines.
    """
    # Create a copy to avoid modifying the original image array
    output_img = image.copy()
    h, w = output_img.shape[:2]
    
    # Iterate through the N columns
    num_points = coords.shape[1]
    
    for i in range(num_points):
        # 1. Draw Vertical Line at x = coords[0, i]
        x = int(coords[0, i])
        # Line from (x, 0) to (x, height)
        cv2.line(output_img, (x, 0), (x, h), color, thickness)
        
        # 2. Draw Horizontal Line at y = coords[1, i]
        y = int(coords[1, i])
        # Line from (0, y) to (width, y)
        cv2.line(output_img, (0, y), (w, y), color, thickness)
        
    return output_img


def get_tiles(image,coords,num_tiles):
    # Create a copy to avoid modifying the original image array
    h, w = image.shape[:2]
    #identify all the tiles defined by the coordinates
    #build the list of UL coordinates
    new_coords = []
    x_coords = [0]+sorted(list(coords[0, :]))+[w]
    y_coords = [0]+sorted(list(coords[1, :]))+[h]
    size = len(x_coords)-1
    processed_tiles=0
    for i in range(size):
        for j in range(size):
            processed_tiles+=1
            if processed_tiles<=num_tiles:
                new_coords.append([(int(x_coords[j]), int(y_coords[i]),int(x_coords[j+1]), int(y_coords[i+1])),processed_tiles])
            else:
                new_coords.append([(int(x_coords[j]), int(y_coords[i]),int(x_coords[j+1]), int(y_coords[i+1])),-1]) #empty tiles
    return new_coords


def recolor_border_via_profiles(image, coords, black_tolerance=5):
    img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, black_tolerance, 255, cv2.THRESH_BINARY)

    
    for tile in coords:
        x1, y1, x2, y2 = tile[0]
        tile_num = tile[1]
        tile_img = thresh[y1:y2, x1:x2]
        # Compute the sum of pixel values along rows and columns
        x_profile = np.sum(tile_img, axis=0)
        y_profile = np.sum(tile_img, axis=1)

        # 4. Find the indices where the profiles are greater than 0
        # This means there is at least one non-black pixel in that row/column
        x_content_indices = np.where(x_profile > 0)[0]
        y_content_indices = np.where(y_profile > 0)[0]

        # Handle the edge case where the image is entirely black
        if len(x_content_indices) == 0 or len(y_content_indices) == 0 or tile_num==-1:
            #set the tile to white
            image[y1:y2, x1:x2] = 255
            continue
        
        # 5. Identify the bounding box coordinates
        # The first and last indices represent the edges of the core content
        x_min, x_max = x_content_indices[0], x_content_indices[-1]
        y_min, y_max = y_content_indices[0], y_content_indices[-1]

        # 7. Create a white background of the original image size
        white_background = np.full_like(img[y1:y2,x1:x2], 255)

        # 8. Paste the core image into the exact same position on the white canvas
        white_background[y_min:y_max+1, x_min:x_max+1] = img[y1+y_min:y1+y_max+1, x1+x_min:x1+x_max+1].copy()

        image[y1:y2, x1:x2] = white_background.copy()
    return image

def inspect_00000():
    id_to_check = 'F1A0F2H5'
    group_to_check = '0998'
    load_path = "/home/a_morelli/datasets/id_lists/final_data_for_training.parquet"
    csv_data = pd.read_parquet(load_path)
    csv_data['subject_id'] = csv_data['unique_id'].str.split('_').str[0]
    csv_data['group_id'] = csv_data['unique_id'].str.split('_').str[1]
    #filter the csv_data to only include the id_to_check
    csv_data = csv_data[csv_data['subject_id'] == id_to_check]
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(csv_data) 
    '''
    csv_data = csv_data[csv_data['group_id'] == group_to_check]
    csv_data = csv_data[['group_id','subject_id','unique_id','case_control','grid_pattern','case_grid_pattern']]
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(csv_data)'''


def prepare_pre_training_data():
    load_path = "/home/a_morelli/datasets/id_lists/final_table_for_matching_splitted_13_7_26.csv"
    csv_data = pd.read_csv(load_path)
    selected = pd.read_parquet("/home/a_morelli/datasets/id_lists/PD_training_set_20_07_26.parquet")
    pre_training_df = prepare_pre_training(csv_data, selected)
    pre_training_df['case_grid_pattern'] = pre_training_df['grid_pattern']
    #save the pre_training_df to a parquet file
    save_path = load_path.replace(".csv", "_pre_training.parquet")
    pre_training_df.to_parquet(save_path, index=False)
    print(f"Pre-training data saved to {save_path}")

def inspect_columns(): 
    load_path = "/home/a_morelli/datasets/id_lists/final_table_for_matching_splitted_13_7_26_pre_training.parquet"
    csv_data = pd.read_parquet(load_path)
    columns = csv_data.columns
    for col in columns:
        print(f"Column: {col}")

def view_params():
    load_path = "/mnt/beegfs02/scratch/a_morelli/model_training/PD/resnet18_model_results/checkpoints/v_21/exp_params.pkl"
    with open(load_path, "rb") as f:
        exp_params = pickle.load(f)
    print("Experiment Parameters:")
    for key, value in exp_params.items():
        print(f"{key}: {value}")


def copy_folder(src, dst, overwrite=False):
    import shutil
    import sys
    from pathlib import Path
    src, dst = Path(src), Path(dst)

    if not src.is_dir():
        sys.exit(f"Source is not a directory: {src}")
    if dst.exists() and not overwrite:
        sys.exit(f"Destination already exists: {dst} (pass overwrite=True to merge)")

    # copytree with dirs_exist_ok lets a re-run fill in missing files
    shutil.copytree(src, dst, dirs_exist_ok=overwrite, symlinks=False)

    # verify: compare file counts and total bytes
    def stats(root):
        files = [p for p in Path(root).rglob("*") if p.is_file()]
        return len(files), sum(p.stat().st_size for p in files)

    n_src, sz_src = stats(src)
    n_dst, sz_dst = stats(dst)

    print(f"source:      {n_src} files, {sz_src / 1e9:.2f} GB")
    print(f"destination: {n_dst} files, {sz_dst / 1e9:.2f} GB")

    if n_src == n_dst and sz_src == sz_dst:
        print("OK — counts and sizes match")
    else:
        sys.exit("MISMATCH — copy is incomplete or corrupted")

def check_wds_signature():
    import inspect
    import webdataset as wds
    print(inspect.signature(wds.WebDataset.__init__))

def check_n_subjects_exclusion_criteria():
    N_Q = 13
    
    
    def combined_pattern(grid: str, case_grid: str) -> str:
        """Elementwise logical AND of two '0'/'1' strings."""
        return "".join(
            "1" if a == "1" and b == "1" else "0" for a, b in zip(grid, case_grid)
        )
    
    
    def _row_has_future_signal(row, start_offset: int) -> bool:
        combo = combined_pattern(row["grid_pattern"], row["case_grid_pattern"])
        start = max(int(row["last_avail_q"]) + start_offset, 0)
        return "1" in combo[start:N_Q]
    
    
    def summarize_splits(
        df: pd.DataFrame,
        start_offset: int = 0,
        id_col: str = "unique_id",
        splits=("train", "val"),
        verbose: bool = True,
        group_col='case_control'
    ) -> pd.DataFrame:
        """Print and return unique-id counts before/after filtering.
    
        An id is kept if ANY of its rows satisfies the rule (relevant only if
        the same unique_id appears on multiple rows; harmless otherwise).
        """
        d = df.copy()
        d["_keep_row"] = d.apply(_row_has_future_signal, axis=1, start_offset=start_offset)
    
        kept_ids = set(d.loc[d["_keep_row"], id_col].unique())
    
        records = []
        for split in splits:
            sub = d[d["split"] == split]
            for cc in sorted(sub[group_col].unique()):
                g = sub[sub[group_col] == cc]
                ids = set(g[id_col].unique())
                before = len(ids)
                after = len(ids & kept_ids)
                records.append(
                    {
                        "split": split,
                        "case_control": cc,
                        "n_before": before,
                        "n_after": after,
                        "n_removed": before - after,
                        "pct_kept": 100 * after / before if before else float("nan"),
                    }
                )
    
        summary = pd.DataFrame.from_records(records)
    
        if verbose:
            for split in splits:
                print(f"=== split: {split} ===")
                block = summary[summary["split"] == split]
                if block.empty:
                    print("  (no rows)")
                    continue
                for _, r in block.iterrows():
                    print(
                        f"  {group_col}={r.case_control}: "
                        f"{r.n_before:>8,} unique ids  ->  {r.n_after:>8,} after filtering "
                        f"({r.pct_kept:5.1f}% kept, {r.n_removed:,} removed)"
                    )
                print(
                    f"  total       : {block.n_before.sum():>8,} unique ids  ->  "
                    f"{block.n_after.sum():>8,} after filtering"
                )
                print()
    
        return summary

    load_path = "/mnt/beegfs02/scratch/a_morelli/model_training/PD/resnet18_model_results/checkpoints/v_21/PD_training_set_20_07_26.parquet"
    data = pd.read_parquet(load_path)

    summarize_splits(data, group_col="diag_park_final1_quest")

    columns = data.columns
    for col in columns:
        print(f"Column: {col}")

def repack_grid_dict():
    """Run once to convert the grid_dict pickle into packed numpy arrays."""
    import numpy as np
    import pickle
    import random

    exp_params = {}
    exp_params['use_grid'] = True
    exp_params['grid_dict_path'] = '/home/a_morelli/datasets/id_lists/h5/PD_data_h5.pkl'
    #"/home/a_morelli/datasets/id_lists/h5/pre_training_data_h5_21_07_26.pkl"
    save_dir = os.path.dirname(exp_params['grid_dict_path'])
    save_dir = os.path.join(save_dir, os.path.basename(exp_params['grid_dict_path']).replace(".pkl", ""))

    def load_grid_dict_local(exp_params):
        if exp_params['use_grid']:
            with open(exp_params['grid_dict_path'], "rb") as f:
                grid_dict = pickle.load(f)
            print("Grid dictionary loaded successfully. Number of entries:", len(grid_dict))
            return grid_dict
        else:
            print("Grid usage is disabled. No grid dictionary will be loaded.")
            return None

    grid_dict = load_grid_dict_local(exp_params)

    # ---------------------------------------------------------------- pack data
    keys = []        # (id, q_name, class_key) triples
    scalars = []
    arrays = []

    for id_, q_dicts in grid_dict.items():
        for q_name, class_dicts in q_dicts.items():
            for class_key, (scalar, arr) in class_dicts.items():
                keys.append((id_, q_name, class_key))
                scalars.append(scalar)
                arrays.append(np.asarray(arr))

    lengths = np.array([a.shape[1] for a in arrays], dtype=np.int64)   # number of columns
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    data = np.concatenate(arrays, axis=1)     # Concatenate all the 2xN_i arrays in a single 2xN array
    #we will slice from this one big array using the offsets to get the original arrays back

    print("value range:", data.min() if data.size else "empty",
          data.max() if data.size else "")

    scalars_arr = np.asarray(scalars)

    print(f"entries: {len(keys):,}")
    print(f"data: shape={data.shape}, dtype={data.dtype}, {data.nbytes/1e9:.2f} GB")
    print(f"scalars: dtype={scalars_arr.dtype}")

    # downcast BEFORE verification so the check validates exactly what gets saved
    data = data.astype(np.int32)          # verified: value range fits int32
    scalars_arr = scalars_arr.astype(np.int32)

    # ---------------------------------------------------------------- build integer-coded index
    ids     = sorted({k[0] for k in keys})
    q_names = sorted({k[1] for k in keys})
    c_keys  = sorted({k[2] for k in keys})
    id_code = {v: i for i, v in enumerate(ids)}
    q_code  = {v: i for i, v in enumerate(q_names)}
    c_code  = {v: i for i, v in enumerate(c_keys)}

    NQ, NC = len(q_names), len(c_keys)
    n_codes = len(ids) * NQ * NC
    print(f"vocab sizes: ids={len(ids):,}  q_names={NQ}  class_keys={NC}")
    print(f"n_codes (dense table size): {n_codes:,}  "
          f"-> int32 table = {n_codes * 4 / 1e9:.2f} GB")

    assert n_codes < 200_000_000, (
        f"n_codes={n_codes:,} too large for a dense table; "
        "switch to the sorted-codes + searchsorted variant instead")

    #every triple is collapsed into a single integer "code" using mixed-radix (row-major) encoding
    codes = np.array(
        [(id_code[a] * NQ + q_code[b]) * NC + c_code[c] for a, b, c in keys],
        dtype=np.int64)
    assert len(np.unique(codes)) == len(codes), "duplicate (id, q_name, class_key) triples!"

    # dense code -> entry index table, -1 = absent
    #entry_of_code is a dense reverse-lookup table of length n_codes, initialized to -1 (meaning "no such entry")
    entry_of_code = np.full(n_codes, -1, dtype=np.int32)
    entry_of_code[codes] = np.arange(len(codes), dtype=np.int32)

    vocabs = {"id": id_code, "q": q_code, "c": c_code, "NQ": NQ, "NC": NC}

    # ---------------------------------------------------------------- verify
    def grid_lookup(id_, q_name, class_key):
        code = (vocabs["id"][id_] * vocabs["NQ"] + vocabs["q"][q_name]) * vocabs["NC"] \
               + vocabs["c"][class_key]
        i = entry_of_code[code]
        if i < 0:
            raise KeyError((id_, q_name, class_key))
        lo, hi = offsets[i], offsets[i + 1]
        return scalars_arr[i], data[:, lo:hi]   # shape (2, N_i), zero-copy view

    # 1. structural checks (exhaustive)
    n_orig = sum(len(cd) for qd in grid_dict.values() for cd in qd.values())
    assert n_orig == len(keys), f"entry count mismatch: {n_orig} vs {len(keys)}"
    assert offsets[-1] == data.shape[1], "offsets don't cover data"
    assert (entry_of_code >= 0).sum() == len(keys), "dense table entry count mismatch"

    # 2. value check on a random sample (set n_check = len(keys) for a full pass)
    n_check = 5000
    rng = random.Random(0)
    for id_, q_name, class_key in rng.sample(keys, min(n_check, len(keys))):
        orig_scalar, orig_arr = grid_dict[id_][q_name][class_key]
        new_scalar, new_arr = grid_lookup(id_, q_name, class_key)

        assert np.isclose(new_scalar, orig_scalar), \
            f"scalar mismatch at {(id_, q_name, class_key)}: {orig_scalar} vs {new_scalar}"
        orig_arr = np.asarray(orig_arr)
        assert new_arr.shape == orig_arr.shape, \
            f"shape mismatch at {(id_, q_name, class_key)}: {orig_arr.shape} vs {new_arr.shape}"
        assert np.allclose(new_arr, orig_arr), \
            f"array values mismatch at {(id_, q_name, class_key)}"

    # 3. absence check: a code with no entry must raise KeyError / return -1
    absent = np.flatnonzero(entry_of_code < 0)
    if len(absent):
        assert entry_of_code[absent[0]] == -1
        print(f"absent codes: {len(absent):,} (correctly marked -1)")

    print(f"verification passed on {min(n_check, len(keys)):,} random entries "
          f"+ structural checks")

    # ---------------------------------------------------------------- save
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "grid_data.npy"), data)
    np.save(os.path.join(save_dir, "grid_offsets.npy"), offsets)
    np.save(os.path.join(save_dir, "grid_scalars.npy"), scalars_arr)
    np.save(os.path.join(save_dir, "grid_entry_of_code.npy"), entry_of_code)
    with open(os.path.join(save_dir, "grid_vocabs.pkl"), "wb") as f:
        pickle.dump(vocabs, f)
    print(f"saved to {save_dir}: grid_data.npy, grid_offsets.npy, "
          f"grid_scalars.npy, grid_entry_of_code.npy, grid_vocabs.pkl")

def get_grid_properties():
    exp_params = {}
    exp_params['use_grid'] = True
    exp_params['grid_dict_path'] = '/home/a_morelli/datasets/id_lists/h5/PD_data_h5.pkl'

    def load_grid_dict_local(exp_params):
        if exp_params['use_grid']:
            with open(exp_params['grid_dict_path'], "rb") as f:
                grid_dict = pickle.load(f)
            print("Grid dictionary loaded successfully. Number of entries:", len(grid_dict))
            return grid_dict
        else:
            print("Grid usage is disabled. No grid dictionary will be loaded.")
            return None

    grid_dict = load_grid_dict_local(exp_params)


    def grid_stats(arr):
        arr = np.asarray(arr)
        x = np.sort(arr[0])          # x coords of vertical lines
        #add coordinate 0
        x = np.insert(x, 0, 0)
        y = np.sort(arr[1])          # y coords of horizontal lines
        y = np.insert(y, 0, 0)
        widths  = np.diff(x)         # chunk widths  (N-1 values)
        heights = np.diff(y)         # chunk heights (N-1 values)

        def stats(v, prefix):
            if v.size == 0:          # single line -> no gaps
                return {f"{prefix}_mean": np.nan,
                        f"{prefix}_median": np.nan,
                        f"{prefix}_std": np.nan}
            return {f"{prefix}_mean":   v.mean(),
                    f"{prefix}_median": np.median(v),
                    f"{prefix}_std":    v.std()}   # ddof=0 (population)

        return {
            "x_span": x[-1] - x[0],
            "y_span": y[-1] - y[0],
            **stats(widths,  "width"),
            **stats(heights, "height"),
        }

    rows = []
    for id_, q_dicts in grid_dict.items():
        for q_name, class_dicts in q_dicts.items():
            for class_key, (scalar, arr) in class_dicts.items():
                rows.append({"id": id_, "q": q_name, "class": class_key,
                            **grid_stats(arr)})

    df = pd.DataFrame(rows)
    
    # numeric columns to describe (everything except the index columns)
    stat_cols = [c for c in df.columns if c not in ("id", "q", "class")]

    out_path = "/home/a_morelli/vscode_projects/model_training/results/tests/grid_stats_by_q_class.txt"

    with open(out_path, "w") as f:
        for class_val in sorted(df["class"].unique()):
            sub = df[df["class"] == class_val]

            f.write(f"class = {class_val}   "
                    f"(n = {len(sub)}) ---\n")
            f.write(sub[stat_cols].describe().to_string())
            f.write("\n\n")
        f.write("\n\n")
        f.write("=" * 70 + "\n")
        f.write("GROUPED BY QUESTIONNAIRE\n")
        f.write("=" * 70 + "\n\n")
        for q_val in sorted(df["q"].unique()):          # q = 1 .. 13
            f.write("=" * 70 + "\n")
            f.write(f"GROUP  q = {q_val}\n")
            f.write("=" * 70 + "\n\n")

            q_df = df[df["q"] == q_val]

            for class_val in sorted(q_df["class"].unique()):
                sub = q_df[q_df["class"] == class_val]

                f.write(f"--- q = {q_val} | class = {class_val}   "
                        f"(n = {len(sub)}) ---\n")
                f.write(sub[stat_cols].describe().to_string())
                f.write("\n\n")

            f.write("\n")

    print(f"saved to {out_path}")


def get_normalization_values():
    input_file = "/home/a_morelli/datasets/others/dataset_overview.csv"
    df = pd.read_csv(input_file)
    pix_sum = df['pixel_sum'].sum()                
    pix_sqsum = df['pixel_sq_sum'].sum()           
    pix_count = df['num_pixels'].sum()             
    mean = pix_sum / pix_count                  
    var  = pix_sqsum / pix_count - mean ** 2
    std  = np.sqrt(np.clip(var, 0, None))                          # clamp guards tiny negative drift
    print("Normalization values:")
    print(f"Mean: {mean}")
    print(f"Std: {std}")
if __name__ == "__main__":
    import io, os, time, torch

    sd = trainer.strategy.lightning_module_state_dict()
    sd = {k: v.cpu() for k, v in sd.items()}

    for tgt in ["/dev/shm/t.ckpt", "/tmp/t.ckpt",
                "/mnt/beegfs02/scratch/a_morelli/t.ckpt"]:
        try:
            t = time.time(); torch.save(sd, tgt); d = time.time() - t
            mb = os.path.getsize(tgt) / 1e6
            print(f"{tgt:50s} {d:6.2f}s  {mb/d:7.1f} MB/s")
            os.remove(tgt)
        except Exception as e:
            print(f"{tgt}: {e}")

    # same payload, one large buffered write instead of torch.save's chunked writes
    buf = io.BytesIO(); torch.save(sd, buf)
    t = time.time()
    with open("/mnt/beegfs02/scratch/a_morelli/t2.ckpt", "wb", buffering=8 << 20) as f:
        f.write(buf.getbuffer())
    print(f"beegfs buffered: {time.time()-t:.2f}s")
    