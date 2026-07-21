import tarfile
import glob
import os
import json
import pandas as pd
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
    load_path = "/home/a_morelli/datasets/id_lists/final_table_for_matching_splitted_13_7_26.csv"
    csv_data = pd.read_csv(load_path)
    selected = pd.read_parquet("/home/a_morelli/datasets/id_lists/PD_training_set_20_07_26.parquet")
    columns = csv_data.columns
    for col in columns:
        print(f"Column: {col}")
    # show unique values of the column rempli_seulq12 including nans, print numerosities of unique values
    unique_values = csv_data['rempli_seulq12'].value_counts(dropna=False)
    print("Unique values and their counts for 'rempli_seulq12':")
    print(unique_values)

def view_params():
    load_path = "/mnt/beegfs02/scratch/a_morelli/model_training/PD/resnet18_model_results/checkpoints/v_21/exp_params.pkl"
    with open(load_path, "rb") as f:
        exp_params = pickle.load(f)
    print("Experiment Parameters:")
    for key, value in exp_params.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    view_params()
    