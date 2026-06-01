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







if __name__ == "__main__":
    test_read_json()
    #test_read_templates()