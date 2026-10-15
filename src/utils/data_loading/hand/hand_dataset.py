# Datasets and dataloaders for handedness model
def prepare_handedness_dataset(shard_pattern, decode_approach='pil',load_in_memory=False, split_workers=True, 
                               batch_size=4, transform = None, modality='X',rate=1, balanced_data=False, exclusion_set=set()):
    def filter(sample):
        # 'sample' is a dictionary. 
        # Assumes you have decoded the class label (e.g., via .cls or custom key)
        label = sample[1].item()  # Adjust this if your label is stored differently
        subject_id = sample[2]  # Assuming the subject ID is stored in the third position of the tuple returned by select_single_modality
        
        if label == -1:
            return False  # Filter out samples with missing labels
        elif subject_id in exclusion_set:
            return False  # Filter out samples whose subject ID is in the exclusion set
        '''elif label == MAJORITY_CLASS_ID and balanced_data:
            # Keep only 20% of the majority class samples
            return random.random() < BALANCING_FACTOR*rate  # Adjust the rate as needed (e.g., 0.2 for 20%)'''
        # Always keep minority classes
        return True
    def select_single_modality(sample, transform=None, modality='X'):
        if modality == 'X':
            modality_string = 'X'
        elif modality == 'text':
            modality_string = 'hand'
        elif modality == 'digit':
            modality_string = 'number_random'

        img_tensor = None
        label = None
        blank_image = torch.zeros(3, 224, 224)  # Assuming 3 channels and 224x224 size for ResNet18
        
        for key, value in sample.items():
            #print(f"Processing key: {key} with value type: {type(value)}")
            if key.endswith((".png", ".jpg", ".jpeg")):
                parts = key.split('.')
                #example key: q5.number_random.png

                #print(f"Processing key: {key} with parts: {parts}")
                #raise Exception("Debugging: Stopping after processing the first image key to check the key structure and modality matching.")

                # If it matches our target modality, process it
                if len(parts) == 3 and parts[1].lower() == modality_string.lower():
                    try:
                        if transform is not None:
                            img_tensor = transform(value) 
                        elif isinstance(value, torch.Tensor):
                            img_tensor = value
                        else:
                            img_tensor = T.ToTensor()(value)
                    except Exception as e:
                        print(f"Skipping corrupted image {key}: {e}")
                            
            elif key.endswith("json"):
                label = torch.tensor(value.get("label", -1), dtype=torch.long)
                subject_id = value.get("subject", "unknown") 
        #raise Exception("Debugging: Stopping after processing the first sample to check the key structure, modality matching, and label extraction.")
                
        # If we are missing either the image or the label, return the filter flag (-1)
        if img_tensor is None or label is None:
            return blank_image, torch.tensor(-1, dtype=torch.long) , subject_id,0,0
        
        # Return the image directly. 
        # Shape will be (Channels, Height, Width) instead of (1, Channels, Height, Width)
        return img_tensor, label , subject_id,0,0
    
    # 1. Use glob to find all files matching the pattern
    shard_files = glob.glob(shard_pattern)
    # Sort them just to be safe so they load in order
    shard_files.sort()


    if load_in_memory:
        return 0
    else:
        # 1. Define the base WDS Pipeline
        dataset = wds.WebDataset(shard_files, shardshuffle=100)

        # 2. Conditionally apply worker splitting
        if split_workers:
            dataset = dataset.select(wds.split_by_worker)

        # 3. Apply the remaining transformations
        dataset = (dataset
            .decode(decode_approach)
            .map(lambda sample: select_single_modality(sample, transform,modality=modality)) 
            .select(filter) # to filter missing data (labelled as -1)
            .batched(batch_size,partial=False) 
        )
    return dataset

def prepare_handedness_dataset_all(shard_pattern, decode_approach='pil', load_in_memory=False, 
                               split_workers=True, batch_size=4, transform=None, exclusion_set=set(), modality='X',huggingface_transform=False,
                               augmentation_transform=None, invert_color=False,n_views=1, grid_dict = None):
    if modality == 'X':
        modality_string = 'X'
    elif modality == 'text':
        modality_string = 'hand'
    elif modality == 'digit':
        modality_string = 'number_random'
    elif modality == 'sent':
        modality_string = 'hand_sentences_full'
    else:
        modality_string = 'all'
    
    # 2. File gathering
    shard_files = glob.glob(shard_pattern)
    shard_files.sort()

    if load_in_memory:
        return 0 # Handle your in-memory logic here
        
    # 3. Build the WebDataset Pipeline
    dataset = wds.WebDataset(shard_files, shardshuffle=100)

    if split_workers:
        dataset = dataset.select(wds.split_by_worker)

    # 4. Apply the flattening and batching
    if modality=='all':
        dataset = (dataset
            #.decode(decode_approach)
            .compose(create_flattener_handedness_multimode(transform,augmentation_transform, modalities_list = ('hand', 'number_random', 'X'),
                                                exclusion_set=exclusion_set, huggingface_transform=huggingface_transform, invert_color=invert_color)) # This replaces .map() and .select()
            .batched(batch_size,partial=False)
        )
    elif grid_dict:
        dataset = (dataset
            #.decode(decode_approach)
            .compose(create_flattener_handedness_grid(transform,augmentation_transform, n_views=n_views ,modality_string=modality_string,
                                                exclusion_set=exclusion_set, huggingface_transform=huggingface_transform, invert_color=invert_color,
                                                grid_dict=grid_dict)) # This replaces .map() and .select()
            .batched(batch_size,partial=False)
        )
    elif n_views > 1:
        dataset = (dataset
            #.decode(decode_approach)
            .compose(create_flattener_handedness_multiview(transform,augmentation_transform, n_views=n_views ,modality_string=modality_string,
                                                exclusion_set=exclusion_set, huggingface_transform=huggingface_transform, invert_color=invert_color)) # This replaces .map() and .select()
            .batched(batch_size,partial=False)
        )
    else:
        dataset = (dataset
            #.decode(decode_approach)
            .compose(create_flattener_handedness(transform,augmentation_transform, modality_string=modality_string,
                                                exclusion_set=exclusion_set, huggingface_transform=huggingface_transform, invert_color=invert_color)) # This replaces .map() and .select()
            .batched(batch_size,partial=False)
        )
    
    return dataset

def create_flattener_handedness(transform_func,augmentation_transform, modality_string, exclusion_set, huggingface_transform=False, invert_color=False):
        def flatten_samples(src):
            """Takes an iterator of samples and yields multiple individual images."""
            for sample in src:
                label = None
                subject_id = "unknown"
                
                # Step A: Find the JSON file first to get the label and subject
                for key, value in sample.items():
                    if key.endswith("json"):
                        try:
                            json_str = value.decode('utf-8') if isinstance(value, bytes) else value
                            json_data = json.loads(json_str)
                            label = torch.tensor(json_data.get("label", -1), dtype=torch.long)
                            subject_id = json_data.get("subject", "unknown")
                        except Exception as e:
                            print(f"Error parsing JSON for sample {key}: {e}")
                        break
                        
                # Step B: Filter at the sample level
                # If the sample is missing a label or has a -1 label, skip the whole sample
                if label is None or label.item() == -1:
                    continue
                    
                # Step C: Process and YIELD images one by one
                for key, value in sample.items():
                    if key.endswith((".png", ".jpg", ".jpeg")):
                        parts = key.split('.')
                        
                        # Expected format: q5.number_random.png
                        if len(parts) == 3 and parts[1].lower() == modality_string.lower():
                            questionnaire = parts[0]
                            modality_type = parts[1]

                            complete_example_id = f"{subject_id}_{questionnaire[1:]}"
                            if complete_example_id in exclusion_set: #the exclusion set ids are the subjects_id + the questionnaire number
                                #since the exclusion is specific for the questionnaire, no tfor the subject 
                                continue
                            
                            try:
                                if isinstance(value, bytes):
                                    img = Image.open(io.BytesIO(value))
                                    img = img.convert('RGB') # Standardize to RGB just in case
                                else:
                                    img = value # Fallback in case it somehow got decoded

                                if augmentation_transform is not None:
                                    img = augmentation_transform(img)
                                if invert_color:
                                    img = ImageOps.invert(img)
                                # Apply transformations
                                if transform_func is not None:
                                    if huggingface_transform:
                                        inputs = transform_func(images=img, return_tensors="pt")
                                        img_tensor = inputs['pixel_values'][0]
                                    else:
                                        img_tensor = transform_func(img) 
                                elif isinstance(img, torch.Tensor):
                                    img_tensor = img
                                else:
                                    img_tensor = T.ToTensor()(img)
                                
                                # YIELD ONE ROW AT A TIME
                                yield img_tensor, label, subject_id, questionnaire, modality_type
                                
                            except Exception as e:
                                print(f"Skipping corrupted image {key}: {e}")
                                
        return flatten_samples

def create_flattener_handedness_multiview(transform_func,augmentation_transform, n_views,modality_string, exclusion_set, 
                                          huggingface_transform=False, invert_color=False):
        def flatten_samples(src):
            """Takes an iterator of samples and yields multiple individual images."""
            for sample in src:
                label = None
                subject_id = "unknown"
                
                # Step A: Find the JSON file first to get the label and subject
                for key, value in sample.items():
                    if key.endswith("json"):
                        try:
                            json_str = value.decode('utf-8') if isinstance(value, bytes) else value
                            json_data = json.loads(json_str)
                            label = torch.tensor(json_data.get("label", -1), dtype=torch.long)
                            subject_id = json_data.get("subject", "unknown")
                        except Exception as e:
                            print(f"Error parsing JSON for sample {key}: {e}")
                        break
                        
                # Step B: Filter at the sample level
                # If the sample is missing a label or has a -1 label, skip the whole sample
                if label is None or label.item() == -1:
                    continue
                    
                # Step C: Process and YIELD images one by one
                for key, value in sample.items():
                    if key.endswith((".png", ".jpg", ".jpeg")):
                        parts = key.split('.')
                        
                        # Expected format: q5.number_random.png
                        if len(parts) == 3 and parts[1].lower() == modality_string.lower():
                            questionnaire = parts[0]
                            modality_type = parts[1]

                            complete_example_id = f"{subject_id}_{questionnaire[1:]}"
                            if complete_example_id in exclusion_set: #the exclusion set ids are the subjects_id + the questionnaire number
                                #since the exclusion is specific for the questionnaire, no tfor the subject 
                                continue
                            
                            try:
                                if isinstance(value, bytes):
                                    img = Image.open(io.BytesIO(value))
                                    img = img.convert('RGB') # Standardize to RGB just in case
                                else:
                                    img = value # Fallback in case it somehow got decoded

                                list_of_views = []
                                for _ in range(n_views):
                                    if augmentation_transform is not None:
                                        img_view = augmentation_transform(img)
                                    else:
                                        raise Exception("Augmentation transform must be provided for multiview flattener to create different views.")
                                    
                                    if invert_color:
                                        img_view = ImageOps.invert(img_view)
                                    
                                    # Apply transformations
                                    if transform_func is not None:
                                        if huggingface_transform:
                                            inputs = transform_func(images=img_view, return_tensors="pt")
                                            img_tensor = inputs['pixel_values'][0]
                                        else:
                                            img_tensor = transform_func(img_view) 
                                    elif isinstance(img, torch.Tensor):
                                        img_tensor = img
                                    else:
                                        img_tensor = T.ToTensor()(img)
                                    
                                    list_of_views.append(img_tensor)
                                
                                stacked_views = torch.stack(list_of_views, dim=0)
                                yield stacked_views, label, subject_id, questionnaire, modality_string
                                
                            except Exception as e:
                                print(f"Skipping corrupted image {key}: {e}")
                                
        return flatten_samples

def create_flattener_handedness_multimode(
    transform_func,
    augmentation_transform, 
    modalities_list=('hand', 'number_random', 'X'), 
    exclusion_set=set(), 
    huggingface_transform=False, 
    invert_color=False
):
    def flatten_samples(src):
        reference_size = (224, 224) 
        for sample in src:
            label = None
            subject_id = "unknown"
            
            # Step A: Extract label and subject from JSON
            for key, value in sample.items():
                if key.endswith("json"):
                    try:
                        json_str = value.decode('utf-8') if isinstance(value, bytes) else value
                        json_data = json.loads(json_str)
                        label = torch.tensor(json_data.get("label", -1), dtype=torch.long)
                        subject_id = json_data.get("subject", "unknown")
                    except Exception as e:
                        print(f"Error parsing JSON for sample {key}: {e}")
                    break
                    
            if label is None or label.item() == -1:
                continue
                
            # Step B: Pre-index all available images in this sample by (q_number, modality)
            # This handles 'subject.qX.modality.png' transparently
            image_pool = {}
            
            for key, value in sample.items():
                if key.endswith((".png", ".jpg", ".jpeg")):
                    parts = key.split('.')
                    q_num = int(parts[0][1:])
                    mod_name = parts[1].lower()
                    image_pool[(q_num, mod_name)] = value

            # Step C: Loop explicitly through questionnaires 1 to 13
            mods_lower = [m.lower() for m in modalities_list]
            
            for X in range(1, 14):
                questionnaire = f"q{X}"
                complete_example_id = f"{subject_id}_{X}"
                
                # Exclusion check per specific questionnaire
                if complete_example_id in exclusion_set:
                    continue
                
                # Check if this questionnaire even has ANY data in the sample before processing blanks
                if not any((X, m) in image_pool for m in mods_lower):
                    continue

                final_tensors = []
                for m in mods_lower:
                    raw_image_data = image_pool.get((X, m))
                    
                    if raw_image_data is not None:
                        if isinstance(raw_image_data, bytes):
                            img = Image.open(io.BytesIO(raw_image_data)).convert('RGB')
                        else:
                            img = raw_image_data
                    else:
                        img = Image.new('RGB', reference_size, color=(255,255,255))  # Blank image if missing
                        #print(f"Warning: Missing modality {m} for questionnaire {questionnaire} in sample {complete_example_id}. Using blank image.")
                            
                    if augmentation_transform is not None:
                        img = augmentation_transform(img)
                    if invert_color:
                        img = ImageOps.invert(img)
                        
                    if transform_func is not None:
                        if huggingface_transform:
                            img_tensor = transform_func(images=img, return_tensors="pt")['pixel_values'][0]
                        else:
                            img_tensor = transform_func(img)
                    else:
                        img_tensor = T.ToTensor()(img)
                        
                    final_tensors.append(img_tensor)
                
                stacked_modalities = torch.stack(final_tensors, dim=0)
                #check that it is 3,3,224,224 
                #if not print the shape and the key to debug
                '''if stacked_modalities.shape != (len(mods_lower), 3, 224, 224):
                    print(f"Unexpected shape for {complete_example_id} in questionnaire {questionnaire}: {stacked_modalities.shape}")
                    continue'''

                yield stacked_modalities, label, subject_id, questionnaire, modalities_list

                
    return flatten_samples

def create_flattener_handedness_grid(transform_func,augmentation_transform, n_views,modality_string, exclusion_set, 
                                          huggingface_transform=False, invert_color=False, grid_dict=None):
        def flatten_samples(src):
            """Takes an iterator of samples and yields multiple individual images."""
            for sample in src:
                label = None
                subject_id = "unknown"
                
                # Step A: Find the JSON file first to get the label and subject
                for key, value in sample.items():
                    if key.endswith("json"):
                        try:
                            json_str = value.decode('utf-8') if isinstance(value, bytes) else value
                            json_data = json.loads(json_str)
                            label = torch.tensor(json_data.get("label", -1), dtype=torch.long)
                            subject_id = json_data.get("subject", "unknown")
                        except Exception as e:
                            print(f"Error parsing JSON for sample {key}: {e}")
                        break
                        
                # Step B: Filter at the sample level
                # If the sample is missing a label or has a -1 label, skip the whole sample
                if label is None or label.item() == -1:
                    continue
                    
                # Step C: Process and YIELD images one by one
                for key, value in sample.items():
                    if key.endswith((".png", ".jpg", ".jpeg")):
                        parts = key.split('.')
                        
                        # Expected format: q5.number_random.png
                        if len(parts) == 3 and parts[1].lower() == modality_string.lower():
                            questionnaire = parts[0]
                            modality_type = parts[1]
                            

                            complete_example_id = f"{subject_id}_{questionnaire[1:]}"
                            if complete_example_id in exclusion_set: #the exclusion set ids are the subjects_id + the questionnaire number
                                #since the exclusion is specific for the questionnaire, no tfor the subject 
                                continue
                            
                            try:
                                if isinstance(value, bytes):
                                    img = Image.open(io.BytesIO(value))
                                    img = img.convert('RGB') # Standardize to RGB just in case
                                else:
                                    img = value # Fallback in case it somehow got decoded
                                
                                num,grid = grid_lookup(grid_dict, subject_id, questionnaire, modality_type)
                                x_coords = [0]+sorted(list(grid[0, :]))+[img.width]
                                n_x = len(x_coords) -1
                                y_coords = [0]+sorted(list(grid[1, :]))+[img.height]
                                #n_y = len(y_coords) -1

                                list_of_views = []
                                for _ in range(n_views):
                                    #sample a random number between 1 and num included
                                    rand_num = random.randint(1,num)
                                    coordinates = (x_coords[(rand_num-1)%n_x], y_coords[(rand_num-1)//n_x], 
                                                   x_coords[(rand_num-1)%n_x+1], y_coords[(rand_num-1)//n_x+1])
                                    chunk = img.crop(coordinates)

                                    if augmentation_transform is not None:
                                        img_view = augmentation_transform(chunk)
                                    else:
                                        img_view = chunk
                                    if invert_color:
                                        img_view = ImageOps.invert(img_view)
                                    # Apply transformations
                                    if transform_func is not None:
                                        if huggingface_transform:
                                            inputs = transform_func(images=img_view, return_tensors="pt")
                                            img_tensor = inputs['pixel_values'][0]
                                        else:
                                            img_tensor = transform_func(img_view) 
                                    elif isinstance(img, torch.Tensor):
                                        img_tensor = img
                                    else:
                                        img_tensor = T.ToTensor()(img)
                                    
                                    list_of_views.append(img_tensor)
                                
                                stacked_views = torch.stack(list_of_views, dim=0)
                                yield stacked_views, label, subject_id, questionnaire, modality_string
                                
                            except Exception as e:
                                print(f"Skipping corrupted image {key}: {e}")
                                
        return flatten_samples

def test_flattener_handedness(transform_func,augmentation_transform, modality_string, exclusion_set, huggingface_transform=False, invert_color=False):
        def flatten_samples(src):
            """Takes an iterator of samples and yields multiple individual images."""
            for sample in src:
                label = None
                subject_id = "unknown"
                
                # Step A: Find the JSON file first to get the label and subject
                for key, value in sample.items():
                    if key.endswith("json"):
                        label = torch.tensor(0) #torch.tensor(value.get("label", -1), dtype=torch.long)
                        subject_id = 'A' # value.get("subject", "unknown")
                        break # Found it, no need to keep checking other keys for json
                        
                # Step B: Filter at the sample level
                # If the sample is missing a label or has a -1 label, skip the whole sample
                if label is None or label.item() == -1:
                    continue
                    
                # Step C: Process and YIELD images one by one
                for key, value in sample.items():
                    if key.endswith((".png", ".jpg", ".jpeg")):
                        parts = key.split('.')
                        
                        # Expected format: q5.number_random.png
                        if len(parts) == 3 and parts[1].lower() == modality_string.lower():
                            questionnaire = parts[0]
                            modality_type = parts[1]

                            complete_example_id = f"{subject_id}_{questionnaire[1:]}"
                            if complete_example_id in exclusion_set: #the exclusion set ids are the subjects_id + the questionnaire number
                                #since the exclusion is specific for the questionnaire, no tfor the subject 
                                continue
                            
                            try:
                                if augmentation_transform is not None:
                                    value = augmentation_transform(value)
                                if invert_color:
                                    value = ImageOps.invert(value)
                                # Apply transformations
                                if transform_func is not None:
                                    if huggingface_transform:
                                        inputs = transform_func(images=value, return_tensors="pt")
                                        img_tensor = inputs['pixel_values'][0]
                                    else:
                                        img_tensor = transform_func(value) 
                                elif isinstance(value, torch.Tensor):
                                    img_tensor = value
                                else:
                                    img_tensor = T.ToTensor()(value)
                                
                                # YIELD ONE ROW AT A TIME
                                yield img_tensor, label, subject_id, questionnaire, modality_type
                                
                            except Exception as e:
                                print(f"Skipping corrupted image {key}: {e}")
                                
        return flatten_samples

def test_handedness_dataset_all(shard_pattern, decode_approach='pil', load_in_memory=False, 
                               split_workers=True, batch_size=4, transform=None, exclusion_set=set(), modality='X',huggingface_transform=False,
                               augmentation_transform=None, invert_color=False):
    if modality == 'X':
        modality_string = 'X'
    elif modality == 'text':
        modality_string = 'hand'
    elif modality == 'digit':
        modality_string = 'number_random'
    
    # 2. File gathering
    shard_files = glob.glob(shard_pattern)
    shard_files.sort()

    if load_in_memory:
        return 0 # Handle your in-memory logic here
        
    # 3. Build the WebDataset Pipeline
    dataset = wds.WebDataset(shard_files, shardshuffle=100)

    if split_workers:
        dataset = dataset.select(wds.split_by_worker)

    # 4. Apply the flattening and batching
    dataset = (dataset
        .decode(decode_approach)
        .compose(test_flattener_handedness(transform,augmentation_transform, modality_string=modality_string,
                                             exclusion_set=exclusion_set, huggingface_transform=huggingface_transform, invert_color=invert_color)) # This replaces .map() and .select()
        .batched(batch_size,partial=False)
    )
    
    return dataset

def load_representations_handedness(output_path, as_dataframe=False, to_torch=False):
    """
    Load what run_inference_to_dataframe(..., output_path=...) wrote, and split
    the representation array along axis 1 into separately named arrays.
 
    Parameters
    ----------
    output_path : str | Path
        The same stem passed to the saver (e.g. "out/run01"). Reads
        <stem>_metadata.parquet, <stem>_representations.npy, <stem>_layout.json.
    as_dataframe : bool
        If True, return a single DataFrame: the metadata plus one column per
        modality/tile (each cell a length-N vector). Otherwise return
        (metadata, reps_by_name).
    to_torch : bool
        If True, the representation arrays are returned as torch.Tensor.
        (Ignored when as_dataframe=True.)
 
    Returns
    -------
    (metadata, reps_by_name)            if as_dataframe is False
        metadata     : pd.DataFrame, row i aligned with each array's row i
        reps_by_name : dict[str, np.ndarray|torch.Tensor], each (num_rows, N)
            keys are the descriptive names recovered from the layout:
              data_modality == 'all'      -> 'text', 'digit', 'cX'
              single modality, tiles == 1 -> '<data_modality>'
              single modality, tiles == k -> '<data_modality>_tile0' ...
    df : pd.DataFrame                    if as_dataframe is True
    """
    base_name = os.path.basename(output_path)
    meta_file = os.path.join(output_path,base_name + '_metadata.parquet')
    repr_file = os.path.join(output_path,base_name + '_representations.npy')
    layout_file = os.path.join(output_path,base_name + '_layout.json')
 
 
    metadata = pd.read_parquet(meta_file)
    representations = np.load(repr_file)
    with open(layout_file) as fh:
        layout = json.load(fh)
 
    axis1_labels = layout.get('axis1_labels')
    data_modality = layout.get('data_modality')
 
    if axis1_labels is None:
        # (num_rows, N): a single representation, named after the modality.
        reps_by_name = {data_modality: representations}
    else:
        # (num_rows, M, N): fan axis 1 out into named (num_rows, N) arrays.
        if representations.shape[1] != len(axis1_labels):
            raise ValueError(
                f"axis-1 size {representations.shape[1]} does not match the "
                f"{len(axis1_labels)} labels in the layout ({axis1_labels})."
            )
        reps_by_name = {
            label: np.ascontiguousarray(representations[:, i, :])
            for i, label in enumerate(axis1_labels)
        }
 
    if as_dataframe:
        df = metadata.copy()
        for name, arr in reps_by_name.items():
            df[f'repr_{name}'] = list(arr)  # one length-N vector per cell
        return df, None
 
    if to_torch:
        reps_by_name = {k: torch.from_numpy(v) for k, v in reps_by_name.items()}
 
    return metadata, reps_by_name
