# Functions and datasets for extracting debug information
def explore_data(shard_pattern, load_in_memory=False, 
                               split_workers=True, batch_size=4):
    def create_df():
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
                            shard_name = json_data.get("shard_name", "unknown") 
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
                        if len(parts) == 3:
                            questionnaire = parts[0]
                            modality_type = parts[1]
                            
                            try:
                                if isinstance(value, bytes):
                                    img_source = Image.open(io.BytesIO(value))
                                    #convert to grayscale
                                    img = img_source.convert('L')
                                else:
                                    img = value # Fallback in case it somehow got decoded
                                #compute mean intensity
                                arr=np.array(img)
                                mean_intensity = arr.mean()
                                #compute std
                                std_intensity = arr.std()
                                #compute ink density
                                threshold = 128                                # pixels darker than this = ink
                                ink_pixels = (arr < threshold).sum()
                                ink_density_binary = ink_pixels / arr.size    # fraction of inked pixels

                                img_properties = {
                                    'format': img_source.format, #img format theimage was loaded from
                                    'num_channels_original': len(img_source.getbands()), #number of channels in the original image before conversion
                                    'mode': img_source.mode, #the color mode (e.g., RGB, RGBA, L)
                                    'size': img.size, #width, height
                                    'ratio': img.size[0] / img.size[1] if img.size[1] != 0 else None, #aspect ratio
                                    'mean_intensity': mean_intensity, #mean pixel intensity
                                    'std_intensity': std_intensity, #std of pixel intensity
                                    'ink_density_binary': ink_density_binary, #fraction of pixels that are ink (binary threshold)
                                }

                                #get the filename from the shard_name path
                                shard_filename = os.path.basename(shard_name)

                                yield subject_id, questionnaire, modality_type, label, shard_filename, img_properties
                                
                            except Exception as e:
                                print(f"Skipping corrupted image {key}: {e}")
                                
        return flatten_samples

    
    # 2. File gathering
    shard_files = glob.glob(shard_pattern)
    shard_files.sort()

    if load_in_memory:
        return 0 # Handle your in-memory logic here
        
    # 3. Build the WebDataset Pipeline
    dataset = wds.WebDataset(shard_files, shardshuffle=False)

    if split_workers:
        dataset = dataset.select(wds.split_by_worker)

    # 4. Apply the flattening and batching
    dataset = (dataset
        #.decode(decode_approach)
        .compose(create_df()) # This replaces .map() and .select()
    )
    
    return dataset

