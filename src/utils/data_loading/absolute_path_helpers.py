import os

# Path selection
def return_file_paths(problem,grouped,pre_training):
    if problem == 'PD' and not grouped and not pre_training:
        list_of_ids_paths = "/home/a_morelli/datasets/id_lists/PD_training_set_20_07_26.parquet" 
        #"/home/a_morelli/datasets/id_lists/PD_training_set_13_7_26.parquet"
        data_folder = '/mnt/beegfs02/scratch/a_morelli/model_training/shards/PD/final_png_whitebg_21_07_26'
        #"/mnt/beegfs02/scratch/a_morelli/model_training/PD/final_png_whitebg"
        #"/home/a_morelli/datasets/shards/PD/final_png_whitebg_21_07_26"
        grid_dict_path = "/home/a_morelli/datasets/id_lists/h5/PD_data_h5"
    elif problem == 'PD' and pre_training:
        list_of_ids_paths = "/home/a_morelli/datasets/id_lists/final_table_for_matching_splitted_13_7_26_pre_training.parquet"
        data_folder = "/mnt/beegfs02/scratch/a_morelli/model_training/shards/PDpretraining/final_png_whitebg_21_07_26"
        grid_dict_path = "/home/a_morelli/datasets/id_lists/h5/pre_training_data_h5_21_07_26"
    elif problem == 'PD' and grouped and not pre_training:
        list_of_ids_paths = "/home/a_morelli/datasets/id_lists/PD_training_set_20_07_26.parquet"
        data_folder = '/mnt/beegfs02/scratch/a_morelli/model_training/shards/PD/final_png_whitebg_grouped_03_08_26'
        grid_dict_path = "/home/a_morelli/datasets/id_lists/h5/PD_data_h5"
    elif problem == 'handedness' and not grouped and not pre_training:
        list_of_ids_paths = "/home/a_morelli/datasets/id_lists/handedness_training_set_2_10_26.parquet"
        data_folder = "/mnt/beegfs02/scratch/a_morelli/model_training/shards/PDpretraining/final_png_whitebg_21_07_26"
        grid_dict_path = "/home/a_morelli/datasets/id_lists/h5/pre_training_data_h5_21_07_26"
    elif problem == 'unsupervised_pretraining':
        list_of_ids_paths = "/home/a_morelli/datasets/id_lists/final_table_for_matching_splitted_13_7_26_pre_training.parquet"
        data_folder = "/mnt/beegfs02/scratch/a_morelli/model_training/shards/PDpretraining/final_png_whitebg_21_07_26"
        grid_dict_path = "/home/a_morelli/datasets/id_lists/h5/pre_training_data_h5_21_07_26"
    return list_of_ids_paths, data_folder, grid_dict_path

def return_dt_path(input_path):
    #get the filename without extension of input_path
    name = os.path.splitext(os.path.basename(input_path))[0]
    map_names = {
        'PD_training_set_20_07_26': "/home/a_morelli/datasets/id_lists/dt_19_10_26.parquet",
    }
    dt_path = map_names.get(name, None)
    if dt_path is None:
        raise ValueError(f"dt_path not found for input_path: {input_path}")
    return dt_path

def get_representation_file_path(model_name):
    source_path = "/home/a_morelli/models/model_training_logs/PD/representation_extraction"
    map_names = {   
        'convnext_tiny_window_5': 'convnext_tiny/11112026_174449',
        'convnext_tiny_window_321': 'convnext_tiny/12112026_174415',
        'convnext_tiny_window_321_wrong': 'convnext_tiny/10112026_183319',
        'clip-vit-large-patch14-inter': 'clip-vit-large-patch14-inter/27082026'
    }
    representation_file_path = map_names.get(model_name, None)
    if representation_file_path is None:
        raise ValueError(f"Representation file path not found for model_name: {model_name}")
    return os.path.join(source_path, representation_file_path,'representations_PD.parquet')

def pre_trained_weights(name):
    if name is None:
        return None
    #options for custom_pre_trained_weights:
    if name == 'pre_trained_E3N_resnet18':
        out_path =os.path.join('/home/a_morelli/models/model_training_logs/pre_trained_models/E3N/resnet18_model_results/checkpoints/v_2',
                    'best-epoch=01-val_loss=0.23.ckpt')
    elif name == 'pre_trained_E3N_resnet18_window':
        out_path = os.path.join('/home/a_morelli/models/model_training_logs/pre_trained_models/E3N/resnet18_model_results/checkpoints/v_4',
                    'best-02-0.2222.ckpt')
    elif name == 'pre_trained_E3N_from_scratch_resnet18_window':
        out_path = os.path.join('/home/a_morelli/models/model_training_logs/pre_trained_models/E3N/resnet18_model_results/checkpoints/v_5',
                    'best-03-008299-0.1090.ckpt')
    elif name == 'pre_trained_E3N_resnet50':
        out_path = os.path.join('/home/a_morelli/models/model_training_logs/pre_trained_models/E3N/resnet50_model_results/checkpoints/v_1',
                    'best-epoch=00-val_loss=0.30.ckpt')
    elif name == 'pre_trained_E3N_resnet50_window':
        out_path = os.path.join('/home/a_morelli/models/model_training_logs/pre_trained_models/E3N/resnet50_model_results/checkpoints/v_3',
                    'best-03-008086-0.1167.ckpt')
    elif name == 'pre_trained_E3N_custom_window':
        out_path = os.path.join('/home/a_morelli/models/model_training_logs/pre_trained_models/E3N/FiveStageResidualStridedConvNet_model_results/checkpoints/v_4',
                    'best-00-001708-0.3423.ckpt')
    elif name == 'pre_trained_E3N_convnext_tiny_window_1':
        out_path = os.path.join('/home/a_morelli/models/model_training_logs/pre_trained_models/E3N/convnext_tiny_model_results/checkpoints/v_2',
                    'best-19-021127-0.1540.ckpt')
    '''os.path.join(
        '/home/a_morelli/models/model_training_logs/pre_trained_models/mnist',
        'resnet50/checkpoints/best-resnet18-mnist-epoch=28-val_loss=0.0197.ckpt'
    ),'''
    #resnet50/checkpoints/best-resnet18-mnist-epoch=28-val_loss=0.0197.ckpt
    #resnet18/checkpoints/best-resnet18-mnist-epoch=05-val_loss=0.0181.ckpt
    return out_path