import pandas as pd
import numpy as np

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
    return list_of_ids_paths, data_folder, grid_dict_path


#Preparing synthetic data dataframe 
def synthetic_data_override(exp_params, verbose=True):
    '''this function generate a new training set with the synthetic classes, save it to file and override the id_list_path
    -> the other functions go read that'''
    if exp_params['synthetic'] is None:
        return exp_params
    train_data = pd.read_parquet(exp_params['list_of_ids_paths'])
    train_data['synth_label'] = 0
    synthetic_modes = exp_params['synthetic']
    synthetic_proportions = exp_params['synthetic_proportions']
    num_synthetic_modes = len(synthetic_modes)
    #assign randomly the synthetic labels to the training set based on the proportions
    #shuffle the train_data
    rng = np.random.default_rng(exp_params['seed'])          # seed for reproducibility
    train_data['synth_label'] = rng.choice(num_synthetic_modes, size=len(train_data), p=synthetic_proportions)
    synthetic_path = exp_params['list_of_ids_paths'].replace('.parquet', '_synthetic.parquet')
    train_data.to_parquet(synthetic_path, index=False)
    if verbose:
        print(f"Synthetic data generated and saved to {synthetic_path} !!!!!!!!!!!!")
        print("Now overriding experiment settings...  ")
    exp_params['list_of_ids_paths'] = synthetic_path
    exp_params['num_classes'] = num_synthetic_modes
    exp_params['class_col'] = 'synth_label'
    exp_params['filter_missing'] = 'all'

    return exp_params

