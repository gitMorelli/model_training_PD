import pandas as pd
import numpy as np


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

