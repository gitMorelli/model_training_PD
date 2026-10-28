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