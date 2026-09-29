import time
from tkinter.font import names
import os
import pandas as pd
import time

from dataclasses import dataclass
from typing import Optional, Sequence

from src.utils.data_loading_utils import load_representations_handedness
from src.utils.model_utils import get_sklearn_model
from src.utils.data_loading_utils import prepare_exclusion_sets_PD, return_file_paths
from src.utils.data_loading_utils import questionnaires_to_keep
from src.debug.PD_model_evaluation import  analyze_results
from sklearn.inspection import permutation_importance
from src.debug.PD_model_evaluation import tee_stdout, plot_probability_trajectories
from src.scripts.t_on_representations import balance_data, reshape_features, mask_features, add_info_to_df 

HOME_PATH = "/home/a_morelli/models/model_training_logs/PD/"
params = {

    "input_file":os.path.join(HOME_PATH, "feature_extraction/17092026/statistics_PD.csv"),
    "signal_column": 'cell_contour_jitter_rms_mean', #cell_sigma_x_mean
    
    "class_col": 'diag_park_final1_quest',
    #To put for compatibility with code
    "grouped": False,
    'filter_missing': 'last_q', #all

    "seed": 42,

    "balanced_data": False,
    'balance_validation': False, #if True the validation set is balanced, if False it is not balanced
    "balancing_factor": 3,

    'mask_features': True, #if True it will mask the features according to the logic in the function mask_features
    'censor_time': 'all_matched',#'pre_diagnosis', #'all_matched',#'first_and_last',#'successive','last_successive_and_previous',#'last_and_successive', #'all', 'pre_diagnosis', 'pre_diagnosis_1y', 'last_and_previous','last_and_successive'
    'filter_modality' : 'digit_original', #'X_original' 'digit_original' 'text_original'
}

params['list_of_ids_paths'], params['data_folder'], params['grid_dict_path'] = return_file_paths('PD', 
                                                                                                 False, False)

def get_type_of_input(params):
    out = None
    if 'feature_extraction' in params["input_file"]:
        out = "feature"
    else:
        out = "DL"
    return out
params["type_of_input"] = get_type_of_input(params)


VERBOSE = True
SAVE_PATH = "/home/a_morelli/vscode_projects/model_training/data/time_series_analysis"
os.makedirs(SAVE_PATH, exist_ok=True)

def main():
    start = time.time()

    #splits = ["train", "val", "test"]
    
    reference_df = pd.read_parquet(params['list_of_ids_paths'])
    reference_df = balance_data(reference_df, params, column="unique_id")

    df = load_file(params) 
    df = balance_data(df, params)

    if params["type_of_input"] == "feature": #i save in the same format as the representation file -> subj_id,q,mo,rep -> i can use the same logic
        df,properties = reshape_features(df, keep_cols=["split"])
        if params["mask_features"]:
            df = mask_features(df, params) #masking features for the representation
        df = add_info_to_df(df, params) #adding the class_col to the df
        #print("[DEBUG: Columns]",df.columns)
        #print(df.head())
        #returns a df with the following columns: ['subject_id', 'split', 'q', 'modality', 'rep', 'unique_id','diag_park_final1_quest'],
        #position in rep is given by the properties list, which is a list of tuples (prop_name, prop_value)
        print(properties)
        #assert 1==0
        df = df[df['split'] == 'val'] #only keep validation set for the analysis
        df = add_case_dt(params['signal_column'], df, reference_df, properties, n_slots=13) #adding the case_dt to the df
    else:
        params['signal_column']='probability_1'
        properties = None
    
    for n in [3, 5, 10, 20, 30]:
        plot_probability_trajectories(df, n_steps=n, ax=None,
                                    class_names=("negative", "positive"),
                                    min_count=1, save_path=SAVE_PATH, save_name=f"probability_trajectories_{n}", signal_column=params['signal_column'])

    assert 1==0

    save_dir = get_save_path(params)
    log_path = os.path.join(save_dir, f"stats.txt") #copy prints also to a log file in the checkpoint folder
    with tee_stdout(log_path):
        results_df, all_probs, all_labels, imp_df = process_representation(df, params, verbose=VERBOSE, prop_names=properties)

        analyze_results(all_probs, all_labels, results_df, split="validation",
                            pos_label=1, threshold=None, strategy="youden",
                            target_recall=0.90, plot=True, out_dir_path=save_dir)
        
        if imp_df is not None:
            per_prop = imp_df.groupby("prop")["importance"].sum().sort_values(ascending=False)
            per_key  = imp_df.groupby("key")["importance"].sum().sort_values(ascending=False)
            print(f"Feature importance per property:\n{per_prop}")
            print(f"Feature importance per key:\n{per_key}")
            imp_df.to_csv(os.path.join(save_dir, "feature_importance.csv"), index=False)
            print(f"Feature importance saved to {os.path.join(save_dir, 'feature_importance.csv')}")

        save_results(save_dir, params, results_df)

def add_case_dt(property_of_interest,result_df, train_df, properties, n_slots=13):
    """Attach case_dt to result_df by matching (unique_id, slot=i) to
    train_df's case_dt_dateq{i+1}. 
    Rename columns for using the trajectory analysis function"""
    cols = [f"case_dt_dateq{i}" for i in range(1, n_slots + 1)]

    long = (train_df[["unique_id", *cols]]
            .melt(id_vars="unique_id", value_vars=cols,
                var_name="_col", value_name="case_dt"))
    long["q"] = long["_col"].str.removeprefix("case_dt_dateq").astype(int) - 1
    #rename columns to match result_df
    long = long.rename(columns={"unique_id": "subject_id"})

    results_df = result_df.merge(long[["subject_id", "q", "case_dt"]],
                        on=["subject_id", "q"], how="left")
    
    results_df = results_df.rename(columns={"diag_park_final1_quest": "true_label"})

    #get index of the property of interest in the properties list
    prop_index = properties.index(property_of_interest) if property_of_interest in properties else None
    if prop_index is None:
        raise ValueError(f"Property {property_of_interest} not found in properties list")
    #build a column for that  property using the rep column
    results_df[property_of_interest] = results_df["rep"].apply(lambda x: x[prop_index])
    #drop the rep column
    results_df = results_df.drop(columns=["rep"])
    return results_df

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="I/O Benchmark for Multi-Tar Dataset")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for DataLoader")
    parser.add_argument("--batches_to_test", type=int, default=50, help="Number of batches to process for benchmark")
    return parser.parse_args()

def load_file(params):
    if params["type_of_input"] == "representation":
        pass
    elif params["type_of_input"] == "feature":
        df = pd.read_csv(params["input_file"])
    return df

'''def questionnaires_to_keep_wrapper(r, censor_time, original_data, sid, subject_images):
    questionnaire_info = {q: {'case_dt_dateq': r[f'case_dt_dateq{q}']} for q in range(1, 14)}
    keep = questionnaires_to_keep(r['last_avail_q'], censor_time, questionnaire_info,
                                  original_data, sid, subject_images,
                                  r['case_grid_pattern'], r['rempli_seulq12'])
    return keep'''

if __name__ == "__main__":
    main()
    