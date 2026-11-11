import time
from tkinter.font import names
import os
import pandas as pd
import time
import numpy as np 
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Sequence
import contextlib
import io

from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm

from src.utils.data_loading import prepare_exclusion_sets_PD, return_file_paths, return_dt_path
from src.utils.evaluation import  analyze_results, plot_probability_trajectories, roc_auc_over_time
from src.utils.evaluation.analysis import pick_threshold
from src.scripts.t_on_representations import balance_data, reshape_features, mask_features, add_info_to_df
from src.utils.evaluation.helpers import build_modality_names
from src.utils.longitudinal_analysis import run_discrimination
from src.utils.longitudinal_analysis import run_analysis, compare_models, compare_range
from src.utils.evaluation.binned_metrics import binned_metrics, plot_binned_metrics
from src.utils.file_utils import tee_stdout

MAP_PREFIXES = {
    'pre': 'pre',
    'all': '',
}

HOME_PATH = "/home/a_morelli/models/model_training_logs/PD/"
def get_input_file_path(name):
    if name=="FE":
        input_file = os.path.join(HOME_PATH, "feature_extraction/17092026/trained_models/xgb/v_13")
    elif name=="deep":
        input_file = os.path.join(HOME_PATH, "convnext_tiny_model_results/checkpoints/v_27")
    return input_file

params = {
    #input info
    "input_file":get_input_file_path("deep"), 
    "force_case_dt": True, #if True it will force the case_dt to the value in the time_df, if False it will use the case_dt in the input_file
    "per_modality": ['all'], #if "all_combinations" combinations are built
    #,'X_window','digit_window','text_window'
    "per_censoring" : ['pre'],#pre, all, * (if * it will repeat results on all censoring options) 
    "thresholds" : {
        'youden':None,
        'target_sensitivity':0.9,
    },
    #this is used if the file path is a directory -> i have per modality results
    "split_to_examine":'val',
    "set_threshold": 'youden',
    
}
VERBOSE = True
EXPERIMENT = "full_analysis" #"time_resolved_discrimination","statistical_analysis" #"time_series_comparison" #"plot_property" , 
#binned_metrics, 

def main(path=None):

    df = pd.read_csv(path)
    #has all columns of reference_df plus the one added during the evaluation 
    #true_label, predicted_label, probability_0, probability_1, has_modality_data, case_dt (case_dt of the last_avail_q before diagnosis)

    df['set'] = df['unique_id'].str.split('_').str[1]
    #convert the set column to int
    df['set'] = df['set'].astype(int)

    #select split to consider
    df = df[df['split']==params['split_to_examine']]
    #count the rows with missing values in the predicted_label column for the two 'true_label' classes
    missing_predicted_label = df[df['predicted_label'].isna()].groupby('true_label').size()
    print(f"Missing predicted_label counts by true_label:\n{missing_predicted_label}")
    #drop rows with missing predicted_label values
    df = df.dropna(subset=['predicted_label'])

    #get the probabilities and labels
    all_probs = df[['probability_0', 'probability_1']].values
    all_labels = df['true_label'].values

    subfolder = 'posthoc_analysis'
    if path is not None:
        local_save_path = os.path.join(os.path.dirname(path), subfolder)
    else:
        local_save_path = os.path.join(os.path.dirname(params['input_file']), subfolder)

    all_thresholds = params['thresholds']
    for t in all_thresholds:
        value = all_thresholds[t]
        save_path = os.path.join(local_save_path,t)
        log_path = os.path.join(save_path,'stats.txt')

        if EXPERIMENT == "full_analysis":
            with tee_stdout(log_path):
                selected_threshold = pick_threshold(y_true, y_scores, strategy="f1", target_recall=0.90,
                    target_sensitivity=0.90, target_specificity=0.90, pos_label=1)
                analyze_results(all_probs, all_labels, df, split=params['split_to_examine'],threshold=, plot=True, 
                                out_dir_path=save_path)





def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="I/O Benchmark for Multi-Tar Dataset")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for DataLoader")
    parser.add_argument("--batches_to_test", type=int, default=50, help="Number of batches to process for benchmark")
    return parser.parse_args()


if __name__ == "__main__":
    #check if params['input_file'] is a directory or a file
    is_directory = os.path.isdir(params['input_file'])
    if is_directory:
        if params['per_modality'] == 'all_combinations':
            modalities = build_modality_names(params['filter_modality'])
        else:
            modalities = params['per_modality']
        if params['per_censoring'] == '*':
            censoring_options = ['pre', 'all']
        else:
            censoring_options = params['per_censoring']
        for m in modalities:
            for c in censoring_options:
                print(f"Running analysis for modality: {m}, censoring: {c}")
                path = os.path.join(params['input_file'], m, f"predictions_{MAP_PREFIXES.get(c, '')}.csv")
                main(path=path)
    else:
        main()
    