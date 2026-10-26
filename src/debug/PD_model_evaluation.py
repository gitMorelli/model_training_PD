import torch
import torchvision.transforms as T
import os
import pandas as pd
import glob
import lightning as L
import random
import sys
import pickle

import sys, os, contextlib


from src.utils.data_loading import prepare_exclusion_sets_PD, load_grid_dict, prepare_loaders_PD, return_file_paths, return_dt_path
from src.utils.image_processing import  get_mu_std
from src.utils.training_utils import PDClassificationModule, load_legacy_state_dict
from src.scripts.train_PD_model import model_initialization, preprocess_PD_csv
from src.utils.evaluation import analyze_results, plot_probability_trajectories


def get_last_best_checkpoint(checkpoint_dir,version):
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir,f"v_{version}", "*best*.ckpt"))
    checkpoint_to_load = max(checkpoint_files, key=os.path.getctime) if checkpoint_files else None
    checkpoint_name = os.path.basename(checkpoint_to_load) if checkpoint_to_load else None
    checkpoint_to_load=f"v_{version}/{checkpoint_name}" if checkpoint_name else None
    print(f"Checkpoint to load: {checkpoint_name}", flush=True)
    return checkpoint_to_load

experiment = "PD" #"pre_trained_models/E3N" # "PD"
SOURCE_PATH = f"/home/a_morelli/models/model_training_logs/{experiment}/"
model_name = 'resnet18_32'#'resnet18_32' #efficientnet_v2_s' #convnext_tiny'#'resnet50'#'FiveStageResidualStridedConvNet' #"FiveStageResidualStridedConvNet"
CHECKPOINT_PATH = f"/home/a_morelli/models/model_training_logs/{experiment}/{model_name}_model_results/checkpoints"
version='14'
override_parameters=False
old_run=False
params_path = os.path.join(CHECKPOINT_PATH,f"v_{version}", "exp_params.pkl")
#get the most recent ckpt file with best in the name 
checkpoint_to_load = get_last_best_checkpoint(CHECKPOINT_PATH,version)
#open and save as exp_params dict
with open(params_path, 'rb') as f:
    exp_params = pd.read_pickle(f) 

prefix=''
if override_parameters:
    print(f"Overriding parameters from {params_path}", flush=True)
    exp_params['censor_time'] = 'pre_diagnosis' #you can test models trained on all also on partial sequences
    prefix='pre_diagnosis'

#exp_params['filter_missing']='all'
#exp_params['censor_time']='all'

exp_params['predict_on_train'] = False
exp_params['balance_validation'] = False
exp_params['batch_size'] = 4
#'precision': "16-mixed",

if exp_params['pre_training']:
    exp_params['matched_validation'] = False
else:
    exp_params['matched_validation'] = True 

#PATHS
SOURCE_PATTERN = os.path.join(SOURCE_PATH,exp_params['data_folder'])
SHARD_PATTERN_val = os.path.join(SOURCE_PATTERN,"val/worker*_shard-*.tar")
SHARD_PATTERN_train = os.path.join(SOURCE_PATTERN,"train/worker*_shard-*.tar")

VERBOSE = False
CLASS_COL = exp_params['class_col'] 

if old_run:
    exp_params['list_of_ids_paths'], exp_params['data_folder'], exp_params['grid_dict_path'] = return_file_paths('PD', False, False)
    print(exp_params['custom_pre_trained_weights'],flush=True) 
    exp_params['custom_pre_trained_weights_old'] = exp_params['custom_pre_trained_weights']
    exp_params['custom_pre_trained_weights'] = None #why do i need this?

def main(exp_params):
    args = get_args()
    worker = args.num_workers
    prefetch_factor = 4 if worker > 0 else None

    #fix all the seeds for reproducibility 
    torch.manual_seed(exp_params['seed'])
    random.seed(exp_params['seed'])
    #with lightning 
    L.seed_everything(exp_params['seed'], workers=True) 

    #load grid_files for selecting chunks from the images during the dataloading
    grid_dict = load_grid_dict(exp_params)

    csv_data = pd.read_parquet(exp_params['list_of_ids_paths']) #if dataset is synthetic the list_of_ids_paths
    #is 

    exp_params['norm_mu'],exp_params['norm_std'] = get_mu_std(exp_params, verbose=VERBOSE)

    #exclude controls from the training if i want to reduce the asimmetry of the dataset (for example if i want to have a 1:1 ratio between cases and controls)
    exclusion_set, val_exclusion_set, counts = prepare_exclusion_sets_PD(exp_params,verbose=VERBOSE,class_col=CLASS_COL)

    #_,transform = get_model(name=exp_params['model'], pretrained=True)
    #transform = get_transforms(exp_params, transform)
    model, transform = model_initialization(None,exp_params,verbose=VERBOSE,val=True, **exp_params['model_parameters'])
    
    train_df = pd.read_parquet(exp_params['list_of_ids_paths'])
    val_exclusion_set = override_val_exclusion(train_df, val_exclusion_set, exp_params)
    train_df, *_ = preprocess_PD_csv(train_df,exp_params.get('selected_properties', None), split_col='split')

    train_loader,val_loader,_,_= prepare_loaders_PD(worker,prefetch_factor,exp_params,exclusion_set,val_exclusion_set, grid_dict, transform, 
                                                    SHARD_PATTERN_train=SHARD_PATTERN_train, SHARD_PATTERN_val=SHARD_PATTERN_val, train_df=train_df,
                                                    running_eval=True)
    
    if exp_params['matched_validation']:
        matched_val_loader = prepare_balanced_validation(worker,prefetch_factor,exp_params, grid_dict, transform)
    
    # 1. Gather predictions using the best checkpoint saved during training
    # Setting ckpt_path="best" tells Lightning to automatically find your top model
    ckpt_path=os.path.join(CHECKPOINT_PATH,checkpoint_to_load) 
    lit_model = litmodel_initialization_from_checkpoint(model, ckpt_path, exp_params)
    if prefix!='':
        save_path = os.path.join(os.path.dirname(ckpt_path), prefix)
    else:
        save_path = os.path.dirname(ckpt_path)

    tb_logger=False
    # 4. Initialize Trainer and Fit
    trainer = L.Trainer(
        max_epochs=1,
        logger = tb_logger,
        accelerator="auto" ,               # Automatically selects GPU/CPU/MPu
        #precision='bf16-mixed'
    )
    '''res= trainer.validate(lit_model, val_loader)
    print(res[0]['val/pr_auc'], flush=True)
    return'''

    has_subsets = callable(getattr(lit_model.model, 'get_modalities_names', None))
    if has_subsets:
        from itertools import combinations
        modalities_names = lit_model.model.get_modalities_names()
        print(f"Model supports subsets. Modalities names: {modalities_names}", flush=True)
        #compute the combinations of modalities to predict on
        modalities_combinations = [None]+ [list(c) for r in range(1, len(modalities_names) + 1) for c in combinations(modalities_names, r)]
        lit_model.predict_subsets = modalities_combinations
        print(f"Predicting on subsets: {lit_model.predict_subsets}", flush=True)
    
    outputs = trainer.predict(lit_model, dataloaders=val_loader)# ckpt_path=os.path.join(CHECKPOINT_PATH,"best.ckpt"))
    results = get_result_dfs_per_group(outputs, class_names=None)

    results_matched=None
    if exp_params['matched_validation']:
        print("#" * 50)
        print(f"Evaluating model on matched validation set using checkpoint: {ckpt_path}")
        outputs_matched = trainer.predict(lit_model, dataloaders=matched_val_loader)# ckpt_path=os.path.join(CHECKPOINT_PATH,"best.ckpt"))
        results_matched = get_result_dfs_per_group(outputs_matched, class_names=None)
    
    results_training=None
    if exp_params['predict_on_train']:
        outputs_training = trainer.predict(lit_model, dataloaders=train_loader)# ckpt_path=os.path.join(CHECKPOINT_PATH,"best.ckpt"))
        results_training = get_result_dfs_per_group(outputs_training, class_names=None)
    
    results_df_per_step=None
    if hasattr(lit_model, 'per_step') and lit_model.per_step: #the trained model returns predictions per step, i can aggregate those
        print("#" * 50)
        print(f"Evaluating model on per_step predictions")
        time_df_path = return_dt_path(exp_params['list_of_ids_paths'])
        time_df = pd.read_parquet(time_df_path)
        results_df_per_step = get_per_step_results_per_group(outputs, time_df, class_names=None)
    
    print(f"Evaluating model on validation set using checkpoint: {ckpt_path}")

    for k in results.keys():
        print(f"Processing results for group: {k}", flush=True)
        process_results_per_group(save_path, k, results, results_df_per_step, results_matched, results_training,
                                 exp_params, csv_data)

def process_results_per_group(save_path,group_name,results, results_df_per_step, results_matched, results_training
                              , exp_params, csv_data):
    if group_name == "all":
        log_path = os.path.join(save_path, f"stats.txt") #copy prints also to a log file in the checkpoint folder
    else:
        save_path = os.path.join(save_path, group_name)
        os.makedirs(save_path, exist_ok=True)
        log_path = os.path.join(save_path, f"stats.txt") #copy prints also to a log file in the checkpoint folder
    with tee_stdout(log_path):
        return_model_info(exp_params)

        results_df, all_probs, all_preds, all_labels = results[group_name]
        analyze_results(all_probs, all_labels, results_df, split="validation",
                        pos_label=1, threshold=None, strategy="youden",
                        target_recall=0.90, plot=True, out_dir_path=save_path)
        
        if results_matched is not None:
            results_df_matched, all_probs_matched, all_preds_matched, all_labels_matched = results_matched[group_name]
            analyze_results(all_probs_matched, all_labels_matched, results_df_matched, split="matched_validation",
                            pos_label=1, threshold=None, strategy="youden",
                            target_recall=0.90, plot=True, out_dir_path=save_path)
        
        if results_df_per_step is not None: #the trained model returns predictions per step, i can aggregate those
            results_to_process = results_df_per_step[group_name]
            plot_probability_trajectories(results_to_process, n_steps=20, ax=None,
                                  class_names=("negative", "positive"),
                                  min_count=1, save_path=save_path)
            #save the per_step results in a csv file
            results_df_per_step.to_csv(os.path.join(save_path, f"per_step_predictions.csv"), index=False)

        if results_training is not None:
            results_df_train, all_probs, all_preds, all_labels = results_training[group_name]
            analyze_results(all_preds, all_labels, results_df, split="train")

            #concatenate the result dataframes
            results_complete = pd.concat([results_df, results_df_train], ignore_index=True)
            results_df = results_complete.copy()
    
    store_results(csv_data, results_df, save_path,exp_params, group_name=group_name)

def get_per_step_results_per_group(outputs, train_df, class_names=None):
    """{subset_name: per-step DataFrame}; handles old and per-subset outputs."""
    if not outputs:
        return {}
    if "probs" in outputs[0]:                                  # old format
        return {"all": get_per_step_results(outputs, train_df, class_names)}
    return {name: get_per_step_results([b[name] for b in outputs], train_df, class_names)
            for name in outputs[0]}
def get_per_step_results(outputs, train_df,class_names=None):
    """Per-timestep predictions from a per_step=True run.

    Returns one row per (subject, slot), sorted subject-major then by slot.
    Empty if the run wasn't per-step (nothing to assemble).
    """
    if not outputs or "tok_probs" not in outputs[0]:
        return pd.DataFrame()

    tok_probs = torch.cat([b["tok_probs"] for b in outputs]).cpu().float().numpy()
    tok_preds = torch.cat([b["tok_preds"] for b in outputs]).cpu().float().numpy()

    # seq_ids index into each batch's subject_ids list, so map them per-batch
    # before concatenating — a global cat would collide across batches.
    subj, label, slot = [], [], []
    for b in outputs:
        sids = b["subject_ids"]                            # list, len = B_batch
        labels_b = b["labels"].cpu().numpy()               # (B_batch,)
        seq_b  = b["tok_seq_ids"].cpu().numpy()            # (n_tok_batch,)
        slot_b = b["tok_slot_ids"].cpu().numpy()
        subj.extend(sids[i] for i in seq_b)
        label.extend(labels_b[i] for i in seq_b)
        slot.extend(slot_b.tolist())

    n_classes = tok_probs.shape[1] 
    if class_names is None:
        class_names = list(range(n_classes))
    assert len(class_names) == n_classes

    df = pd.DataFrame({
        "unique_id":       subj,
        "slot":            slot,
        "true_label":      label,
        "predicted_label": tok_preds,
    })
    for i, name in enumerate(class_names):
        df[f"probability_{name}"] = tok_probs[:, i]
    
    def add_case_dt(result_df, train_df, n_slots=13):
        """Attach case_dt to result_df by matching (unique_id, slot=i) to
        train_df's case_dt_dateq{i+1}. Missing pairs get NaN."""
        cols = [f"case_dt_dateq{i}" for i in range(1, n_slots + 1)]

        long = (train_df[["unique_id", *cols]]
                .melt(id_vars="unique_id", value_vars=cols,
                    var_name="_col", value_name="case_dt"))
        long["slot"] = long["_col"].str.removeprefix("case_dt_dateq").astype(int) - 1

        return result_df.merge(long[["unique_id", "slot", "case_dt"]],
                            on=["unique_id", "slot"], how="left")
    
    if "tok_present" in outputs[0]:
        df["has_modality_data"] = torch.cat([b["tok_present"] for b in outputs]).numpy()
    result_df = add_case_dt(df, train_df)

    return result_df.sort_values(["unique_id", "slot"]).reset_index(drop=True)
def get_result_dfs_per_group(outputs, class_names=None):
    """Returns {subset_name: (results_df, all_probs, all_preds, all_labels)}.
    Handles both the old flat output and the per-subset output."""
    if "probs" in outputs[0]:                       # old format (predict_subsets = None)
        return {"all": get_result_df(outputs, class_names)}
    return {name: get_result_df([batch[name] for batch in outputs], class_names)
            for name in outputs[0]}
def get_result_df(outputs, class_names=None):
    all_probs = torch.cat([batch["probs"] for batch in outputs]).detach().cpu()
    all_preds = torch.cat([batch["preds"] for batch in outputs]).detach().cpu()
    all_labels = torch.cat([batch["labels"] for batch in outputs]).detach().cpu()
    all_subjects = [sid for batch in outputs for sid in batch["subject_ids"]]

    n_classes = all_probs.shape[1]
    if class_names is None:
        class_names = range(n_classes)
    assert len(class_names) == n_classes

    results_df = pd.DataFrame({
        "unique_id": all_subjects,
        "true_label": all_labels.numpy(),
        "predicted_label": all_preds.float().numpy(),
    })
    probs_np = all_probs.float().numpy()
    for i, name in enumerate(class_names):
        results_df[f"probability_{name}"] = probs_np[:, i]

    return results_df, all_probs, all_preds, all_labels


def override_val_exclusion(train_df, val_exclusion_set, exp_params):
    if exp_params['pre_training']:
        N=2000
        #reduce the number of samples in the validation set to 1000 for pre-training
        #identify the remaining ids after filtering for val_exclusion_set
        val_df = train_df[train_df['split']=='val']
        all_val_ids = set(val_df['unique_id'].unique())
        remaining_ids = set(val_df[~val_df['unique_id'].isin(val_exclusion_set)]['unique_id'].unique())
        #randomly sample N ids from the remaining ids
        sampled_ids = random.sample(list(remaining_ids), N)
        val_exclusion_set = all_val_ids - set(sampled_ids)
        print(f"Reduced the number of samples in the validation set from {len(all_val_ids)} to {N} for pre-training. Excluded {len(val_exclusion_set)} samples.", flush=True)
    return val_exclusion_set

def return_model_info(params):
    print(f"Model properties:")
    print(f"Model name -----> {params['model']}")
    if 'custom_pre_trained_weights_old' in params:
        print(f"Initial weights -----> {params['custom_pre_trained_weights_old']} (old run)")
    else:
        print(f"Initial weights -----> {params['custom_pre_trained_weights']}")
    print(f"Decoder structure -----> {params['model_structure']}")
    print(f"Fine tuning strategy -----> {params['layers_to_unfreeze']}")
    print(f"Data modality -----> {params['data_modality']}")
    print(f"Balancing strategy -----> factor={params['balancing_factor']}, balanced_data={params['balanced_data']}, weighted_loss={params['use_balanced_weights']}, balance_validation={params['balance_validation']}")
    print(f"Selected sequence -----> {params['censor_time']}")
    print('Training parameters:')
    print(f"  use_opt_groups: {params['use_opt_groups']}")
    print(f"  lr_backbone: {params['lr_backbone']}")
    print(f"  lr_classifier_head: {params['lr_classifier_head']}")
    print(f"  lr_scheduling: {params['lr_scheduling']}")
    #print(f"  batch_size: {params['batch_size']}")
    print(f"  num_epochs: {params['num_epochs']}")
    print(f"  patience: {params['patience']}")
    print(f"  eta_min_cosine: {params['eta_min_cosine']}")
    print(f"  weight_decay: {params['weight_decay']}")
    print(f"  warmup_fraction: {params['warmup_fraction']}")
    print(f"Head parameters: {params['model_parameters']}")

def prepare_balanced_validation(worker,prefetch_factor,exp_params, grid_dict, transform):
    exp_params_temp=exp_params.copy()
    exp_params_temp['balance_validation'] = True
    exp_params_temp['balancing_factor'] = 1.0

    train_df = pd.read_parquet(exp_params_temp['list_of_ids_paths'])
    exclusion_set, val_exclusion_set, counts = prepare_exclusion_sets_PD(exp_params_temp,verbose=VERBOSE,class_col=CLASS_COL, exclude_cases=True)
    train_df, *_ = preprocess_PD_csv(train_df,exp_params.get('selected_properties', None), split_col='split')

    _,val_loader,_,_= prepare_loaders_PD(worker,prefetch_factor,exp_params_temp,exclusion_set,val_exclusion_set, grid_dict, transform, 
                                                    SHARD_PATTERN_train=SHARD_PATTERN_train, SHARD_PATTERN_val=SHARD_PATTERN_val, train_df=train_df)
    return val_loader

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="I/O Benchmark for Multi-Tar Dataset")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for DataLoader")
    parser.add_argument("--batches_to_test", type=int, default=50, help="Number of batches to process for benchmark")
    return parser.parse_args()

class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()

@contextlib.contextmanager
def tee_stdout(path, mode="w"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        old = sys.stdout
        sys.stdout = _Tee(old, f)
        try:
            yield
        finally:
            sys.stdout = old


def store_results(csv_data, results_df, ckpt_path,params, group_name=None):
    merged_df = pd.merge(csv_data, results_df, on='unique_id', how='left')
    #check for duplicate rows
    if merged_df.duplicated(subset=['unique_id']).any():
        print("Warning: There are duplicate rows in the merged dataframe based on 'unique_id'.")
    else:
        print("No duplicate rows found in the merged dataframe based on 'unique_id'.")
    #save the merged dataframe in a csv file
    merged_df.to_csv(os.path.join(ckpt_path, f"predictions.csv"), index=False)
    #save params dict as predictions_metadata.pkl
    #save the exp_params dictionary to a pickle file in the checkpoint folder
    with open(os.path.join(ckpt_path, f"predictions_metadata.pkl"), 'wb') as f:
        pickle.dump(params, f)

def litmodel_initialization_from_checkpoint(model, ckpt_path, exp_params):
    if exp_params['grouped']:
        raise ValueError("Grouped models are not supported in this function. Please use the appropriate function for grouped models.")
    else:
        model_class=PDClassificationModule
    try:
        lit_model = model_class.load_from_checkpoint(ckpt_path, write_log=False, model=model, strict=False)
    except Exception as e:
        from src.utils.training_utils import DataShape
        print(f"Strict loading failed with error: {e}. Attempting to load with legacy")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        class_counts = (1,1) # Placeholder; replace with actual class counts if available
        lit_model = PDClassificationModule(model, class_counts, DataShape(total_units=1), num_classes=1)
        load_legacy_state_dict(lit_model, ckpt["state_dict"])
    return lit_model


if __name__ == "__main__":
    main(exp_params)
    