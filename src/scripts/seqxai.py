"""Single-file entry point: model, data and settings all live here.

    python main.py                                   # all stages
    python main.py --stages spatial,report           # re-run some stages
    python main.py --predictions other.csv --out xai_run2 --device cuda:1

Fill in the TODO blocks; nothing else needs a separate setup file.
"""
import argparse
import os
import shutil
import pandas as pd

import numpy as np
import torch

from src.utils.data_loading import prepare_exclusion_sets_PD, load_grid_dict
from src.utils.image_processing import  get_mu_std
from src.utils.seqxai.seqxai import STAGES, Config, Pipeline, SelectionRule, SequenceSample, TorchDatasetAdapter
from src.utils.file_utils import get_last_best_checkpoint
from src.scripts.train_PD_model import model_initialization
from src.debug.PD_model_evaluation import litmodel_initialization_from_checkpoint
from src.utils.data_loading.map_style_dataset.pd_indexed import build_pd_dataset, make_pd_collate, make_seqxai_adapter
from src.utils.evaluation.analysis import get_threshold_from_results_df


VERBOSE = True

# ---------------------------------------------------------------- run
def main():
    delete_folder = False
    cl_args = get_args()
    args = {
        'workers': cl_args.num_workers,
        'model_name': 'convnext_tiny',
        'version': '27',
        'test_on': 'pre', # or 'all'

        'strategy_threshold': 'youden',  # or 'youden' or 'target_sensitivity'
        'threshold_target': 0.9,  # used only if strategy_threshold == 'target_sensitivity'
        'fixed_threshold': None,  # used only if strategy_threshold == 'fixed'
    }
    args['checkpoint_dir'] = f"/home/a_morelli/models/model_training_logs/PD/{args['model_name']}_model_results/checkpoints"
    args['checkpoint_file'] = os.path.join(args['checkpoint_dir'],get_last_best_checkpoint(args['checkpoint_dir'], args['version']))
    suffix_pre_all=''
    if args['test_on'] == 'pre':
        suffix_pre_all='_pre'
    args['prediction_file'] = os.path.join(args['checkpoint_dir'],f"v_{args['version']}", "all", f"predictions{suffix_pre_all}.csv")
    args['out_dir'] = os.path.join(args['checkpoint_dir'],f"v_{args['version']}", f"xai_results{suffix_pre_all}")
    if delete_folder:
        shutil.rmtree(args['out_dir'], ignore_errors=True)
    os.makedirs(args['out_dir'], exist_ok=True)
    args['stages'] = STAGES
    args['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    params_path = os.path.join(args['checkpoint_dir'],f"v_{args['version']}", "exp_params.pkl")
    with open(params_path, 'rb') as f:
        exp_params = pd.read_pickle(f) 
    SHARD_PATTERN_val = os.path.join(exp_params['data_folder'],"val/worker*_shard-*.tar")
    SHARD_PATTERN_train = os.path.join(exp_params['data_folder'],"train/worker*_shard-*.tar")
    args['index_csv']= '/home/a_morelli/datasets/id_lists/ids_to_shards/final_png_whitebg_21_07_26_id_to_shard_offsets.csv'
    VAL_PREFIX = 'val/'     # the folder of your validation shards, relative to data_folder path

    print(f"[INFO] Model was trained with censor time: {exp_params['censor_time']}",
          f"[INFO] XAI evaluation is made on {args['test_on']}",
          f"Expect mismateches if they differ.", flush=True)
    #load grid_files for selecting chunks from the images during the dataloading
    grid_dict = load_grid_dict(exp_params)
    csv_data = pd.read_parquet(exp_params['list_of_ids_paths']) #if dataset is synthetic the list_of_ids_paths
    exp_params['norm_mu'],exp_params['norm_std'] = get_mu_std(exp_params, verbose=VERBOSE)

    model, transform = model_initialization(None,exp_params,verbose=VERBOSE,val=True, **exp_params['model_parameters'])

    adapter = build_adapter(args,exp_params, transform, grid_dict, set(), csv_data,
                            args['prediction_file'], id_col='unique_id', val_prefix=VAL_PREFIX)
    #i can leave set empty because i will call only the data i have in the predictions csv 
    
    args['threshold'] = get_threshold_from_results_df(args['prediction_file'], split=VAL_PREFIX[:-1], 
                                                      strategy=args['strategy_threshold'], target=args['threshold_target'], 
                                                      pos_label=1, threshold=args['fixed_threshold'])
    print(f"[INFO] Using threshold: {args['threshold']}", flush=True)
    pipe = Pipeline(build_model(model, exp_params, args), adapter, args['prediction_file'],
                    build_config(args))
    pipe.run(args['stages'])


# ---------------------------------------------------------------- model
def build_model(model,exp_params, args):
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    lit_model = litmodel_initialization_from_checkpoint(model, args['checkpoint_file'], exp_params)
    net = lit_model.model                               # attribute name inside PDClassificationModule
    net.eval()

    # strict=False and the legacy path can both skip weights silently: list what stayed at init
    after = net.state_dict()
    unchanged = [k for k, v in before.items()
                 if v.is_floating_point() and torch.equal(v, after[k].cpu())]
    if unchanged:
        print(f'[warning] {len(unchanged)}/{len(before)} tensors still equal their random init, e.g.:')
        print('   ', unchanged[:10])
    return net


# ---------------------------------------------------------------- data
def build_adapter(args,exp_params, transform, grid_dict, val_exclusion_set, train_df,
                  predictions_csv, id_col, val_prefix):
    # serve exactly the subjects that have a prediction
    predictions_df = pd.read_csv(predictions_csv, dtype={id_col: str})
    split_data = predictions_df[predictions_df['split'] == val_prefix[:-1]]  # remove trailing slash
    #remove rows with predicted_label NaN and print a warning if any were dropped
    if split_data['predicted_label'].isna().any():
        num_dropped = split_data['predicted_label'].isna().sum()
        print(f"[warning] Dropping {num_dropped} rows with NaN predicted_label from {predictions_csv}")
    split_data = split_data.dropna(subset=['predicted_label'])
    ids = (split_data[id_col].drop_duplicates().tolist())

    ds = build_pd_dataset(args['index_csv'], exp_params['data_folder'], exp_params, transform, grid_dict,
                          exclusion_set=val_exclusion_set, train_df=train_df,
                          shard_prefix=val_prefix, ids=ids)
    collate = make_pd_collate(exp_params)

    def to_display(img):
        # TODO: check one figure; this assumes input_range='ink_pm1' with ink = +1
        x = img.detach().float().cpu().numpy()
        x = x[0] if x.shape[0] == 1 else x.transpose(1, 2, 0) if x.shape[0] == 3 else x.mean(0)
        return np.clip(1 - (x + 1) / 2, 0, 1)      # ink dark on white

    return make_seqxai_adapter(ds, collate, to_display=to_display)

# ---------------------------------------------------------------- settings
def build_config(args):
    return Config(
        out_dir=args['out_dir'],
        device=args['device'],
        id_col='unique_id',          
        label_col='true_label',
        prob_cols='probability_1',             # binary: P(class 1); multiclass: ['p0', 'p1', ...]
        threshold=args['threshold'],
        target_layer=None,            # e.g. 'layer4'; None = last Conv2d of vision_model
        pixel_baseline='blur',        # or the normalized value of a blank page
        selection=SelectionRule(n_per_cell=8, n_random=20, n_clusters=8),
        num_workers=args['workers'],
        # metadata_csv='subjects_meta.csv',
        # concepts_csv='concepts.csv',
    )

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="I/O Benchmark for Multi-Tar Dataset")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of DataLoader workers")
    return parser.parse_args()

if __name__ == '__main__':
    main()