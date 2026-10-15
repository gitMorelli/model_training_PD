import pandas as pd
import random


#Training samples selection
def melt_df(df,modality,threshold=1, questionnaires_to_include=None):
    if questionnaires_to_include is None:
        questionnaires_to_include = [str(q) for q in range(1,14)]
    exclusion_set = set()
    avail_columns=[f'q_{q}_num_{modality}' for q in questionnaires_to_include]
    df_source = df[['ident_projet', 'lateralite','split'] + avail_columns]
    df_long = df_source.melt(
        id_vars=['ident_projet', 'lateralite','split'], 
        value_vars=avail_columns,
        var_name='original_col', 
        value_name='score'
    )
    print(f"Length of melted df before filtering: {len(df_long)}")

    # 3. Extract the 'q' number from the column name
    # This regex looks for 'q_' followed by digits at the start of the string
    df_long['questionnaire'] = df_long['original_col'].str.extract(r'^q_(\d+)_').astype(int)

    df_long['ident_projet'] = df_long['ident_projet'].astype(str) + '_' + df_long['questionnaire'].astype(str)

    # 2. Filter rows where the score/value is >= 1
    df_filtered = df_long[df_long['score'] < threshold]
    ident_projets_to_exclude = set(df_filtered['ident_projet'].unique())
    df_long = df_long[df_long['score'] >= threshold]

    print(f"Length of melted df after filtering: {len(df_long)}")

    # 4. Drop the temporary columns to get your final desired structure
    new_df = df_long[['ident_projet', 'lateralite','split']].reset_index(drop=True)

    return new_df,ident_projets_to_exclude


# Class rebalancing / exclusion
def generate_exclusion_set_val(csv_data, data_modality, majority_class_id, balancing_factor, label_col='lateralite',id_col='ident_projet',split='train'):
    train_data = csv_data[csv_data['split'] == split]
    class_counts = train_data[label_col].value_counts()
    print(f"Class distribution in val set (after filtering for modality {data_modality}):\n{class_counts}")
    num_0 = class_counts.get(0.0, 0)  # Count of class 0 (e.g., right-handed)
    num_1 = class_counts.get(1.0, 0)  # Count of class 1 (e.g., left-handed)
    majority_class_ids = train_data[train_data[label_col] == majority_class_id][id_col].unique()
    # randomly select a fraction of those ids based on the balancing factor
    majority_ids_to_include = random.sample(list(majority_class_ids), min(int(num_1*balancing_factor), len(majority_class_ids)))
    exclusion_set = set(majority_class_ids) - set(majority_ids_to_include)
    return exclusion_set

def _exclude_subjects_by_pattern(exp_params, csv_data, split='train', verbose=False):
    #exclude the subjects for which grid_pattern or case_grid_pattern has a certain pattern (eg all 0 or 0 before last avail q or ...)
    if exp_params['filter_missing'] == 'all':
        csv_data['last_avail_q'] = 13 #set last avail q to 13 for all subjects to reuse the same code
    if verbose:
        print(f"Initial number of samples in {split} set: {len(csv_data)}")
    def prefix_has_one(pattern, n):
        return '1' in pattern[:int(n)]
    mask = csv_data.apply(
        lambda r: prefix_has_one(r['grid_pattern'], r['last_avail_q'])
                and prefix_has_one(r['case_grid_pattern'], r['last_avail_q']),
        axis=1
    )

    return mask

def generate_exclusion_set_PD(csv_source,exp_params,split='train', original_data=None, exclude_cases=False, 
                              class_col='diag_park_final1_quest', verbose =False):
    '''
    this funciton computes, for a specific split, all the ids that should be ignored during loading, accounting for all exclusion/inclusion
    conditions.

    If original_data=None we assume csv_source contains all the train/val ids and we 
    remove the ids for which we don't have data and optionally some controls to reduce the case/control ratio

    If original_data=None -> the dataset was pre-filtered externally -> we perform the same exclusion but also exclude
    all ids that were present in the full dataset but not in the pre-filtered one
    '''

    if original_data is None: #for compatibility with old code
        original_data = csv_source.copy()
    
    csv_data = csv_source.copy()
    csv_data = csv_data[csv_data['split'] == split]
    if verbose:
        print(f"[Prepare exclusion] [pre] Number of samples in {split} set: {len(csv_data)}")
    all_original_ids = set(original_data[original_data['split']==split]['unique_id'].unique())

    if exp_params['grouped']: #if grouped modality i want only one case in each group -> have to deal with the cases that are
        #also selected as controls
        #remove all rows with case_control==0 and diag_park_final1_quest==1
        if verbose:
            print("Removing all rows with case_control==0 and diag_park_final1_quest==1 to avoid having cases also as controls in grouped modality")
            print(f"Initial number of samples in {split} set: {len(csv_data)}")
        csv_data = csv_data[~((csv_data['case_control']==0) & (csv_data['diag_park_final1_quest']==1))]
        
        if verbose:
            print(f"Number of samples in {split} set after removing cases also as controls: {len(csv_data)}")
            print("-" * 50)

    #exclude the subjects for which grid_pattern or case_grid_pattern has a certain pattern (eg all 0 or 0 before last avail q or ...)
    mask = _exclude_subjects_by_pattern(exp_params, csv_data, split=split, verbose=verbose)

    #subjects_with_no_pre_avail_q_data = csv_data[~mask]['unique_id'].unique()
    csv_data = csv_data[mask]
    if verbose:
        print(f"Number of samples in {split} set after filtering for 000.. string: {len(csv_data)}")
    if 'case_control' in csv_data.columns:
        if verbose:
            print(f"Number of unique subjects with case_control==1 in {split} set after filtering for 000.. string: {csv_data[csv_data['case_control']==1]['unique_id'].nunique()}")

    if (split=='train' and exp_params['balanced_data']) or (split=='val' and exp_params['balance_validation']):
        csv_data['group_id'] = csv_data['unique_id'].str.split('_').str[1].astype(int)
        if verbose:
            print(f"Balancing the dataset for {split} set with balancing factor {exp_params['balancing_factor']})")
        #keep only the controls
        if exclude_cases:
            if verbose:
                print("Excluding cases from the controls for balancing")
            csv_data_controls = csv_data[(csv_data['case_control'] == 0) & (csv_data[class_col] == 0)]
        else:
            csv_data_controls = csv_data[csv_data['case_control'] == 0]
        csv_data_cases = csv_data[csv_data['case_control'] == 1]

        #keep balancing factor ids in each group for each case, sampling with priority from the at_least_warning==0 controls
        n_target = int(exp_params['balancing_factor'])
        seed = exp_params['seed']
        def pick(group):
            n = min(len(group), n_target)
            priority = group.loc[group['at_least_warning'] == 0, 'unique_id']
            rest     = group.loc[group['at_least_warning'] != 0, 'unique_id']

            if len(priority) >= n:
                chosen = priority.sample(n, random_state=seed)
            else:
                chosen = pd.concat([
                    priority,  # take all priority ids
                    rest.sample(n - len(priority), random_state=seed)  # fill remainder
                ])
            return chosen
        selected_ids = set(
            csv_data_controls.groupby('group_id', group_keys=False).apply(pick)
        )

        #keep only the selected controls
        csv_data_controls = csv_data_controls[csv_data_controls['unique_id'].isin(selected_ids)]
        #re-join cases and controls
        csv_data = pd.concat([csv_data_cases, csv_data_controls])
    
    ids_to_exclude = all_original_ids - set(csv_data['unique_id'].unique()) #i exclude all the ids that are not still in the df
    
    return ids_to_exclude

def generate_exclusion_set_hand(csv_source,exp_params,split='train', original_data=None, 
                              class_col='diag_park_final1_quest', verbose =False):

    if original_data is None: #for compatibility with old code
        original_data = csv_source.copy()
    
    csv_data = csv_source.copy()
    csv_data = csv_data[csv_data['split'] == split]
    if verbose:
        print(f"[Prepare exclusion] [pre] Number of samples in {split} set: {len(csv_data)}")
    #the shards will be put together -> i will always iterate on train+val ->
    #e.g. during validation i have to exclude all the ids from all the other splits part the ones i will select
    all_original_ids = set(original_data['unique_id'].unique())

    mask = _exclude_subjects_by_pattern(exp_params, csv_data, split=split, verbose=verbose)
    #subjects_with_no_pre_avail_q_data = csv_data[~mask]['unique_id'].unique()
    csv_data = csv_data[mask]
    if verbose:
        print(f"Number of samples in {split} set after filtering for 000.. string: {len(csv_data)}")
    
    if (split=='train' and exp_params['balanced_data']) or (split=='val' and exp_params['balance_validation']):
        if verbose:
            print(f"Balancing the dataset for {split} set with balancing factor {exp_params['balancing_factor']})")

        #keep balancing factor ids in each group for each case, sampling with priority from the at_least_warning==0 controls
        n_target = int(exp_params['balancing_factor'])*len(csv_data[csv_data[class_col] == 1]) #number of controls to keep = balancing_factor * number of cases
        seed = exp_params['seed']
        
        #select n_target ids in the class_col==0 group 
        selected_ids = set(
            csv_data[csv_data[class_col] == 0].sample(n=min(n_target, len(csv_data[csv_data[class_col] == 0])), random_state=seed)['unique_id']
        )

        #the ids to keep are the selected ones plus all the ids in class_col==1
        csv_data = csv_data[csv_data['unique_id'].isin(selected_ids) | (csv_data[class_col] == 1)]
    
    ids_to_exclude = all_original_ids - set(csv_data['unique_id'].unique()) #i exclude all the ids that are not still in the df
    
    return ids_to_exclude


def prepare_exclusion_sets_PD(exp_params,verbose=True,class_col='', pre_computed_csv=None, exclude_cases=False):
    '''returns the num_0 and num_1 in the training set after considering the exclusion -> the true class numerosity'''
    exclusion_set = set()
    val_exclusion_set = set()
    
    if pre_computed_csv is not None:
        original_data = pd.read_parquet(exp_params['list_of_ids_paths'])
        csv_data = pre_computed_csv.copy()
    else:
        original_data = pd.read_parquet(exp_params['list_of_ids_paths'])
        csv_data = original_data.copy()
    if verbose:
        print("[Prepare exclusion] [pre] Initial CSV data loaded. First row example:")
        '''for col in csv_data.columns:
            print(f"{col}: {csv_data[col].iloc[0]}")
        print('#' * 50)'''
        print("Unique subjects in the dataset:", csv_data['unique_id'].nunique())
        print("Unique subjects in training set:", csv_data[csv_data['split'] == 'train']['unique_id'].nunique())
        print("Unique subjects in validation set:", csv_data[csv_data['split'] == 'val']['unique_id'].nunique())
        print('#' * 50)
        if class_col in csv_data.columns:
            print("Class distribution in the entire dataset:\n", csv_data[class_col].value_counts())
            print("Class distribution in the training set:\n", csv_data[csv_data['split'] == 'train'][class_col].value_counts())
            print('#' * 50)

    if class_col in ['lateralite']:
        print("[DEBUG] Handedness specific balancing")
        exclusion_set = generate_exclusion_set_hand(csv_data,exp_params, split='train',original_data=original_data, 
                                                class_col=class_col, verbose=verbose) 
        val_exclusion_set = generate_exclusion_set_hand(csv_data,exp_params, split='val', original_data=original_data, 
                                                    class_col=class_col, verbose=verbose)
    else:
        exclusion_set = generate_exclusion_set_PD(csv_data,exp_params, split='train',original_data=original_data, exclude_cases=exclude_cases, 
                                                class_col=class_col, verbose=verbose) 
        val_exclusion_set = generate_exclusion_set_PD(csv_data,exp_params, split='val', original_data=original_data, exclude_cases=exclude_cases, 
                                                    class_col=class_col, verbose=verbose)
    
    if verbose:
        print(len(exclusion_set), "samples will be excluded from the training set to achieve balancing.")
        print('#' * 50)
    
    filtered_csv_data_train = csv_data[~csv_data['unique_id'].isin(exclusion_set)]
    train = filtered_csv_data_train[filtered_csv_data_train['split'] == 'train']
    #compute the number of samples for each class in the training set
    if class_col in train.columns:
        counts = (
            train[class_col]
            .value_counts()
            .reindex(range(train[class_col].max() + 1), fill_value=0)
            .sort_index()
        )
    else:
        counts = [0,0]  # Default counts if class_col is not present (eg when i iterate in debug mode on pre_training dataset)

    if verbose:
        print("[Prepare exclusion] [post] After applying the exclusion set, the training set has:")
        print(f"Class 0: {counts[0]} samples")
        print(f"Class 1: {counts[1]} samples")
        print(f"Ratio of Class 1 to Class 0: {counts[1] / counts[0] if counts[0] > 0 else 'undefined'}")

    return exclusion_set, val_exclusion_set, counts

def prepare_test_exclusion_set(exp_params,verbose=True,class_col='', pre_computed_csv=None, exclude_cases=False):
    '''returns the num_0 and num_1 in the training set after considering the exclusion -> the true class numerosity'''
    exclusion_set = set()
    
    if pre_computed_csv is not None:
        original_data = pd.read_parquet(exp_params['list_of_ids_paths'])
        csv_data = pre_computed_csv.copy()
    else:
        original_data = pd.read_parquet(exp_params['list_of_ids_paths'])
        csv_data = original_data.copy()
    if verbose:
        print("Initial CSV data loaded. First row example:")
        '''for col in csv_data.columns:
            print(f"{col}: {csv_data[col].iloc[0]}")
        print('#' * 50)'''
        print("Unique subjects in the dataset:", csv_data['unique_id'].nunique())
        print("Unique subjects in test set:", csv_data[csv_data['split'] == 'test']['unique_id'].nunique())
        print('#' * 50)
        if class_col in csv_data.columns:
            print("Class distribution in the entire dataset:\n", csv_data[class_col].value_counts())
            print("Class distribution in the test set:\n", csv_data[csv_data['split'] == 'test'][class_col].value_counts())
            print('#' * 50)

    
    exclusion_set = generate_exclusion_set_PD(csv_data,exp_params, split='test',original_data=original_data, exclude_cases=exclude_cases, 
                                              class_col=class_col) 
    
    if verbose:
        print(len(exclusion_set), "samples will be excluded from the test set to achieve balancing.")
        print('#' * 50)
    
    filtered_csv_data_train = csv_data[~csv_data['unique_id'].isin(exclusion_set)]
    train = filtered_csv_data_train[filtered_csv_data_train['split'] == 'test']
    #compute the number of samples for each class in the training set
    if class_col in train.columns:
        counts = (
            train[class_col]
            .value_counts()
            .reindex(range(train[class_col].max() + 1), fill_value=0)
            .sort_index()
        )
    else:
        counts = [0,0]  # Default counts if class_col is not present (eg when i iterate in debug mode on pre_training dataset)

    if verbose:
        print("After applying the exclusion set, the test set has:")
        print(f"Class 0: {counts[0]} samples")
        print(f"Class 1: {counts[1]} samples")
        print(f"Ratio of Class 1 to Class 0: {counts[1] / counts[0] if counts[0] > 0 else 'undefined'}")

    return exclusion_set, counts

 