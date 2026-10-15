#Merging data from the full dataset 
def merge_properties_from_full_dataset_PD(exp_params, csv_data, properties_to_add, verbose=True):
    if 'full_dataset' not in exp_params:
        raise ValueError("The 'full_dataset' key is missing in exp_params. Specify it to load properties from that dataset.")
    full_dataset_path = exp_params['full_dataset']
    full_df = pd.read_csv(full_dataset_path, encoding='cp1252')

    if csv_data is None:
        csv_data = pd.read_parquet(exp_params['list_of_ids_paths'])
    
    if verbose:
        print("Lenght before merging:", len(csv_data))  
    # Extract ident_projet from "{ident_projet}_XXXX" by stripping the trailing suffix
    csv_data['_ident_projet'] = csv_data['unique_id'].str.rsplit('_', n=1).str[0]

    # Merge in the desired columns
    csv_data = csv_data.merge(
        full_df[['ident_projet'] + properties_to_add],
        left_on='_ident_projet',
        right_on='ident_projet',
        how='left'
    )

    # Clean up helper columns
    csv_data = csv_data.drop(columns=['_ident_projet'])
    if 'ident_projet' not in csv_data.columns:  # only drop if merge created a duplicate
        pass
    else:
        csv_data = csv_data.drop(columns=['ident_projet'])
    
    if verbose:
        print("Lenght after merging:", len(csv_data))  
        print("#" * 50)
    
    return csv_data

#preparing the pre-training dataset (csv)
def prepare_pre_training(df, data_selected_source):
    data_selected = data_selected_source.copy()
    def combine_avail_columns(df):
        cols = [f"q_{i}_avail" for i in range(1, 14)]
        df['avail_pattern'] = df[cols].astype(int).astype(str).agg(''.join, axis=1)
        df = df.drop(columns=cols)
        return df
    def combine_grid_columns(df):
        cols = [f"q_{i}_grid_file_avail" for i in range(1, 14)]
        df['grid_pattern'] = df[cols].astype(int).astype(str).agg(''.join, axis=1)
        df = df.drop(columns=cols)
        return df
    df = combine_avail_columns(df)
    df = combine_grid_columns(df)
    df = df[['grid_pattern', 'avail_pattern', 'ident_projet', 'split','rempli_seulq11','rempli_seulq12']]

    #create a ident_projet column from the unique_id column of data_selected; unique_id is in the form XXXX_YYYYY with XXXX the ident_projet
    data_selected['ident_projet'] = data_selected['unique_id'].str.split('_').str[0]
    #get the unique ident_projet from data_selected
    unique_ident_projet = data_selected['ident_projet'].unique()
    #exclude the rows in df that are in unique_ident_projet
    df = df[~df['ident_projet'].isin(unique_ident_projet)]
    #change the name of ident_projet to unique_id in df
    df = df.rename(columns={'ident_projet': 'unique_id'})

    return df