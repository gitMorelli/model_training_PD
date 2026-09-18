import numpy as np
import pandas as pd
import warnings

from sklearn.preprocessing import OneHotEncoder, StandardScaler

_MAP_COLUMNS = {
    'etudegp':'one_hot',
    'profq2': 'one_hot',
    'lateralite': 'one_hot',
    'relative_age': 'normalize',
    'case_dt_dateq*': 'normalize',
}

def preprocess_PD_csv(df, columns, split_col='split'):
    '''
    take col names with the naming convention used for the model conditioning
    authomatically maps to which are to one hot and which to normalize
    performs the pre_processing
    
    if columns=[] or you have only one modalit of columns the code returns don't crash but returns the df as is, with no ohe or scaler

    '''
    
    ohe_cols = []
    num_cols = []
    if columns is None:
        return df, None, None, {}
    for col in columns:
        col_type = _MAP_COLUMNS.get(col, None)
        col_list=[]
        if '*' in col:
            for i in range(1, 14):
                col_list.append(col.replace('*', str(i)))
        else:
            col_list.append(col)
        if col_type == 'one_hot':
            ohe_cols.extend(col_list)
        elif col_type == 'normalize':
            num_cols.extend(col_list)

    df = df.copy()
    train_mask = df[split_col] == 'train'

    # ---------- One-hot encoding ----------
    ohe, ohe_mapping = None, {}
    ohe_df = pd.DataFrame(index=df.index)
    if ohe_cols:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        ohe.fit(df.loc[train_mask, ohe_cols])
        ohe_names = [f'_{col}_{i}'
                     for col, cats in zip(ohe_cols, ohe.categories_)
                     for i in range(len(cats))]
        ohe_df = pd.DataFrame(ohe.transform(df[ohe_cols]),
                              columns=ohe_names, index=df.index).astype(int)
        ohe_mapping = {f'_{col}_{i}': cat
                       for col, cats in zip(ohe_cols, ohe.categories_)
                       for i, cat in enumerate(cats)}

    # ---------- Normalization ----------
    scaler = None
    num_df = pd.DataFrame(index=df.index)
    if num_cols:
        num = df[num_cols].copy()
        for c in num_cols:
            if pd.api.types.is_datetime64_any_dtype(num[c]):
                num[c] = (num[c] - pd.Timestamp('1970-01-01')).dt.days
        scaler = StandardScaler()
        scaler.fit(num.loc[train_mask])
        num_df = pd.DataFrame(scaler.transform(num),
                              columns=[f'_{c}_0' for c in num_cols], index=df.index)

    df = pd.concat([df, ohe_df, num_df], axis=1)
    return df, ohe, scaler, ohe_mapping
