import argparse
import os
import gc

import numpy as np
import pandas as pd
from sklearn import *
from tqdm import tqdm
from pathlib import Path

def feature_atomtype(df, s):
    # https://www.kaggle.com/jazivxt/all-this-over-a-dog
    df['atom1'] = df['type'].map(lambda x: str(x)[2])
    lbl = preprocessing.LabelEncoder()
    for i in range(4):
        df['type'+str(i)] = lbl.fit_transform(df['type'].map(lambda x: str(x)[i]))

    df = pd.merge(train, s.rename(columns={'atom_index':'atom_index_0', 'x':'x0', 'y':'y0', 'z':'z0', 'atom':'atom1'}))
    df = pd.merge(train, s.rename(columns={'atom_index':'atom_index_1', 'x':'x0', 'y':'y0', 'z':'z0', 'atom':'atom1'}))
    return df

def reduce_mem_usage(df, verbose=True):
    # somewhere from kaggle kernel
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: 
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    
    return df

def save_df(path, df):
    print(f"saving df to {path}")
    if not path.parent.exists():
        os.system(f'mkdir -p {path.parent}')
    df.to_csv(path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create features for training')
    parser.add_argument('name', help='feature dataset name')
    parser.add_argument('--feature_dir', required=False, default='features', help='feature dataset directory')
    parser.add_argument('--subset', action='store_true', help='only use small subset for debug')
    args = parser.parse_args()

    PATH = Path('../data')

    if args.subset:
        train = pd.read_csv(PATH/'train.csv')[::10]
        test = pd.read_csv(PATH/'test.csv')[::10]
        print("using subset of data")
    else:
        train = pd.read_csv(PATH/'train.csv')
        test = pd.read_csv(PATH/'test.csv')
        print("using all of data")

    structures = pd.read_csv(PATH/'structures.csv')
    train = feature_atomtype(train, structures)
    test = feature_atomtype(test, structures)
    
    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)

    FEATURE = Path(args.feature_dir)
    suffix = '_subset' if not args.subset else ''
    save_df(FEATURE/f'train_{args.name}.csv', train)
    save_df(FEATURE/f'test_{args.name}.csv', test)
