import argparse
import os
import gc

import numpy as np
import pandas as pd
from sklearn import *
from tqdm import tqdm
from pathlib import Path

from utils import *

PATH = Path('../data')

def feature_atomtype(df, s):
    # https://www.kaggle.com/jazivxt/all-this-over-a-dog
    df['atom1'] = df['type'].map(lambda x: str(x)[2])
    df['atom2'] = df['type'].map(lambda x: str(x)[3])
    lbl = preprocessing.LabelEncoder()
    for i in range(4):
        df['type'+str(i)] = lbl.fit_transform(df['type'].map(lambda x: str(x)[i]))

    df = pd.merge(df, s.rename(columns={'atom_index':'atom_index_0', 'x':'x0', 'y':'y0', 'z':'z0', 'atom':'atom1'}), how='left', on=['molecule_name', 'atom_index_0', 'atom1'])
    df = pd.merge(df, s.rename(columns={'atom_index':'atom_index_1', 'x':'x0', 'y':'y0', 'z':'z0', 'atom':'atom2'}), how='left', on=['molecule_name', 'atom_index_1', 'atom2'])
    return df

def feature_basic(df):
    structures = pd.read_csv(PATH/'structures.csv')
    df = feature_atomtype(df, structures)
    df = reduce_mem_usage(df)
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create features for training')
    parser.add_argument('name', help='feature dataset name')
    parser.add_argument('--feature_dir', required=False, default='features', help='feature dataset directory')
    parser.add_argument('--subset', action='store_true', help='only use small subset for debug')
    args = parser.parse_args()

    if args.subset:
        train = pd.read_csv(PATH/'train.csv')[::10]
        test = pd.read_csv(PATH/'test.csv')[::10]
        print("using subset of data")
    else:
        train = pd.read_csv(PATH/'train.csv')
        test = pd.read_csv(PATH/'test.csv')
    print(f"[{len(train)} rows] for train")
    print(f"[{len(test)} rows] for test")

    train = feature_basic(train)
    test = feature_basic(test)
    
    FEATURE = Path(args.feature_dir)
    suffix = '_subset' if args.subset else ''
    save_df(FEATURE/f'train_{args.name}{suffix}.csv', train)
    save_df(FEATURE/f'test_{args.name}{suffix}.csv', test)
