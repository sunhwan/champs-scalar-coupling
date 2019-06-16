import argparse
import os, sys
import gc

import numpy as np
import pandas as pd
from sklearn import *
from tqdm import tqdm
from pathlib import Path

from utils import *

PATH = Path('../data')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create features for training')
    parser.add_argument('name', help='experiment name')
    parser.add_argument('--feature', default='basic', help='feature')
    parser.add_argument('--feature_dir', default='features', help='feature dataset directory')
    parser.add_argument('--subset', action='store_true', help='only use small subset for debug')
    parser.add_argument('--model', choices=['ExtraTreesRegressor', 'LightGBM'], default='ExtraTreesRegressor', help='choice for model')
    parser.add_argument('--test', action='store_true', help='test with training set')
    parser.add_argument('--testonly', action='store_true', help='test and exit')
    parser.add_argument('--submission_dir', default='submissions', help='feature dataset directory')
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

    FEATURE = Path(args.feature_dir)
    suffix = '_subset' if args.subset else ''
    train = pd.merge(train, pd.read_csv(FEATURE/f'train_{args.feature}{suffix}.csv'))
    test = pd.merge(test, pd.read_csv(FEATURE/f'test_{args.feature}{suffix}.csv'))

    train.drop(columns=['id', 'molecule_name', 'atom1', 'atom2', 'atom_index_0', 'atom_index_1'], inplace=True)
    test.drop(columns=['molecule_name', 'atom1', 'atom2', 'atom_index_0', 'atom_index_1'], inplace=True)
    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)
    gc.collect()

    col = [c for c in train.columns if c not in ['scalar_coupling_constant']]
    print("columns used for training:", col)

    if args.model == 'ExtraTreesRegressor':
        model = ensemble.ExtraTreesRegressor(n_jobs=-1, n_estimators=20, random_state=4, verbose=1)

    if args.test:
        X_train, X_test, y_train, y_test = model_selection.train_test_split(train[col], train['scalar_coupling_constant'], test_size=0.2)
        model.fit(X_train.drop(['type'], axis=1), y_train)
        y_pred = model.predict(X_test.drop('type', axis=1))
        score = group_mean_log_mae(y_test, y_pred, X_test.type)
        print(f"test performance: {score}")

    if args.testonly:
        sys.exit()
    gc.collect()

    model.fit(train[col].drop(['type'], axis=1), train['scalar_coupling_constant'])
    y_pred = reg.predict(test.drop(['id', 'type'], axis=1))
    test['scalar_coupling_constant']  = y_pred

    SUBMISSION = Path(args.submission_dir)
    test[['id', 'scalar_coupling_constant']].to_csv(SUBMISSION/f'submission_{args.name}{suffix}.csv', index=False) #float_format='%.9f'

