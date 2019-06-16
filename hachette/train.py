import argparse
import os, sys
import gc
import pickle
import random

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
    parser.add_argument('--model_dir', default='models', help='model save directory')
    parser.add_argument('--test', action='store_true', help='test with training set')
    parser.add_argument('--testonly', action='store_true', help='test and exit')
    parser.add_argument('--submission_dir', default='submissions', help='feature dataset directory')
    args = parser.parse_args()

    FEATURE = Path(args.feature_dir)
    suffix = '_subset' if args.subset else ''
    if args.subset:
        train = pd.read_csv(FEATURE/f'train_basic.csv')[::10]
        test = pd.read_csv(FEATURE/f'test_basic.csv')[::10]
    else:
        train = pd.read_csv(FEATURE/f'train_basic.csv')
        test = pd.read_csv(FEATURE/f'test_basic.csv')
    print(f"[{len(train)} rows] for train")
    print(f"[{len(test)} rows] for test")

    train.drop(columns=['id', 'molecule_name', 'atom1', 'atom2', 'atom_index_0', 'atom_index_1'], inplace=True)
    test.drop(columns=['molecule_name', 'atom1', 'atom2', 'atom_index_0', 'atom_index_1'], inplace=True)
    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)
    gc.collect()

    col = [c for c in train.columns if c not in ['scalar_coupling_constant']]
    print("columns used for training:", col)

    MODEL = Path(args.model_dir)
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
        pickle.dump(model, open(MODEL/f'etr_model_{args.name}{suffix}.pkl', 'wb'))

        y_pred = reg.predict(test.drop(['id', 'type'], axis=1))
        test['scalar_coupling_constant']  = y_pred
        SUBMISSION = Path(args.submission_dir)
        test[['id', 'scalar_coupling_constant']].to_csv(SUBMISSION/f'submission_{args.name}{suffix}.csv', index=False) #float_format='%.9f'

    if args.model == 'LightGBM':
        import lightgbm as lgb

        # https://github.com/selimsef/dsb2018_topcoders/blob/master/victor/train_classifier.py
        num_split_iters = 50
        folds_count = 3
        gbm_models = []

        for it in range(num_split_iters):
            kf = model_selection.KFold(n_splits=folds_count, random_state=it+1, shuffle=True)
            it2 = -1
            for train_index, valid_index in kf.split(train):
                it2 += 1

                random.seed(it*1000+it2)
                np.random.seed(it*1000+it2)

                lr = random.random()*0.1 + 0.02
                ff = random.random()*0.5 + 0.5
                nl = random.randint(6, 50)            
                print('training lgb', it, it2, 'lr:', lr, 'ff:', ff, 'nl:', nl)

                X_train, X_valid = train[col].iloc[train_index], train[col].iloc[valid_index]
                y_train, y_valid = train.scalar_coupling_constant.iloc[train_index], \
                                   train.scalar_coupling_constant.iloc[valid_index]

                col2 = [_ for _ in col if _ != 'type']
                lgb_train = lgb.Dataset(X_train[col2], y_train)
                lgb_eval = lgb.Dataset(X_valid[col2], y_valid, reference=lgb_train)
            
                params = {
                    'task': 'train',
                    'boosting_type': 'gbdt',
                    'objective': 'regression',
                    'metric': {'l2'},
                    'num_leaves': nl,
                    'learning_rate': lr,
                    'feature_fraction': ff,
                    'bagging_fraction': 0.95,
                    'bagging_freq': 1,
                    'verbosity': 0,
                    'metric_freq': 100,
                    'seed': it*1000+it2,
                    'num_threads': -1
                }

                gbm = lgb.train(params,
                                lgb_train,
                                num_boost_round=400,
                                valid_sets=lgb_eval,
                                verbose_eval=False,
                                early_stopping_rounds=5)
                
                gbm.free_dataset()
                gbm_models.append(gbm)
                gbm.save_model(str(MODEL/'gbm_model_{0}_{1}.txt'.format(it, it2)))

                y_pred = gbm.predict(X_valid[col2])
                score = group_mean_log_mae(y_valid, y_pred, X_valid.type)
                print(f"test performance: {score}")

    
