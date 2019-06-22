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
n_cpu = 6

def feature_atomtype(df, s):
    # https://www.kaggle.com/jazivxt/all-this-over-a-dog
    df['atom1'] = df['type'].map(lambda x: str(x)[2])
    df['atom2'] = df['type'].map(lambda x: str(x)[3])
    lbl = preprocessing.LabelEncoder()
    for i in range(4):
        df['type'+str(i)] = lbl.fit_transform(df['type'].map(lambda x: str(x)[i]))

    df = pd.merge(df, s.rename(columns={'atom_index':'atom_index_0', 'x':'x0', 'y':'y0', 'z':'z0', 'atom':'atom1'}), how='left', on=['molecule_name', 'atom_index_0', 'atom1'])
    df = pd.merge(df, s.rename(columns={'atom_index':'atom_index_1', 'x':'x1', 'y':'y1', 'z':'z1', 'atom':'atom2'}), how='left', on=['molecule_name', 'atom_index_1', 'atom2'])
    return df

def feature_pair_geometry(df):
    p0 = df[['x0', 'y0', 'z0']].values
    p1 = df[['x1', 'y1', 'z1']].values
    r = np.linalg.norm(p0 - p1, axis=1)
    df['dist'] = r

    for agg in ['min', 'max', 'mean']:
        tmp = eval('df.groupby(["type"], as_index=False).dist.' + agg + '()')
        tmp.rename(columns={"dist":agg + "_dist"}, inplace=True)
        df = pd.merge(df, tmp, how='left', on=['type'])
    return df

# fix atom bonds
# dsgdb9nsd_059827: hydrogen has is far apart
nblist = {
    'dsgdb9nsd_059827': {
        13: 3
    }
}

def _feature_atom(atom):
    prop = {}
    nb = [a.GetSymbol() for a in atom.GetNeighbors()] # neighbor atom type symbols
    nb_h = sum([_ == 'H' for _ in nb])
    nb_o = sum([_ == 'O' for _ in nb])
    nb_c = sum([_ == 'C' for _ in nb])
    nb_n = sum([_ == 'N' for _ in nb])
    nb_na = len(nb) - nb_h - nb_o - nb_n - nb_c
    prop['degree'] = atom.GetDegree()
    prop['hybridization'] = int(atom.GetHybridization())
    prop['inring'] = int(atom.IsInRing())
    prop['inring4'] = int(atom.IsInRingSize(4))
    prop['inring5'] = int(atom.IsInRingSize(5))
    prop['inring6'] = int(atom.IsInRingSize(6))
    prop['nb_h'] = nb_h
    prop['nb_o'] = nb_o
    prop['nb_c'] = nb_c
    prop['nb_n'] = nb_n
    prop['nb_na'] = nb_na
    return prop

def _feature_neighbors(args):
    idx, row = args
    molecule_name = row.molecule_name
    atom_index_0 = int(row.atom_index_0)
    atom_index_1 = int(row.atom_index_1)
    
    prop = {'molecule_name': molecule_name,
            'atom_index_0': atom_index_0,
            'atom_index_1': atom_index_1}

    # atom_0 is always hydrogen
    m = MolFromXYZ(PATH/f'structures/{molecule_name}.xyz') # less memory intensive in multiprocessing.Pool
    a0 = m.GetAtomWithIdx(atom_index_0)

    # neighbor of atom_0
    try:
        a0_nb_idx = [a.GetIdx() for a in a0.GetNeighbors() if a.GetIdx() != a0].pop()
    except:
        if molecule_name in nblist and atom_index_0 in nblist[molecule_name]:
            a0_nb_idx = nblist[molecule_name][atom_index_0]
        else:
            print(molecule_name)
            print(row)

    a0_nb = m.GetAtomWithIdx(a0_nb_idx)
    prop['atom_index_0_nb'] = a0_nb_idx

    # neighbor of atom_1
    a1 = m.GetAtomWithIdx(atom_index_1)
    a1_nb = {a.GetIdx(): a.GetSymbol() for a in a1.GetNeighbors() if a.GetIdx()}
    a1_nb = sorted(a1_nb.items(), key=lambda kv: kv[1])

    try:
        a1_nb_idx = [a.GetIdx() for a in a1.GetNeighbors() if a.GetIdx() != a1].pop()
    except:
        if molecule_name in nblist and atom_index_1 in nblist[molecule_name]:
            a1_nb_idx = nblist[molecule_name][atom_index_1]
        else:
            print(molecule_name)
            print(row)

    a1_nb = m.GetAtomWithIdx(a1_nb_idx)
    prop['atom_index_1_nb'] = a1_nb_idx
    return prop

def feature_neighbors(df):
    prop = []
    keys = []
    with Pool(n_cpu) as p:
        n = len(df)
        #rows = [(row.molecule_name, row.atom_index_0, row.atom_index_1) for idx, row in df.iterrows()]
        res = _feature((0, df.iloc[0]))
        keys = res.keys()
        
        with tqdm(total=n) as pbar:
            for res in p.imap_unordered(_feature_neighbors, df.iterrows()):
                prop.append([res[_] for _ in keys])
                pbar.update()
    
    prop = pd.DataFrame.from_records(prop, columns=keys)
    df = pd.merge(df, prop, how='left', on=['molecule_name', 'atom_index_0', 'atom_index_1'])
    return df

def feature_basic(df):
    structures = pd.read_csv(PATH/'structures.csv')
    df = feature_atomtype(df, structures)
    df = feature_pair_geometry(df)
    df = reduce_mem_usage(df)
    gc.collect()
    return df

def feature_atomic(df):
    df = feature_basic(df)
    df = feature_neighbors(df)
    gc.collect()
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create features for training')
    parser.add_argument('name', help='feature dataset name')
    parser.add_argument('--feature', required=True, choices=['basic', 'atomic'], help='feature dataset directory')
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

    if args.feature == 'basic':
        train = feature_basic(train)
        test = feature_basic(test)
    if args.feature == 'atomic':
        train = feature_atomic(train)
        test = feature_atomic(test)
    
    FEATURE = Path(args.feature_dir)
    suffix = '_subset' if args.subset else ''
    save_df(FEATURE/f'train_{args.name}{suffix}.csv', train)
    save_df(FEATURE/f'test_{args.name}{suffix}.csv', test)

