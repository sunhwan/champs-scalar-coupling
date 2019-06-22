import os
import numpy as np

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

# https://www.kaggle.com/uberkinder/efficient-metric
def group_mean_log_mae(y_true, y_pred, types, floor=1e-9):
    maes = (y_true-y_pred).abs().groupby(types).mean()
    return np.log(maes.map(lambda x: max(x, floor))).mean()

# for atomic features
# rdkit & xyz2mol
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import SanitizeFlags

# https://github.com/jensengroup/xyz2mol
from xyz2mol import xyz2mol, xyz2AC, AC2mol, read_xyz_file
from pathlib import Path
import pickle

def chiral_stereo_check(mol):
    # avoid sanitization error e.g., dsgdb9nsd_037900.xyz
    Chem.SanitizeMol(mol, SanitizeFlags.SANITIZE_ALL - SanitizeFlags.SANITIZE_PROPERTIES)
    Chem.DetectBondStereochemistry(mol,-1)
    # ignore stereochemistry for now
    #Chem.AssignStereochemistry(mol, flagPossibleStereoCenters=True, force=True)
    #Chem.AssignAtomChiralTagsFromStructure(mol,-1)
    return mol

def xyz2mol(atomicNumList,charge,xyz_coordinates,charged_fragments,quick):
    AC,mol = xyz2AC(atomicNumList,xyz_coordinates)
    new_mol = AC2mol(mol,AC,atomicNumList,charge,charged_fragments,quick)
    new_mol = chiral_stereo_check(new_mol)
    return new_mol

def MolFromXYZ(filename):
    charged_fragments = True
    quick = True
    cache_filename = filename.parent/f'{filename.stem}.pkl'
    if cache_filename.exists():
        return pickle.load(open(cache_filename, 'rb'))
    else:
        try:
            atomicNumList, charge, xyz_coordinates = read_xyz_file(filename)
            mol = xyz2mol(atomicNumList, charge, xyz_coordinates, charged_fragments, quick)
            pickle.dump(mol, open(cache_filename, 'wb'))
        except:
            print(filename)
    return mol

def MolFromXYZ_(filename):
    return filename.stem, MolFromXYZ(filename)

