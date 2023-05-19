import os
import sys
import re
import glob
import pickle
import copy
import torch
import wfdb
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer

from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data

# ICBEB dataset
# label_dict = {'NORM':0, 'AFIB':1, '1AVB':2, 'CLBBB':3, 'CRBBB':4, 'PAC':5, 'VPC':6, 'STD_':7, 'STE_':8}

# PTB-XL dataset
label_dict = {'NORM':0, 'CD':1, 'HYP':2, 'MI':3, 'STTC':4}

drop_index = []

# Define ECG Graph Dataset
class ECGDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) :
        return ["records100"]
    
    @property
    def processed_file_names(self) :
        return ['data.pt']
    
    def download(self):
        pass

    def process(self):
        data_list = []
        edge_indexs = torch.tensor([[0,0,1,1,1,2,2,1,1,1,1,1,1, 3,4,3,4,5,4,5,6,7,8,9,10,11]
                                    ,[3,4,3,4,5,4,5,6,7,8,9,10,11, 0,0,1,1,1,2,2,1,1,1,1,1,1]])
        rawdata_path = "you_path/data/ptb/raw/"
        X, label, _ = load_dataset(rawdata_path, 100)
        for idx in range(X.shape[0]):
            # if X[idx].shape[0] < 1000 :   # For ICBEB, if the recode time is low 10s, then remove
            #     drop_index.append(idx)
            #     continue
            # print(Y.iloc[idx]['diagnostic_superclass'])
            data = Data(x=torch.tensor(X[idx]).permute(1,0), edge_index=edge_indexs, y=label[idx])
            data_list.append(data)
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def select_dataset(dataset, Y):
    
    if len(drop_index) > 0:
        Y.drop(drop_index, inplace=True)
       
        print("===do drop===")
    
    Y.index = range(len(Y))

    train_dataset = dataset[list(Y[Y.strat_fold <= 8].index)]
    val_dataset = dataset[list(Y[Y.strat_fold == 9].index)]
    test_dataset = dataset[list(Y[Y.strat_fold == 10].index)]

    return train_dataset, val_dataset, test_dataset

def load_dataset(path, sampling_rate, release=False):
    if path.split('/')[-3] == 'ptb':
        # load and convert annotation data
        Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load raw signal data
        X = load_raw_data_ptbxl(Y, sampling_rate, path)
        agg_df = pd.read_csv(os.path.join(path, "scp_statements.csv"), index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]

        def agg(y_dic):
            temp = []

            for key in y_dic.keys():
                if key in agg_df.index:
                    c = agg_df.loc[key].diagnostic_class
                    if str(c) != "nan":
                        temp.append(c)
            return list(set(temp))

        Y["diagnostic_superclass"] = Y.scp_codes.apply(agg)
        Y["superdiagnostic_len"] = Y["diagnostic_superclass"].apply(lambda x: len(x))
        counts = pd.Series(np.concatenate(Y.diagnostic_superclass.values)).value_counts()
        Y["diagnostic_superclass"] = Y["diagnostic_superclass"].apply(
            lambda x: list(set(x).intersection(set(counts.index.values)))
        )

        X = X[Y["superdiagnostic_len"] >= 1]
        Y = Y[Y["superdiagnostic_len"] >= 1]
        
        mlb = MultiLabelBinarizer()
        mlb.fit(Y["diagnostic_superclass"])
        label = mlb.transform(Y["diagnostic_superclass"].values)

    elif path.split('/')[-3] == 'ICBEB':
        # load and convert annotation data
        Y = pd.read_csv(path+'icbeb_database.csv', index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
        global drop_index
        drop_index = [797, 1871, 2662, 2694, 4228, 5276]
        # Load raw signal data
        X = load_raw_data_icbeb(Y, sampling_rate, path)

        mlb = MultiLabelBinarizer()
        mlb.fit(Y["scp_codes"])
        y = mlb.transform(Y["scp_codes"].values)

    return X, label, Y

def load_raw_data_ptbxl(df, sampling_rate, path):
    if sampling_rate == 100:
        if os.path.exists(path + 'raw100.npy'):
            data = np.load(path+'raw100.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path+f) for f in tqdm(df.filename_lr)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path+'raw100.npy', 'wb'), protocol=4)
    elif sampling_rate == 500:
        if os.path.exists(path + 'raw500.npy'):
            data = np.load(path+'raw500.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path+f) for f in tqdm(df.filename_hr)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path+'raw500.npy', 'wb'), protocol=4)
    return data

def load_raw_data_icbeb(df, sampling_rate, path):

    if sampling_rate == 100:
        if os.path.exists(path + 'raw100.npy'):
            data = np.load(path+'raw100.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path + 'records100/'+str(f)) for f in tqdm(df.index)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path+'raw100.npy', 'wb'), protocol=4)
    elif sampling_rate == 500:
        if os.path.exists(path + 'raw500.npy'):
            data = np.load(path+'raw500.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path + 'records500/'+str(f)) for f in tqdm(df.index)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path+'raw500.npy', 'wb'), protocol=4)
    return data


def challenge_metrics(y_true, y_pred, beta1=2, beta2=2, class_weights=None, single=False):
    f_beta = 0
    g_beta = 0
    if single: # if evaluating single class in case of threshold-optimization
        sample_weights = np.ones(y_true.sum(axis=1).shape)
    else:
        sample_weights = y_true.sum(axis=1)
    for classi in range(y_true.shape[1]):
        y_truei, y_predi = y_true[:,classi], y_pred[:,classi]
        TP, FP, TN, FN = 0.,0.,0.,0.
        for i in range(len(y_predi)):
            sample_weight = sample_weights[i]
            if y_truei[i]==y_predi[i]==1: 
                TP += 1./sample_weight
            if ((y_predi[i]==1) and (y_truei[i]!=y_predi[i])): 
                FP += 1./sample_weight
            if y_truei[i]==y_predi[i]==0: 
                TN += 1./sample_weight
            if ((y_predi[i]==0) and (y_truei[i]!=y_predi[i])): 
                FN += 1./sample_weight 
        f_beta_i = ((1+beta1**2)*TP)/((1+beta1**2)*TP + FP + (beta1**2)*FN)
        g_beta_i = (TP)/(TP+FP+beta2*FN)

        f_beta += f_beta_i
        g_beta += g_beta_i

    print(f'F_beta_macro:{f_beta/y_true.shape[1]}, G_beta_macro: {g_beta/y_true.shape[1]}')


def data_scaler(train_dataset, val_dataset, test_dataset):
    X_train, X_val, X_test = [], [], []

    for data in train_dataset:
        X_train.append(data.x)
    
    for data in val_dataset: 
        X_val.append(data.x)
      
    for data in test_dataset:
        X_test.append(data.x) 

 #   print(X_train[0][1][:10])
    # Standardization
    scaler = StandardScaler()
    scaler.fit(np.vstack(X_train).flatten()[:, np.newaxis].astype(float))
    X_train_scale = apply_scaler(X_train, scaler)
    X_test_scale = apply_scaler(X_test, scaler)
    X_val_scale = apply_scaler(X_val, scaler)
    print(X_train_scale.shape, X_val_scale.shape, X_test_scale.shape)
  #  print(X_train_scale[0,0,:10])

    for i in range(len(train_dataset)):
        # train_dataset[i].x = X_train_scale[i]
        setattr(train_dataset[i], 'x', X_train_scale[i])
    for i in range(len(val_dataset)):
       # val_dataset[i].x = X_val_scale[i]
        setattr(val_dataset[i], 'x', X_val_scale[i])
    for i in range(len(test_dataset)):
       # test_dataset[i].x = X_test_scale[i]
        setattr(test_dataset[i], 'x', X_test_scale[i])
    return train_dataset, val_dataset, test_dataset

def apply_scaler(inputs: np.array, scaler: StandardScaler) -> np.array:
    """Applies standardization to each individual ECG signal.

    Parameters
    ----------
    inputs: np.array
        Array of ECG signals.
    scaler: StandardScaler
        Standard scaler object.

    Returns
    -------
    np.array
        Array of standardized ECG signals.

    """

    temp = []
    for x in inputs:
        x_shape = x.shape
        temp.append(scaler.transform(x.flatten()[:, np.newaxis]).reshape(x_shape))
    temp = np.array(temp)
    return temp




