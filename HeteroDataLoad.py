# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 18:48:58 2023

@author: 雷雨
"""

import os
import os.path as osp
from itertools import product

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch

from torch_geometric.data import (
    HeteroData,
    Dataset,
)


def HeteroDataPre(path, name):
    raw_mat = sio.loadmat(osp.join(path, name+'.mat'))
    
    heterographall = raw_mat['HeteroGraph_binary']
    features_bold_all = raw_mat['bold_node_feature']
    features_dti_all = raw_mat['dti_node_feature']
    label_all = raw_mat['label']
    
    out_adjM = './data/raw/adjM'
    out_bold = './data/raw/bold_features'
    out_dti = './data/raw/dti_features'
    out_label = './data/raw/label'
    
    if not osp.exists(out_adjM):
        os.makedirs(out_adjM)
    if not osp.exists(out_bold):
        os.makedirs(out_bold)
    if not osp.exists(out_dti):
        os.makedirs(out_dti)
    if not osp.exists(out_label):
        os.makedirs(out_label)
        
    for ind in range(heterographall.shape[2]):
        heterograph = heterographall[:,:,ind]
        features_bold = features_bold_all[:,:,ind]
        features_dti = features_dti_all[:,:,ind]
        label = label_all[ind]
        
        np.save(osp.join(out_adjM, f'sub_{ind}.npy'), heterograph)
        np.save(osp.join(out_bold, f'sub_{ind}.npy'), features_bold)
        np.save(osp.join(out_dti, f'sub_{ind}.npy'), features_dti)
        np.save(osp.join(out_label, f'sub_{ind}.npy'), label)


class MyHeteroData(Dataset):
    def __init__(self, root, transform = None, pre_transform = None):
        super().__init__(root, transform, pre_transform)
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return []
    
    def download(self):
        pass
    
    def process(self):   
        adjM_path = osp.join(self.raw_dir, 'adjM')
        bold_path = osp.join(self.raw_dir, 'bold_features')
        dti_path = osp.join(self.raw_dir, 'dti_features')
        label_path = osp.join(self.raw_dir, 'label')
        
        sub_dir = os.listdir(adjM_path)
        sub_dir.sort(key=lambda x: int(x[4:-4]))
        
        node_types = ['bold', 'dti']
        
        ind = 0        
        for name in sub_dir:
            data = HeteroData()
            
            heterograph = np.load(osp.join(adjM_path, name))
            features_bold = np.load(osp.join(bold_path, name))
            features_dti = np.load(osp.join(dti_path, name))
            label = np.load(osp.join(label_path, name))
            
            data['bold'].x = torch.from_numpy(features_bold).to(torch.float)
            data['dti'].x = torch.from_numpy(features_dti).to(torch.float)
            
            data['bold'].y = torch.from_numpy(label).to(torch.long)
            data['dti'].y = torch.from_numpy(label).to(torch.long)
            
            s= {}
            N_bold = data['bold'].num_nodes
            N_dti = data['dti'].num_nodes
            s['bold'] = (0, N_bold)
            s['dti'] = (N_bold, N_bold + N_dti)
            
            A = sp.csr_matrix(heterograph)
            for src, dst in product(node_types, node_types):
                A_sub = A[s[src][0]:s[src][1], s[dst][0]:s[dst][1]].tocoo()
                if A_sub.nnz > 0:
                    row = torch.from_numpy(A_sub.row).to(torch.long)
                    col = torch.from_numpy(A_sub.col).to(torch.long)
                    data[src, dst].edge_index = torch.stack([row, col], dim=0)
                    data[src, dst].edge_attr = A_sub.data
                    
            if self.pre_transform is not None:
                data = self.pre_transform(data)
                
            torch.save(data, osp.join(self.processed_dir, f'data_{ind}.pt'))
            
            ind += 1
        
    def len(self):
        return len(os.listdir(self.processed_dir)) - 2

    def get(self, ind):
        data = torch.load(osp.join(self.processed_dir, f'data_{ind}.pt'))
        return data   
        