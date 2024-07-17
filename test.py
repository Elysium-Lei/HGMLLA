import os
import argparse
import logging
from tqdm import tqdm
import numpy as np
import scipy.io as sio

import torch
from torch import nn
from torch_geometric.loader import DataLoader

from HeteroDataLoad import HeteroDataPre, MyHeteroData
from utils import(
    random_split,
    FocalLoss,
    accuracy,
    auc,
    sen_spe,
)
from add_random_walk_pe import create_pe
from model.hgmamba import HGMamba

def test(model, loader, local_rank, criterion, args):
    model.eval()
    test_acc = 0.
    test_auc = 0.
    test_sen = 0.
    test_spe = 0.
    test_loss = 0.

    for ind, data in enumerate(loader):
        if local_rank is not None:
            data = data.cuda(local_rank)
        else:
            assert args.device is not None, f'Please set device'
            data = data.to(args.device)
        pe_dict = create_pe(data, args.walk_length)

        if args.model_mode == 'mlla':
                out = model(data.x_dict, data.edge_index_dict, data.batch_dict)
        else:
            out = model(data.x_dict, pe_dict, data.edge_index_dict,  data.batch_dict)
        
        loss_tmp = criterion(out, data.y_dict['bold'])
        
        logits = out.detach().cpu().numpy()
        label = data.y_dict['bold'].detach().cpu().numpy()
        
        correct, acc_tmp = accuracy(logits, label) 
        auc_tmp = auc(logits, label)
        sen_tmp, spe_tmp = sen_spe(out, data.y_dict['bold'])
        
        test_acc += acc_tmp
        test_auc += auc_tmp
        test_sen += sen_tmp
        test_spe += spe_tmp
        test_loss += loss_tmp
        
    test_acc = test_acc/(ind+1)
    test_auc = test_auc/(ind+1)
    test_sen = test_sen/(ind+1)
    test_spe = test_spe/(ind+1)
    test_loss = test_loss/(ind+1)
    
    return test_acc, test_auc, test_sen, test_spe, test_loss