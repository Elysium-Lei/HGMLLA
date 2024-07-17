import os
import argparse
import logging
from tqdm import tqdm
import numpy as np
import scipy.io as sio

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.loader import DataLoader

from HeteroDataLoad import HeteroDataPre, MyHeteroData
from utils import(
    random_split,
    FocalLoss,
    focal_loss,
    CosineScheduler,
    accuracy,
    auc,
    sen_spe,
)
from add_random_walk_pe import create_pe
from test import test
from model.hgmamba import HGMamba
from model.hgmlla import HGMLLA

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='./data', type=str, help='path of dataset')
parser.add_argument('--data_name', default='HeteroGraph_ADNI_M_DFNC', type=str, help='name of dataset')
parser.add_argument('--mask_name', default='train_val_test_mask_DFNC', type=str, help='name of mask')
parser.add_argument('--model_mode', default='mlla', type=str, help='model mode')
parser.add_argument('--use_pe', default=False, type=bool, help='use positional embedding')
parser.add_argument('--walk_length', default=20, type=int, help='length of random walk')
parser.add_argument('--pe_dim', default=8, type=int, help='dimension of positional embedding')
parser.add_argument('--gpu', default='1', type=str, help='GPU ID')
parser.add_argument('--seed', default=108, type=int, help='manual seed')
parser.add_argument('--epochs', default=1000, type=int, help='num of epochs')
parser.add_argument('--warmup_epochs', default=10, type=int, help='warmup epochs for scheduler')
parser.add_argument('--const_epochs', default=0, type=int, help='const epochs for schedular')
parser.add_argument('--hidden_channels', default=128, type=int, help='num of hidden channels')
parser.add_argument('--patience', default=50, type=int, help='patience for early stopping')
parser.add_argument('--batch_size', default=36, type=int, help='batch size')
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay')
parser.add_argument('--dropout_ratio', default=0.45, type=float, help='dropout ratio')

def count_parameters(model, only_trainable=False):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def main(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    test_set, train_set, val_set = random_split(Dataset, os.path.join(args.data_path, args.mask_name+'.mat'))

    trainLoader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valLoader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)
    testLoader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

    logging.info('Prepare model...')
    node_types = Dataset.get(0).node_types
    metadata = Dataset.get(0).metadata()

    in_channels = {}
    in_channels[node_types[0]] = 187
    in_channels[node_types[-1]] = 57

    if args.model_mode == 'mamba':
        model = HGMamba(
            in_channels=in_channels,
            pe_dim=args.pe_dim,
            walk_length=args.walk_length,
            out_channels=args.hidden_channels,
            metadata=metadata, 
            num_layers=3,
            model_type='mamba',
            use_pe=False,
            use_local_mamba=False,
            use_global_mamba=True,
            use_gma_message_passing=False,
            use_meta_path_attn=False,
            message_passing_method=None,
        ).to(args.device)

    if args.model_mode == 'han':
        model = HGMamba(
            in_channels=in_channels,
            pe_dim=args.pe_dim,
            walk_length=args.walk_length,
            out_channels=args.hidden_channels,
            metadata=metadata, 
            num_layers=3,
            model_type='han',
            use_pe=False,
        ).to(args.device)
    
    if args.model_mode == 'mlla':
        model = HGMLLA(
            in_channels=in_channels,
            hidden_channels=args.hidden_channels,
            metadata=metadata, 
            num_layers=3,
            heads=8,
        ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = CosineScheduler(optimizer, param_name='lr', t_max=args.epochs, value_min=args.lr * 1e-2, 
                                   warmup_t=args.warmup_epochs, const_t=args.const_epochs)
    wd_scheduler = CosineScheduler(optimizer, param_name='weight_decay', t_max=args.epochs)
    criterion = FocalLoss(class_num=2, alpha=torch.from_numpy(np.array([0.45,0.55])), gamma=1, size_average=True)

    cnt_wait = 0
    best = 1e9
    best_acc = 0.5
    best_model_state_dict = None

    logging.info('Train model...')
    for epoch in tqdm(range(args.epochs)):
        model.train()
        tra_loss = 0.0
        tra_acc = 0.0
        tra_auc = 0.0
        tra_sen = 0.0
        tra_spe = 0.0
        
        for ind, data in enumerate(trainLoader):
            optimizer.zero_grad()
            data = data.to(args.device)
            if args.use_pe:
                pe_dict = create_pe(data, args.walk_length)
            else:
                pe_dict = None

            if args.model_mode == 'mlla':
                out = model(data.x_dict, data.edge_index_dict, data.batch_dict)
            else:
                out = model(data.x_dict, pe_dict, data.edge_index_dict,  data.batch_dict)
            loss = criterion(out, data.y_dict['bold'])

            loss.backward()
            optimizer.step()
            lr_scheduler.step(epoch + 1)
            wd_scheduler.step(epoch + 1)

            logits = out.detach().cpu().numpy()
            label = data.y_dict['bold'].detach().cpu().numpy()
            
            correct_train, acc_train = accuracy(logits, label)
            auc_train = auc(logits, label)
            sen_train, spe_train = sen_spe(out, data.y_dict['bold'])
            
            tra_acc += acc_train
            tra_auc += auc_train
            tra_sen += sen_train
            tra_spe += spe_train
            tra_loss += loss     
        
        tra_acc = tra_acc/(ind+1)
        tra_auc = tra_auc/(ind+1)
        tra_sen = tra_sen/(ind+1)
        tra_spe = tra_spe/(ind+1)
        tra_loss = tra_loss / (ind+1)

        val_acc, val_auc, val_sen, val_spe, val_loss = test(model,valLoader, None, criterion, args)
        test_acc, test_auc, test_sen, test_spe, test_loss = test(model, testLoader, None, criterion, args)
        params = count_parameters(model, only_trainable=True)

        print("\nTra loss:{:.5f}  acc:{:.4f}  auc:{:.4f}  sen:{:.4f}  spe:{:.4f} , Val loss:{:.5f}  acc:{:.4f}  auc:{:.4f}  sen:{:.4f}  spe:{:.4f}".format(tra_loss,
            tra_acc, tra_auc, tra_sen, tra_spe, val_loss, val_acc, val_auc, val_sen, val_spe))
        print("Test accuracy:{:.4f}\t auc:{:.4f}\t sen:{:.4f}\t spe:{:.4f}".format(test_acc, test_auc, test_sen, test_spe))
        print("Params:{}".format(params))
        
        if test_acc >= best_acc:
            best_acc = test_acc
            best_model_state_dict = model.state_dict()
            torch.save(best_model_state_dict, './best_model.pth')

        if loss < best:
            best = loss
            cnt_wait = 0
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!')
            break

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    args = parser.parse_args()

    assert args.gpu is not None, f'Please set GPU ID'
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    logging.info('Prepare data...')
    HeteroDataPre(args.data_path, args.data_name)
    Dataset = MyHeteroData(args.data_path)
    args.Dataset = Dataset
    
    main(args)