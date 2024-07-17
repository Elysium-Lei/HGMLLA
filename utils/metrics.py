# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 10:59:37 2023

@author: 雷雨
"""

import torch
import numpy as np
from scipy.special import softmax
from sklearn.metrics import roc_auc_score

def accuracy(preds, labels):
    """Accuracy, auc with masking.Acc of the masked samples"""
    correct_prediction = np.equal(np.argmax(preds, 1), labels).astype(np.float32)
    return np.sum(correct_prediction), np.mean(correct_prediction)

def auc(preds, labels, is_logit=True):
    ''' input: logits, labels  ''' 
    if is_logit:
        pos_probs = softmax(preds, axis=1)[:, 1]
    else:
        pos_probs = preds[:,1]
    try:
        auc_out = roc_auc_score(labels, pos_probs)
    except:
        auc_out = 0
    return auc_out

def sen_spe(output, target):
    '''
        这里类别数为3
        
        传入参数：
        output --> tensor(80,3) 从outputs, _ = net(inputs)中获取
        target --> tensor(80)
        
        返回值：
        sensitivity --> np.array
    '''
    # 取得到分类分数最大的值，返回第一维度是value，第二维度是index
    _, pred = output.max(1) 
    # 将 pred 展开成 one-hot编码形式
    pre_mask = torch.zeros(output.size()).scatter_(1, pred.cpu().view(-1, 1), 1.)
    # 将 target 也展开成 one-hot编码形式
    tar_mask = torch.zeros(output.size()).scatter_(1, target.data.cpu().view(-1, 1), 1.)
    # 计算 acc 的one-hot编码形式
    acc_mask = pre_mask * tar_mask
    # 计算 sen和spe
    temp = acc_mask.sum(0) / tar_mask.sum(0)   # 第一列是为负样本的个数，第二列是正样本的个数
    
    # 转换成numpy()
    sen = temp[1].numpy()
    spe = temp[0].numpy()

    return sen,spe


def ppv_npv(output, target):

    # 取得到分类分数最大的值，返回第一维度是value，第二维度是index
    _, pred = output.max(1) 
    # 将 pred 展开成 one-hot编码形式
    pre_mask = torch.zeros(output.size()).scatter_(1, pred.cpu().view(-1, 1), 1.)
    # 将 target 也展开成 one-hot编码形式
    tar_mask = torch.zeros(output.size()).scatter_(1, target.data.cpu().view(-1, 1), 1.)
    # 计算 acc 的one-hot编码形式
    acc_mask = pre_mask * tar_mask
    
    temp = acc_mask.sum(0) / pre_mask.sum(0)   # 第一列是为负样本的个数，第二列是正样本的个数
    
    # 转换成numpy()
    ppv = temp[1].numpy()
    npv = temp[0].numpy()

    return ppv,npv