"""
Train model using prepared Criteo data.
"""

import os
import sys
import shutil
import pickle
import math

import sys
import logging
import importlib

importlib.reload(logging)
logging.basicConfig(stream=sys.stdout, format='%(asctime)s %(levelname)-6s %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

import numpy as np
import torch
assert torch.__version__>='1.2.0', 'Expect PyTorch>=1.2.0 but get {}'.format(torch.__version__)
from torch import nn

EPOCHES = 10
BATCH_SIZE = 2048

def train_model(model, train_data, test_data, loss_fn, optimizer, device):
    """
    Train model using prepared Criteo data.
    : param model
    : param train_data
    : param test_data
    : param loss_fn
    : param optimizer
    : param device
    """
    global EPOCHES
    global BATCH_SIZE
    
    train_feature_index, train_feature_value, train_label = train_data
    test_feature_index, test_feature_value, test_label = test_data
    
    test_feature_index = torch.from_numpy(test_feature_index).long().to(device)
    test_feature_value = torch.from_numpy(test_feature_value).float().to(device)
    test_label = torch.from_numpy(test_label).float().to(device)
    
    index_ub = train_label.shape[0]
    div, mod = divmod(index_ub, BATCH_SIZE)
    n_batch = int(div + min(mod, 1))
    
    for epoch in range(1, EPOCHES+1):
        running_loss = 0
        for batch_index in range(n_batch):
            batch_start = batch_index*BATCH_SIZE
            batch_end = min((batch_index+1)*BATCH_SIZE, index_ub-1)          
            if batch_start < batch_end:
                batch_index_list = np.arange(batch_start, batch_end)
                inp_index = torch.from_numpy(train_feature_index[batch_index_list]).long().to(device)
                inp_value = torch.from_numpy(train_feature_value[batch_index_list]).float().to(device)
                inp_label = torch.from_numpy(train_label[batch_index_list]).float().to(device)
                
                optimizer.zero_grad()
                pred_label = model(inp_index, inp_value)
                loss = loss_fn(pred_label, inp_label)

                if model.reg_l1:
                    for param in model.parameters():
                        loss += model.reg_l1 * torch.sum(torch.abs(param))
                if model.reg_l2:
                    for param in model.parameters():
                        loss += model.reg_l2 * torch.sum(torch.pow(param, 2))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100)
                optimizer.step()
                
                running_loss += loss.item()
                
                if (batch_index+1)%1000==0:
                    logger.info('Epoch {}/{} - Batch {}/{} Done - Training Loss: {:.6f}'.format(epoch, EPOCHES, (batch_index+1), n_batch, running_loss/(batch_index+1)))
        
        pred_test_label = model(test_feature_index, test_feature_value)
        test_loss = loss_fn(pred_test_label, test_label)
        
        logger.info('Epoch {}/{} - Test Loss: {:.6f}'.format(epoch, EPOCHES, test_loss.item()))
                    

        