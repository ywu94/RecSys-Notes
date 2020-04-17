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
default_logger = logging.getLogger(__name__)

import numpy as np
import torch
assert torch.__version__>='1.2.0', 'Expect PyTorch>=1.2.0 but get {}'.format(torch.__version__)
from torch import nn

from sklearn.metrics import roc_auc_score

EPOCHES = 5
BATCH_SIZE = 2048

EPOCHES_SEP = 5
BATCH_SIZE_SEP = 4096

def get_roc_auc_score_separate_inp(model, test_data, device, logger=default_logger):
    """
    Get roc score for given model on prepared Criteo data. 
    The model will take a dense tensor and a sparse tensor as input.
    """
    global BATCH_SIZE_SEP
    BATCH_SIZE = BATCH_SIZE_SEP
    
    model.eval()
    
    pred_y, true_y = [], []
    test_feature_index, test_feature_value, test_label = test_data

    test_index_ub = test_label.shape[0]
    test_div, test_mod = divmod(test_index_ub, BATCH_SIZE)
    test_n_batch = int(test_div + min(test_mod, 1))

    for batch_index in range(test_n_batch):
        batch_start = batch_index*BATCH_SIZE
        batch_end = min((batch_index+1)*BATCH_SIZE, test_index_ub-1)          
        if batch_start < batch_end and batch_end - batch_start == BATCH_SIZE:
            batch_index_list = np.arange(batch_start, batch_end)
            
            inp_index = test_feature_index[batch_index_list]
            inp_value = test_feature_value[batch_index_list]
            inp_label = test_label[batch_index_list]
            
            inp_dense = np.where(inp_index[:,:13]>39, 1, 0) * inp_value[:,:13]
            inp_sparse = inp_index[:,13:]
            
            inp_dense = torch.from_numpy(inp_dense).float().to(device)
            inp_sparse = torch.from_numpy(inp_sparse).long().to(device)
            inp_label = torch.from_numpy(inp_label).float().to(device)
            
            pred_label = torch.nn.Sigmoid()(model(inp_dense, inp_sparse))
            
            pred_y.extend(list(pred_label.cpu().detach().numpy()))
            true_y.extend(list(inp_label.cpu().detach().numpy()))
            
            if (batch_index+1)%1000==0:
                logger.info('Batch {}/{} Done'.format(batch_index+1, test_n_batch))
            
    t = np.array(true_y).reshape((-1,))
    p = np.array(pred_y).reshape((-1,))
    
    roc_score = roc_auc_score(t,p)
    
    return roc_score

def train_model_separate_inp(model, train_data, test_data, loss_fn, optimizer, device, checkpoint_dir, checkpoint_prefix, logger=default_logger):
    """
    Train model using prepared Criteo data.
    The model will take a dense tensor and a sparse tensor as input.
    : param model
    : param train_data
    : param test_data
    : param loss_fn
    : param optimizer
    : param device
    """
    global EPOCHES_SEP
    EPOCHES = EPOCHES_SEP
    global BATCH_SIZE_SEP
    BATCH_SIZE = BATCH_SIZE_SEP
    
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    
    train_feature_index, train_feature_value, train_label = train_data
    test_feature_index, test_feature_value, test_label = test_data
    
    index_ub = train_label.shape[0]
    div, mod = divmod(index_ub, BATCH_SIZE)
    n_batch = int(div + min(mod, 1))
    
    test_index_ub = test_label.shape[0]
    test_div, test_mod = divmod(test_index_ub, BATCH_SIZE)
    test_n_batch = int(test_div + min(test_mod, 1))
    
    for epoch in range(1, EPOCHES+1):
        
        running_loss = 0
        test_running_loss = 0
        
        model.train()
        
        for batch_index in range(n_batch):
            batch_start = batch_index*BATCH_SIZE
            batch_end = min((batch_index+1)*BATCH_SIZE, index_ub-1)          
            if batch_start < batch_end and batch_end - batch_start == BATCH_SIZE:
                batch_index_list = np.arange(batch_start, batch_end)
                
                inp_index = train_feature_index[batch_index_list]
                inp_value = train_feature_value[batch_index_list]
                inp_label = train_label[batch_index_list]
            
                inp_dense = np.where(inp_index[:,:13]>39, 1, 0) * inp_value[:,:13]
                inp_sparse = inp_index[:,13:]
            
                inp_dense = torch.from_numpy(inp_dense).float().to(device)
                inp_sparse = torch.from_numpy(inp_sparse).long().to(device)
                inp_label = torch.from_numpy(inp_label).float().to(device)
                
                optimizer.zero_grad()
                pred_label = model(inp_dense, inp_sparse)
                loss = loss_fn(pred_label, inp_label)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100)
                optimizer.step()
                
                running_loss += loss.item()
                
                if (batch_index+1)%1000==0:
                    model.eval()
                    
                    batch_index_list = np.random.choice(test_index_ub, 512)
                    
                    inp_index = test_feature_index[batch_index_list]
                    inp_value = test_feature_value[batch_index_list]
                    inp_label = test_label[batch_index_list]
            
                    inp_dense = np.where(inp_index[:,:13]>39, 1, 0) * inp_value[:,:13]
                    inp_sparse = inp_index[:,13:]
            
                    inp_dense = torch.from_numpy(inp_dense).float().to(device)
                    inp_sparse = torch.from_numpy(inp_sparse).long().to(device)
                    inp_label = torch.from_numpy(inp_label).float().to(device)
                    
                    pred_label = model(inp_dense, inp_sparse)
                    val_loss = loss_fn(pred_label, inp_label)
                    
                    logger.info('Epoch {}/{} - Batch {}/{} Done - Train Loss: {:.6f}, Val Loss: {:.6f}'.format(epoch, EPOCHES, (batch_index+1), n_batch, running_loss/(batch_index+1), val_loss.item()))
                    
                    model.train()
                    
        pred_y = []
        true_y = []
        
        model.eval()
        
        for batch_index in range(test_n_batch):
            batch_start = batch_index*BATCH_SIZE
            batch_end = min((batch_index+1)*BATCH_SIZE, test_index_ub-1)          
            if batch_start < batch_end and batch_end - batch_start == BATCH_SIZE:
                batch_index_list = np.arange(batch_start, batch_end)
                
                inp_index = test_feature_index[batch_index_list]
                inp_value = test_feature_value[batch_index_list]
                inp_label = test_label[batch_index_list]
            
                inp_dense = np.where(inp_index[:,:13]>39, 1, 0) * inp_value[:,:13]
                inp_sparse = inp_index[:,13:]
            
                inp_dense = torch.from_numpy(inp_dense).float().to(device)
                inp_sparse = torch.from_numpy(inp_sparse).long().to(device)
                inp_label = torch.from_numpy(inp_label).float().to(device)
            
                pred_label = torch.nn.Sigmoid()(model(inp_dense, inp_sparse))
                
                pred_y.extend(list(torch.nn.Sigmoid()(pred_label).cpu().detach().numpy()))
                true_y.extend(list(inp_label.cpu().detach().numpy()))
            
                loss = loss_fn(pred_label, inp_label)
                
                test_running_loss += loss.item()
                
        t = np.array(true_y).reshape((-1,))
        p = np.array(pred_y).reshape((-1,))
        roc_score = roc_auc_score(t,p)      
        
        logger.info('Epoch {}/{} - Train Loss: {:.6f}, Test Loss: {:.6f}, Test ROC Score: {:.6f}'.format(epoch, EPOCHES, running_loss/(n_batch), test_running_loss/(test_n_batch), roc_score))
        
        ck_file_name = checkpoint_prefix + '_{}'.format(epoch)
        ck_file_path = os.path.join(checkpoint_dir, ck_file_name)
        
        torch.save(model.state_dict(), ck_file_path)
                    

def get_roc_auc_score(model, test_data, device, logger=default_logger):
    """
    Get roc score for given model on prepared Criteo data.
    The model will take a index tensor and value tensor as input.
    """
    global BATCH_SIZE
    
    model.eval()
    
    pred_y, true_y = [], []
    test_feature_index, test_feature_value, test_label = test_data

    test_index_ub = test_label.shape[0]
    test_div, test_mod = divmod(test_index_ub, BATCH_SIZE)
    test_n_batch = int(test_div + min(test_mod, 1))

    for batch_index in range(test_n_batch):
        batch_start = batch_index*BATCH_SIZE
        batch_end = min((batch_index+1)*BATCH_SIZE, test_index_ub-1)          
        if batch_start < batch_end and batch_end - batch_start == BATCH_SIZE:
            batch_index_list = np.arange(batch_start, batch_end)
            inp_index = torch.from_numpy(test_feature_index[batch_index_list]).long().to(device)
            inp_value = torch.from_numpy(test_feature_value[batch_index_list]).float().to(device)
            inp_label = torch.from_numpy(test_label[batch_index_list]).float().to(device)
            
            pred_label = torch.nn.Sigmoid()(model(inp_index, inp_value))
            pred_y.extend(list(pred_label.cpu().detach().numpy()))
            true_y.extend(list(inp_label.cpu().detach().numpy()))
            
    t = np.array(true_y).reshape((-1,))
    p = np.array(pred_y).reshape((-1,))
    
    roc_score = roc_auc_score(t,p)
    
    return roc_score

def train_model(model, train_data, test_data, loss_fn, optimizer, device, checkpoint_dir, checkpoint_prefix, logger=default_logger):
    """
    Train model using prepared Criteo data.
    The model will take a index tensor and value tensor as input.
    : param model
    : param train_data
    : param test_data
    : param loss_fn
    : param optimizer
    : param device
    """
    global EPOCHES
    global BATCH_SIZE
    
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    
    train_feature_index, train_feature_value, train_label = train_data
    test_feature_index, test_feature_value, test_label = test_data
    
    index_ub = train_label.shape[0]
    div, mod = divmod(index_ub, BATCH_SIZE)
    n_batch = int(div + min(mod, 1))
    
    test_index_ub = test_label.shape[0]
    test_div, test_mod = divmod(test_index_ub, BATCH_SIZE)
    test_n_batch = int(test_div + min(test_mod, 1))
    
    for epoch in range(1, EPOCHES+1):
        
        running_loss = 0
        test_running_loss = 0
        
        model.train()
        
        for batch_index in range(n_batch):
            batch_start = batch_index*BATCH_SIZE
            batch_end = min((batch_index+1)*BATCH_SIZE, index_ub-1)          
            if batch_start < batch_end and batch_end - batch_start == BATCH_SIZE:
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
                    model.eval()
                    
                    batch_index_list = np.random.choice(test_index_ub, 8192)
                    val_index = torch.from_numpy(test_feature_index[batch_index_list]).long().to(device)
                    val_value = torch.from_numpy(test_feature_value[batch_index_list]).float().to(device)
                    val_label = torch.from_numpy(test_label[batch_index_list]).float().to(device)
                    
                    val_pred_label = model(val_index, val_value)
                    val_loss = loss_fn(val_pred_label, val_label)
                    
                    logger.info('Epoch {}/{} - Batch {}/{} Done - Train Loss: {:.6f}, Val Loss: {:.6f}'.format(epoch, EPOCHES, (batch_index+1), n_batch, running_loss/(batch_index+1), val_loss.item()))
                    
                    model.train()
                    
        pred_y = []
        true_y = []
        
        model.eval()
        
        for batch_index in range(test_n_batch):
            batch_start = batch_index*BATCH_SIZE
            batch_end = min((batch_index+1)*BATCH_SIZE, test_index_ub-1)          
            if batch_start < batch_end and batch_end - batch_start == BATCH_SIZE:
                batch_index_list = np.arange(batch_start, batch_end)
                inp_index = torch.from_numpy(test_feature_index[batch_index_list]).long().to(device)
                inp_value = torch.from_numpy(test_feature_value[batch_index_list]).float().to(device)
                inp_label = torch.from_numpy(test_label[batch_index_list]).float().to(device)
                
                pred_label = model(inp_index, inp_value)
                
                pred_y.extend(list(torch.nn.Sigmoid()(pred_label).cpu().detach().numpy()))
                true_y.extend(list(inp_label.cpu().detach().numpy()))
            
                loss = loss_fn(pred_label, inp_label)

                if model.reg_l1:
                    for param in model.parameters():
                        loss += model.reg_l1 * torch.sum(torch.abs(param))
                        
                if model.reg_l2:
                    for param in model.parameters():
                        loss += model.reg_l2 * torch.sum(torch.pow(param, 2))
                
                test_running_loss += loss.item()
                
        t = np.array(true_y).reshape((-1,))
        p = np.array(pred_y).reshape((-1,))
        roc_score = roc_auc_score(t,p)      
        
        logger.info('Epoch {}/{} - Train Loss: {:.6f}, Test Loss: {:.6f}, Test ROC Score: {:.6f}'.format(epoch, EPOCHES, running_loss/(n_batch), test_running_loss/(test_n_batch), roc_score))
        
        ck_file_name = checkpoint_prefix + '_{}'.format(epoch)
        ck_file_path = os.path.join(checkpoint_dir, ck_file_name)
        
        torch.save(model.state_dict(), ck_file_path)
                    

        