"""
Prepare Criteo data.
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

import multiprocessing as mp
from multiprocessing import Process, Lock, Manager
from functools import reduce

import numpy as np

def prepare_single_file(file_name, embedding_map_dict, process_list, process_lock, training=True, shuffle=True):
    """
    Read a file and return feature indexes, feature values and labels if the file contains training data
        : index 1~39 are reserved for missing features
        : index 40~52 are reserved for numerical features
        : index 53 onwards are for categorical feature encoding
    : param training: whether the 
    : return feature_index: list[int]
    : return feature_value: list[int]
    : return label: list[int] (train) / None (test)
    """
    feature_index = []
    feature_value = []
    label = [] if training else None
    
    logger.info('Processing {}'.format(file_name))
    f = open(file_name, 'r')
    
    for line_index, line in enumerate(f):
        if line_index+1%200000==0:
            logger.info('{}: {}/3276800'.format(file_name, line_index+1))
        feature = line.rstrip('\n').split('\t')
        if training:
            assert len(feature)==40, 'Encounter error in reading line {}: {}'.format(line_index, line)
            label.append(float(feature[0]))
            for index, value in enumerate(feature[1:14], start=1):
                if value and int(value) >= 0:
                    feature_index.append(index+39)
                    feature_value.append(pow(math.log(1+int(value)),2))
                else:
                    feature_index.append(index)
                    feature_value.append(1)
            for index, value in enumerate(feature[14:], start=14):
                if value:
                    token = str(index)+value
                    if token in embedding_map_dict:
                        feature_index.append(embedding_map_dict[token])
                        feature_value.append(1)
                        continue
                feature_index.append(index)
                feature_value.append(1)
        else:
            assert len(feature)==39, 'Encounter error in reading line {}: {}'.format(line_index, line)
            for index, value in enumerate(feature[:13], start=1):
                if value and int(value) >= 0:
                    feature_index.append(index+39)
                    feature_value.append(pow(math.log(1+int(value)),2))
                else:
                    feature_index.append(index)
                    feature_value.append(1)
            for index, value in enumerate(feature[13:], start=14):
                if value:
                    token = str(index)+value
                    if token in embedding_map_dict:
                        feature_index.append(embedding_map_dict[token])
                        feature_value.append(1)
                        continue
                feature_index.append(index)
                feature_value.append(1)
                
    f.close()
    
    feature_index = np.array(feature_index).reshape((-1,39))
    feature_value = np.array(feature_value).reshape((-1,39))
    label = np.array(label).reshape((-1,1)) if training else None
    
    if shuffle:
        index_list = np.arange(len(feature_index))
        np.random.shuffle(index_list)
        feature_index = feature_index[index_list,:]
        feature_value = feature_value[index_list,:]
        label = label[index_list,:] if training else None
    
    process_lock.acquire()
    process_list.append((feature_index, feature_value, label))
    process_lock.release()
    
    logger.info('Done with {}'.format(file_name))
    
    return

if __name__ == '__main__':
    cwd = os.getcwd()
    
    embedding_map_dict_pkl_path = os.path.join(cwd, 'criteo_feature_dict_artifact/categorical_feature_map_dict.pkl')
    with open(embedding_map_dict_pkl_path, 'rb') as f:
        embedding_map_dict = pickle.load(f)
        
    train_raw_root = os.path.join(cwd, 'criteo_train_raw_artifact')
    train_raw_list = sorted([os.path.join(train_raw_root, f) for f in os.listdir(train_raw_root)], key = lambda x:int(x.split('-')[-1]))
    
    process_lock = Lock()
    process_manager = Manager()
    process_list = process_manager.list()

    ns = process_manager.Namespace()
    ns.embedding_map_dict = embedding_map_dict
    
    processes = []
    for artifact_path in train_raw_list:
        processes.append(Process(target=prepare_single_file, args=(artifact_path, ns.embedding_map_dict, process_list, process_lock)))
    for p in processes:
        p.start()
    for p in processes:
        p.join()
        
    train_prepared_dir = os.path.join(cwd, 'criteo_train_prepared_artifact')
    if os.path.isdir(train_prepared_dir):
        shutil.rmtree(train_prepared_dir, onerror=handle_error)
    os.mkdir(train_prepared_dir)
        
    feature_index = np.vstack([i[0] for i in process_list])
    np.save(os.path.join(train_prepared_dir, 'feature_index.npy'), feature_index)
    
    feature_value = np.vstack([i[1] for i in process_list])
    np.save(os.path.join(train_prepared_dir, 'feature_value.npy'), feature_value)
    
    label = np.vstack([i[2] for i in process_list])
    np.save(os.path.join(train_prepared_dir, 'label.npy'), label)
    
    

    

    
        
    
    