"""
Prepare raw Criteo data into numerical data
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

def prepare_criteo(file_name, embedding_map_dict, training=True):
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
    global continuous_min
    global continuous_diff
    
    feature_index = []
    feature_value = []
    label = [] if training else None
    
    logger.info('Processing {}'.format(file_name))
    f = open(file_name, 'r')
    
    for line_index, line in enumerate(f):
        feature = line.rstrip('\n').split('\t')
        if training:
            assert len(feature)==40, 'Encounter error in reading line {}: {}'.format(line_index, line)
            label.append(feature[0])
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
    
    return feature_index, feature_value, label
    