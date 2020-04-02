"""
Split the data file into smaller files and generate embedding map dictionary.
"""

import os
import sys
import shutil
import pickle

import sys
import logging
import importlib
importlib.reload(logging)

logging.basicConfig(stream=sys.stdout, format='%(asctime)s %(levelname)-6s %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

def handle_error(func, path, exc_info):
    """
    Handle error for shutil.
    """
    print('Handling Error for file {}'.format(path))
    print(exc_info)
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWUSR)
        func(path)

def check():
    """
    Check if the raw data are placed in the right repository, and create directory for new files.
    """
    cwd = os.getcwd()
    
    raw_dir = os.path.join(cwd, 'criteo_raw_artifact')
    assert os.path.isdir(raw_dir), 'Missing raw data folder: ./criteo_raw_artifact'
    
    train_path = os.path.join(raw_dir, 'train.txt')
    assert os.path.isfile(train_path), 'Missing training data: ./criteo_raw_artifact/train.txt'
    
    test_path = os.path.join(raw_dir, 'test.txt')
    assert os.path.isfile(test_path), 'Missing test data: ./criteo_raw_artifact/test.txt'
    
    train_split_dir = os.path.join(cwd, 'criteo_train_raw_artifact')
    if os.path.isdir(train_split_dir):
        shutil.rmtree(train_split_dir, onerror=handle_error)
    os.mkdir(train_split_dir)
    
    test_split_dir = os.path.join(cwd, 'criteo_test_raw_artifact')
    if os.path.isdir(test_split_dir):
        shutil.rmtree(test_split_dir, onerror=handle_error)
    os.mkdir(test_split_dir)
    
    return cwd, train_path, test_path, train_split_dir, test_split_dir

def split_train(train_path, train_split_dir, row_per_file=2**13*100):
    """
    Split the training data into smaller files. 
    """
    feature_dict = {}
    categorical_index_start = 14
    
    partition_index = 0
    output_file = os.path.join(train_split_dir,'train-part-{}'.format(partition_index))
    logger.info('Writting {}'.format(output_file.split('/')[-1]))
    
    f_in = open(train_path, 'r')
    f_out = open(output_file, 'w')
    
    for line_index, line in enumerate(f_in):
        feature = line.rstrip('\n').split('\t')
        assert len(feature)==40, 'Encounter error in reading line {}: {}'.format(line_index, line)
        for index, value in enumerate(feature[categorical_index_start:], start=categorical_index_start):
            if index not in feature_dict:
                feature_dict[index] = {}
            if value not in feature_dict[index]:
                feature_dict[index][value] = 1
            else:
                feature_dict[index][value] += 1
        if line_index%row_per_file==0 and line_index>0:
            f_out.close()
            partition_index += 1
            output_file = os.path.join(train_split_dir,'train-part-{}'.format(partition_index))
            f_out = open(output_file, 'w')
            logger.info('Writting {}'.format(output_file.split('/')[-1]))
        f_out.write(line)
        
    f_out.close()
    f_in.close()
    
    cwd = os.getcwd()
    feature_dict_dir = os.path.join(cwd, 'criteo_feature_dict_artifact')
    if os.path.isdir(feature_dict_dir):
        shutil.rmtree(feature_dict_dir, onerror=handle_error)
    os.mkdir(feature_dict_dir)
    
    feature_dict_path = os.path.join(feature_dict_dir, 'categorical_feature_dict_raw.pkl')
    with open(feature_dict_path, 'wb') as fsave:
        pickle.dump(feature_dict, fsave)
    logger.info('Categorical feature dictionary is saved to {}'.format(feature_dict_path))
    
    embedding_index = 53
    embedding_map_dict = {}
    for index in feature_dict.keys():
        for k, v in feature_dict[index].items():
            if k and v >= 10:
                embedding_map_dict[str(index)+k] = embedding_index
                embedding_index += 1
                
    embedding_map_dict_path = os.path.join(feature_dict_dir, 'categorical_feature_map_dict.pkl')
    with open(embedding_map_dict_path, 'wb') as fsave:
        pickle.dump(embedding_map_dict, fsave)
    logger.info('Embedding map dictionary is saved to {}'.format(embedding_map_dict_path))
    
def split_test(test_path, test_split_dir, row_per_file=2**13*100):
    """
    Split the training data into smaller files. 
    """
    partition_index = 0
    output_file = os.path.join(test_split_dir,'test-part-{}'.format(partition_index))
    
    f_in = open(test_path, 'r')
    f_out = open(output_file, 'w')
    logger.info('Writting {}'.format(output_file.split('/')[-1]))
    
    for line_index, line in enumerate(f_in):
        if line_index%row_per_file==0 and line_index>0:
            f_out.close()
            partition_index += 1
            output_file = os.path.join(test_split_dir,'test-part-{}'.format(partition_index))
            f_out = open(output_file, 'w')
            logger.info('Writting {}'.format(output_file.split('/')[-1]))
        f_out.write(line)
        
    f_out.close()
    f_in.close()

def split_criteo(row_per_file=2**13*100):
    """
    Main function for spliting criteo raw data.
    """
    cwd, train_path, test_path, train_split_dir, test_split_dir = check()
    split_train(train_path, train_split_dir, row_per_file=row_per_file)
    split_test(test_path, test_split_dir, row_per_file=row_per_file)
    raw_dir = os.path.join(cwd, 'criteo_raw_artifact')
    shutil.rmtree(raw_dir, onerror=handle_error)
    
if __name__ == '__main__':
    ROW_PER_FILE = 2**15*100
    split_criteo(row_per_file=ROW_PER_FILE)
    