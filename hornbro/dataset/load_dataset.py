import pickle
import os
from os.path import join, getsize
def load_dataset(dataset_dir= 'dataset/'):
    import os  
    for root, dirs, files in os.walk(dataset_dir): 
        for file in files:
            if file.endswith('.pkl'):
                with open(join(root, file), 'rb') as f:
                    data = pickle.load(f)
                    length = len(data)
                    right_circuits, wrong_circuits = data[:length//2], data[length//2:]
                    ## 解析root 信息  algorithm/n_qubits/n_errors
                    algorithm, n_qubits, n_errors = root.split('/')[-3:]
                    datainfo = {'algorithm': algorithm, 'n_qubits': int(n_qubits), 'n_errors': int(n_errors)}
                    datainfo['size'] = getsize(join(root, file))
                    datainfo['length'] = length//2
                    datainfo['right_circuits'] = right_circuits
                    datainfo['wrong_circuits'] = wrong_circuits

                    yield datainfo


    