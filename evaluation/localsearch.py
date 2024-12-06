import os
from hornbro.dataset import load_dataset
from qiskit import transpile
from tqdm import tqdm
from qiskit import QuantumCircuit
from hornbro.circuit import Circuit
# from hornbro.clliford.clliford_gate_variables import CllifordCorrecter
from hornbro.clliford.cllifor_gate_parr_multilayer import CllifordCorrecter
from hornbro.clliford.utills import generate_inout_stabilizer_tables
from hornbro.clliford.layercircuit import LayerCllifordProgram
from hornbro.paramizefix.pennylane_siwei import generate_input_states as generate_input_states_pennylane, GateParameterOptimizer
from hornbro.paramizefix.qiskit import apply_circuit, generate_bugged_circuit, generate_input_states, optimize_parameters, replace_param_gates_with_clifford
import json
from time import perf_counter
import numpy as np
from jax import numpy as jnp
import ray
from hornbro.paramizefix.qiskit import genetic_algorithm_optimize

def repair(correct_circuit:QuantumCircuit, bugged_circuit,algoname,n_qubits, n_errors,idx):
    basic_gates = ['cx','u']  # 'x', 'y', 'z', 'cz', 
    
    savepath = f'results/local_search/{algoname}/qubits_{n_qubits}_errors_{n_errors}/'
    ## save the metrics to json
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    metrics = {}
    repair_start = perf_counter()
    for i in range(5):
        print(f"repairing {i+1}th time")
        # TODO: 先梯度下降
        
        # 先找 patches
        correct_clliford = replace_param_gates_with_clifford(correct_circuit) 
        correct_clliford = transpile(correct_clliford, basis_gates=['x','y','s','h','cx','id'], optimization_level=0)
        bugged_clifford = replace_param_gates_with_clifford(bugged_circuit) 
        bugged_clifford = transpile(bugged_clifford, basis_gates=['x','y','s','h','cx','id'], optimization_level=0)
        
        start = perf_counter()
        input_states = generate_input_states(n_qubits, n_states=min(2**n_qubits, 100))
        target_output_states = []
        for input_state in input_states:
            target_output_states.append(Circuit(correct_circuit).run(input_state.data))
        
        find_program = genetic_algorithm_optimize(bugged_clifford,input_states,target_output_states,population_size=10,generations=20)
        end = perf_counter()
        metrics[f'clliford_time_{i}'] = end - start
        print(f"clliford_time_{i}:", end - start)
        # 找不到就用旧的
        
        correct_circuit_inverse = Circuit(correct_circuit.inverse())
        optimizer = GateParameterOptimizer.from_circuit(find_program)
        input_states = generate_input_states(n_qubits, n_states=min(2**n_qubits//2, 100))
        start = perf_counter()
        repaired_circuit, history = optimizer.optimize_mirror(correct_circuit_inverse, input_states, n_epochs=1000, n_batch = 20)
        total_epochs = history.epoch
        end = perf_counter()
        metrics[f'param_time_{i}'] = end - start
        metrics[f'epochs_{i}'] = total_epochs
        metrics[f'num_params_{i}'] = optimizer.get_n_params()
        input_test_states = generate_input_states(n_qubits, n_states=min(2**n_qubits//2, 100))
        dist = optimizer.test_mirror(correct_circuit_inverse,history.best_params,input_test_states)
        print('test mirror', dist)
        if dist < 0.1:
            break
        # correct_circuit = correct_circuit.to_qiskit()
        bugged_circuit = transpile(repaired_circuit.to_qiskit(), basis_gates=basic_gates, optimization_level=3)
        ## save history by pikcle
        import pickle
        with open(os.path.join(savepath , f'history{idx}_repair_{i}.pkl'),'wb') as f:
            pickle.dump(history,f)

    metrics['distance'] = float(dist)
    metrics['repaired_circuit'] = repaired_circuit.to_json()
    metrics['correct_circuit'] = Circuit(correct_circuit).to_json()
    metrics['total_time'] = perf_counter() - repair_start
    metrics['repaired_times'] = i+1
    if dist < 0.1:
        
        metrics['state'] = 'success'
    else:
        metrics['state'] = 'failed'
    
    with open(savepath +f'metrics{idx}.json', 'w') as f:
        json.dump(metrics, f)

import traceback
@ray.remote
def repair_remote(*args, **kwargs):
    try:
        # raise Exception('test')
        return repair(*args, **kwargs)
    except Exception as e:
        with open('errorlocal.log', 'a') as f:
            traceback.print_exc(file=f)
        return None
if __name__ == '__main__':
    ray.init()
    futures = []
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetdir', type=str, default='dataset/')
    args = parser.parse_args()
    datasetdir = args.datasetdir
    for data in load_dataset(datasetdir):
        algoname = data['algorithm']
        n_qubits = data['n_qubits']
        n_errors = data['n_errors']
        qc_corrects, qc_bugs = data['right_circuits'],data['wrong_circuits']
        futures +=[repair_remote.remote(qc_correct, qc_bug,algoname,n_qubits, n_errors,id=i) for i, (qc_correct, qc_bug) in enumerate(list(zip(qc_corrects, qc_bugs))[:4])]
    
    ray.get(futures)
