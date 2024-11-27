import os
from hornbro.dataset import load_dataset
from qiskit import transpile
from tqdm import tqdm
# from hornbro.clliford.clliford_gate_variables import CllifordCorrecter
from hornbro.clliford.cllifor_gate_parr_multilayer import CllifordCorrecter
from hornbro.clliford.utills import generate_inout_stabilizer_tables
from hornbro.clliford.layercircuit import LayerCllifordProgram
from hornbro.circuit import Circuit
from hornbro.paramizefix.pennylane_siwei import GateParameterOptimizer
from hornbro.paramizefix.qiskit import replace_param_gates_with_clifford
import json
from time import perf_counter
import numpy as np
import ray
import time

def repair(correct_circuit, bugged_circuit,algoname,n_qubits, n_errors,id):
    basic_gates = ['cx','u']  # 'x', 'y', 'z', 'cz', 
    
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

        program: LayerCllifordProgram = LayerCllifordProgram.from_qiskit_circuit(bugged_clifford)  # TODO: n_layers
        correct_program = LayerCllifordProgram.from_qiskit_circuit(correct_clliford)  # TODO: n_layers
        correcter = CllifordCorrecter(program, time_out_eff = 1, is_soft= True, insert_layer_indexes= np.random.choice([i for i in range(len(program))],min(3,len(program)),replace=False).tolist())
        inputs, outputs = [], []
        for _ in tqdm(range(min(2**n_qubits//2, 20))):
            input_stabilizer_table, output_stabilizer_table = generate_inout_stabilizer_tables(
                program.n_qubits, correct_program)
            middle_stabilizer_table = program.output_stablizers(input_stabilizer_table)
            inputs.append(middle_stabilizer_table)
            outputs.append(output_stabilizer_table)
        correcter.add_iout(inputs, outputs)
        start = perf_counter()
        solve_program = correcter.solve()  # 60
        end = perf_counter()
        metrics[f'clliford_time_{i}'] = end - start
        print(f"clliford_time_{i}:", end - start)
        new_program = program.copy()
        # 找不到就用旧的
        if solve_program:
            new_program.extend(solve_program)
        find_program = new_program

        correct_circuit = Circuit(correct_circuit)
        optimizer = GateParameterOptimizer.from_circuit(find_program)
        # input_states = generate_input_states(n_qubits, n_states=min(2**n_qubits//2, 100))
        start = perf_counter()
        repaired_circuit, history = optimizer.optimize_minor(correct_circuit, n_epochs=1000)
        dist = history.min_loss
        total_epochs = history.epoch
        end = perf_counter()
        metrics[f'param_time_{i}'] = end - start
        metrics[f'epochs_{i}'] = total_epochs
        metrics[f'num_params_{i}'] = optimizer.get_n_params()

        
        if dist < 0.1:
            break
        
        correct_circuit = correct_circuit.to_qiskit()
        bugged_circuit = transpile(repaired_circuit.to_qiskit(), basis_gates=basic_gates, optimization_level=3)

    metrics['distance'] = float(dist)
    metrics['repaired_circuit'] = repaired_circuit.to_json()
    metrics['total_time'] = perf_counter() - repair_start
    metrics['repaired_times'] = i+1
    if dist < 0.1:
        
        metrics['state'] = 'success'
        
        ## save the repaired circuit as qasm
        if not os.path.exists(f'data/repaired_circuits/{n_qubits}_errors_{n_errors}/'):
            os.makedirs(f'data/repaired_circuits/{n_qubits}_errors_{n_errors}/')
        repaired_circuit.to_qasm(f'data/repaired_circuits/{n_qubits}_errors_{n_errors}/repaired_circuit{id}.qasm')
    else:
        metrics['state'] = 'failed'
        
    ## save the metrics to json
    save_path = f'results/Effectiveness/haierBro/{algoname}/{n_qubits}_errors_{n_errors}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(save_path+f'metrics{id}.json', 'w') as f:
        json.dump(metrics, f)

import traceback
@ray.remote
def repair_remote(*args, **kwargs):
    try:
        # raise Exception('test')
        return repair(*args, **kwargs)
    except Exception as e:
        with open('error.log', 'a') as f:
            traceback.print_exc(file=f)
        return None
if __name__ == '__main__':
    futures = []
    for data in load_dataset('dataset/'):
        algoname = data['algorithm']
        n_qubits = data['n_qubits']
        n_errors = data['n_errors']
        qc_corrects, qc_bugs = data['right_circuits'],data['wrong_circuits']
        futures +=[repair_remote.remote(qc_correct, qc_bug,algoname,n_qubits, n_errors,id=i) for i, (qc_correct, qc_bug) in enumerate(zip(qc_corrects, qc_bugs))]
        # [repair(qc_correct, qc_bug,n_qubits, n_errors,id=i) for i, (qc_correct, qc_bug) in enumerate(zip(qc_corrects, qc_bugs))]
        time.sleep(120)
    
    ray.get(futures)