import os
from hornbro.dataset import load_dataset
from qiskit import transpile
from tqdm import tqdm
# from hornbro.clliford.clliford_gate_variables import CllifordCorrecter
from hornbro.clliford.cllifor_gate_optimize import CllifordCorrecter
from hornbro.clliford.utills import generate_inout_stabilizer_tables
from hornbro.clliford.layercircuit import LayerCllifordProgram
from hornbro.circuit import Circuit
from hornbro.paramizefix.qiskit import replace_param_gates_with_clifford
from time import perf_counter
import numpy as np
import ray
import time
import json
import random
def generate_bugged_program(correct_program,n_errors):
    program: LayerCllifordProgram = correct_program.copy()
    gates = ['h','s','cx']
    buggy_idxes = []
    buggy_qubits = []
    for idx in np.random.choice(range(len(program)),n_errors):
        error_type = random.choice(['add', 'delete'])
        qubits = list(range(correct_program.n_qubits))
        if error_type == 'add':
            gate = random.choice(gates)
            qargs = [random.choice(qubits)]
            if gate == 'cx':
                qargs.append(random.choice([q for q in qubits if q != qargs[0]]))
            program.insert(idx, [{'name': gate, 'qubits': qargs}])
            buggy_idxes.append(int(idx))
            buggy_qubits.append([int(q) for q in qargs])
        elif error_type == 'delete' and len(program[idx]) > 0:
            index = random.randint(0, len(program[idx]) - 1)
            gate = program[idx].pop(index)
            buggy_idxes.append(int(idx))
            buggy_qubits.append([int(q) for q in gate['qubits']])
    return program, buggy_idxes, buggy_qubits

def repair(n_qubits,n_errors):
    import uuid
    uid = str(uuid.uuid4())
    metrics = {}
    repair_start = perf_counter()
    correct_program: LayerCllifordProgram = LayerCllifordProgram.random_clifford_program(n_qubits, 10)

    program, buggy_idxes, buggy_qubits = generate_bugged_program(correct_program,n_errors)
    correcter = CllifordCorrecter(program, time_out_eff = 10, is_soft= False, insert_layer_indexes= np.random.choice([i for i in range(len(program))],min(3,len(program)),replace=False).tolist())
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
    insert_idxes = correcter.insert_idxes
    insert_qubits = correcter.insert_qubits
    end = perf_counter()
    metrics[f'clliford_time'] = end - start
    print(f"clliford_time:", end - start)
    # calculate metrics
    similarity_score = 0
    for idx, qubits in zip(buggy_idxes, buggy_qubits):
        if ((idx in insert_idxes) or (idx+1 in insert_idxes) or (idx-1 in insert_idxes)) and (qubits in insert_qubits or (qubits[::-1] in insert_qubits)):
            similarity_score += 1
    metrics['similarity_score'] = similarity_score / len(buggy_idxes)
    metrics['repair_time'] = end - repair_start
    metrics['repair_n_gates'] = sum([len(layer) for layer in solve_program]) - sum([len(layer) for layer in program])
    metrics['repair_n_layers'] = len(solve_program) - len(program)
    metrics['repair_n_qubits'] = n_qubits
    metrics['repair_n_errors'] = n_errors
    metrics['right_circuit'] = correct_program.to_list()
    metrics['buggy_circuit'] = program.to_list()
    metrics['buggy_idxes'] = buggy_idxes
    metrics['buggy_qubits'] = buggy_qubits
    print(f"repair_time:", end - repair_start)
    print(f"similarity_score:", similarity_score / len(buggy_idxes))

         
    ## save the metrics to json
    save_path = f'results/LocalizeRepair/haierBro/{n_qubits}_errors_{n_errors}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(save_path+f'metrics{uid}.json', 'w') as f:
        json.dump(metrics, f)


import traceback
@ray.remote
def repair_remote(*args, **kwargs):
    try:
        # raise Exception('test')
        return repair(*args, **kwargs)
    except Exception as e:
        with open('errorlocation.log', 'a') as f:
            traceback.print_exc(file=f)
        return None
if __name__ == '__main__':
    # repair(5,2)
    futures =[repair_remote.remote(n_qubits, n_errors) for n_qubits in [5,10,15] for n_errors in [2,4,6]]    
    ray.get(futures)