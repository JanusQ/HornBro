
from .utills import custom_random_circuit,generate_bugged_circuit 
from qiskit import transpile
import pickle
import os
from mqt.bench import get_benchmark
import argparse
from qiskit.circuit.library import HGate, SGate,CXGate,XGate, YGate, ZGate, TGate, TdgGate,RXGate, RYGate, RZGate, CRXGate
import random
def generate_bugged_circuit(correct_circuit, n_errors: int=2):
    bugged_circuit = correct_circuit.copy()
    # n_gates = len(correct_circuit.data)
    # n_errors = int(n_gates * error_rate)
    gates = [HGate(), XGate(), CXGate(),  SGate()]  
    single_qubit_gates = [HGate(), XGate(),SGate()]
    two_qubit_gates = [CXGate()]  
    buggy_idxes = []
    for _ in range(n_errors):
        error_type = random.choice(['add', 'delete', 'replace'])
        qubits = list(range(correct_circuit.num_qubits))
        if error_type == 'add':
            gate = random.choice(gates)
            qargs = [random.choice(qubits)]
            if gate.num_qubits == 2:
                qargs.append(random.choice([q for q in qubits if q != qargs[0]]))
            insert_idx = random.randint(0, len(bugged_circuit.data)-1)
            bugged_circuit.data.insert(insert_idx, (gate, qargs, []))
            buggy_idxes.append(insert_idx)
        elif error_type == 'delete' and len(bugged_circuit.data) > 0:
            index = random.randint(0, len(bugged_circuit.data) - 1)
            bugged_circuit.data.pop(index)
        elif error_type == 'replace' and len(bugged_circuit.data) > 0:
            index = random.randint(0, len(bugged_circuit.data) - 1)
            qargs = bugged_circuit.data[index][1]
            if len(qargs) == 1:
                gate = random.choice(single_qubit_gates)
            else:
                gate = random.choice(two_qubit_gates)
            bugged_circuit.data[index] = (gate, qargs, []) 

    return bugged_circuit

def get_error_right_clliford_pair(n_qubits,n_errors):
    correct_circuit = custom_random_circuit(n_qubits,5,gate_set = ['s','h','cx','x'] )
    bugged_circuit = generate_bugged_circuit(correct_circuit.copy(), n_errors=n_errors)
    
    return correct_circuit,bugged_circuit
