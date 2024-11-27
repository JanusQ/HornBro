import numpy as np
from qiskit.quantum_info import random_clifford,Clifford
from qiskit.circuit.library import HGate, SGate, SdgGate, CXGate, CZGate, IGate, XGate, YGate, ZGate, TGate, TdgGate,RXGate, RYGate, RZGate, CRXGate

from qiskit import QuantumCircuit, transpile
import random

def custom_random_circuit(n_qubits, depth, gate_set):
    qc = QuantumCircuit(n_qubits)
    depth = depth // 2
    for _ in range(depth):
        for qubit in range(n_qubits):
            gate = np.random.choice(gate_set)
            if gate == 'h':
                qc.h(qubit)
            elif gate == 'x':
                qc.x(qubit)
            elif gate == 'y':
                qc.y(qubit)
            elif gate == 'z':
                qc.z(qubit)
            elif gate == 's':
                qc.s(qubit)
            elif gate == 'rx':
                theta = np.random.uniform(0, np.random.rand() * 2 * np.pi)
                qc.rx(theta, qubit)
            elif gate == 'ry':
                theta = np.random.uniform(0, np.random.rand() * 2 * np.pi)
                qc.ry(theta, qubit)
            elif gate == 'rz':
                theta = np.random.uniform(0, np.random.rand() * 2 * np.pi)
                qc.rz(theta, qubit)
            elif gate == 'cx':
                if n_qubits > 1:
                    target = (qubit + 1) % n_qubits
                    qc.cx(qubit, target)
            elif gate == 'cz':
                if n_qubits > 1:
                    target = (qubit + 1) % n_qubits
                    qc.cz(qubit, target)
        # qc.barrier()
    qc = qc.compose(qc.inverse())
    for i in range(n_qubits):
        qc.x(i)
    # qc = transpile(qc, basis_gates=gate_set, optimization_level=3)
    return qc

def generate_bugged_circuit(correct_circuit, n_errors: int=2):
    bugged_circuit = correct_circuit.copy()
    # n_gates = len(correct_circuit.data)
    # n_errors = int(n_gates * error_rate)
    gates = [HGate(), XGate(), YGate(), ZGate(), CXGate(),  SGate()]  # CZGate(),
    single_qubit_gates = [HGate(), XGate(), YGate(), ZGate(), SGate()]
    two_qubit_gates = [CXGate()]  # , CZGate()
    # param_gates = [RXGate(Parameter('θ')), RYGate(Parameter('θ')), RZGate(Parameter('θ'))]
    for _ in range(n_errors):
        error_type = random.choice(['add', 'delete', 'replace'])
        qubits = list(range(correct_circuit.num_qubits))
        if error_type == 'add':
            gate = random.choice(gates)
            qargs = [random.choice(qubits)]
            if gate.num_qubits == 2:
                qargs.append(random.choice([q for q in qubits if q != qargs[0]]))
            bugged_circuit.append(gate, qargs)
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
