from qiskit_aer import Aer
from qiskit.quantum_info import Clifford,random_clifford,Statevector
from itertools import product
import numpy as np
def get_inititial_statevector(circuit):
    backend = Aer.get_backend('statevector_simulator')
    job = backend.run(circuit)
    statevector = job.result().get_statevector(circuit)
    return statevector.data

def generate_input_states_clifford(n_qubits, n_states=8):
    states = []
    for _ in range(n_states):
        cllifordgate = random_clifford(n_qubits)
        qc = cllifordgate.to_circuit()
        backend = Aer.get_backend('statevector_simulator')
        job = backend.run(qc)
        output_state = job.result().get_statevector(qc)
        states.append(output_state.data)
    return states

def generate_input_states_quito(n_qubits, n_states=8):
    labels = [''.join(state) for state in product(['0','1'],repeat=n_qubits)]
    np.random.shuffle(labels)
    input_states =  [Statevector.from_label(label).data for label in  labels[:n_states]]
    return input_states


def generate_input_states_QuSBT(n_qubits, n_states=8):
    labels = [''.join(state) for state in product(['0','1','+','r'],repeat=n_qubits)]
    np.random.shuffle(labels)
    input_states =  [Statevector.from_label(label).data for label in  labels[:n_states]]
    return input_states

def generate_input_states_random(n_qubits, n_states=8):
    states = []
    for _ in range(n_states):
        real = np.random.random(2**n_qubits)
        img = np.random.random(2**n_qubits)
        state = real+img*1j
        state = state/np.linalg.norm(state)
        states.append(state)
    return states


    '''
        通过对process 进行多次初始化，得到 多个 output,其线性叠加对应与整个process 的函数
        process: 电路的中间表示
        out_qubits: 输出的qubits
        input_qubits: 输入的qubits
        method: 采样的方法
        base_num: 采样的次数
        initial_label: 采样的初始态
        device: 采样的设备
            ibmq: ibm 的 真实量子计算机
            simulate: 本地模拟器
        return:
            initialState: statevector
            new_circuit: initialized circuits
    '''
    n_qubits = len(input_qubits)
    if clifford_tomography:
        get_output = lambda  state: ExcuteEngine.excute_on_pennylane(process,type='definedinput',shots=1000,output_qubits=output_qubits,input_qubits=input_qubits,input_state=state)
    get_output = lambda  state_circuit: ExcuteEngine.output_state_tomography(state_circuit+process,input_qubits,device=device)
    if method == 'base':
        labels = [''.join(state) for state in product(['0','1','+','r'],repeat=n_qubits)][:base_num]
        input_states =  [Statevector.from_label(state).data for state in  labels]
        input_circuits = [get_circuit_from_label(label,n_qubits) for label in labels]
    if method == 'basis':
        labels = [''.join(state) for state in product(['0','1'],repeat=n_qubits)][:base_num]
        input_states =  [Statevector.from_label(state).data for state in  labels]
        input_circuits = [get_circuit_from_label(label,n_qubits) for label in labels]
    elif method == 'random':
        input_circuits = list(map(get_inititial_circuit,tqdm([n_qubits]*base_num,desc='producing input circuits for random sampling')))
        input_states = list(map(get_inititial_statevector,tqdm(input_circuits,desc='producing input states for random sampling')))
        input_circuits = list(map(lambda x: qiskit_circuit_to_layer_cirucit(x),input_circuits))
    
    output_states = list(map(get_output,tqdm(input_circuits,desc='producing output states')))
    return input_states , output_states