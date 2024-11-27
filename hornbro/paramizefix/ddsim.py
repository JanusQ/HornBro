from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
import numpy as np
from mqt import ddsim
import pennylane as qml
import numpy as np
from jax import config
from qiskit import QuantumCircuit
from hornbro.circuit import Circuit
from tqdm import tqdm
from hornbro.optimizer import OptimizingHistory

config.update("jax_enable_x64", True)

def get_probs_by_ddsim(circuit):

    backend = ddsim.DDSIMProvider().get_backend("qasm_simulator")

    job = backend.run(circuit, shots=10000)
    counts = job.result().get_counts(circuit)

    return counts.get('0'*circuit.num_qubits, 0) / 10000

def circuit_to_qiskit(circuit: Circuit):
    qiskit_circuit = QuantumCircuit(circuit.n_qubits)
    num_params = 0
    for layer in circuit:
        for gate in layer:
            if gate['name'].lower() == 'x':
                qiskit_circuit.x(gate['qubits'][0])
                
            elif gate['name'].lower() == 's':
                qiskit_circuit.s(gate['qubits'][0])
            
            elif gate['name'].lower() == 'z':
                qiskit_circuit.z(gate['qubits'][0])
                
            elif gate['name'].lower() == 'sdg':
                qiskit_circuit.sdg(gate['qubits'][0])
                
            elif gate['name'].lower() == 'cx' or gate['name'].lower() == 'cnot':
                qiskit_circuit.cx(gate['qubits'][0], gate['qubits'][1])
            elif gate['name'].lower() == 'cz':
                qiskit_circuit.cz(gate['qubits'][0], gate['qubits'][1])
            elif gate['name'].lower() == 'rz':
                param = Parameter('param'+str(num_params))
                qiskit_circuit.rz(param, gate['qubits'][0])
                num_params += 1
            elif gate['name'].lower() == 'id':
                pass
            else:
                pass
                # raise ValueError("Unsupported gate: ", gate['name'])
    
    
    
    return qiskit_circuit
                
                



class GateParameterOptimizer:
    def __init__(self, circuit: Circuit, right_circuit=None):
        self.circuit = circuit
    @staticmethod
    def from_circuit(circuit: Circuit):
        return GateParameterOptimizer(circuit)

    def get_n_params(self):
        n_params = 0
        for gates in self.circuit:
            for gate in gates:
                if gate['name'].lower() == 'rz':
                    n_params += 1
        return n_params

    def optimize_minor(self, original_circuit: Circuit, n_epochs=50, n_batch=10, lr=0.1):
        # input_states = jnp.array(input_states)
        from qiskit_algorithms.optimizers import COBYLA,ADAM
        
        counts = 0
        def objective_function(param_values):
            nonlocal counts
            counts += 1
            templecircuit = circuit_to_qiskit(self.circuit)
            circuit = templecircuit.assign_parameters(param_values)
            original_cir = circuit_to_qiskit(original_circuit)
            minor_circuit = circuit.compose(original_cir.inverse())
            zero_prob = get_probs_by_ddsim(minor_circuit)
            return 1 - zero_prob
    
        optimizer = COBYLA()
        n_params = self.get_n_params()
        templecircuit = circuit_to_qiskit(self.circuit)
        initial_values = np.random.uniform(0, 2*np.pi,n_params)
        history = OptimizingHistory(initial_values, learning_rate= lr)
        result = optimizer.minimize(objective_function, initial_values)
        optimized_circuit = templecircuit.assign_parameters(result.x)
        history.epoch = counts
        history.min_loss =  objective_function(result.x)
        return optimized_circuit, history
