from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, state_fidelity
from qiskit.circuit.library import HGate, SGate, SdgGate, CXGate, CZGate, IGate, XGate, YGate, ZGate, TGate, TdgGate,RXGate, RYGate, RZGate, CRXGate
from qiskit.quantum_info import Clifford,random_clifford
from qiskit.circuit import Parameter
import numpy as np
from qiskit_aer import Aer,AerSimulator,StatevectorSimulator
import random
from tqdm import tqdm

from bugfix.clliford.clliford_gate_variables import CllifordCorrecter
from bugfix.clliford.utills import CllifordProgram, generate_inout_stabilizer_tables
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

clifford_1q_gates = [
    [XGate,],
    [HGate,],
    [SGate,],
    [IGate,],
]
clifford_2q_gates = [
    [CXGate,],
]

for gate_op in clifford_1q_gates:
    gate_op.append(Operator(gate_op[0]()).data)
    
for gate_op in clifford_2q_gates:
    gate_op.append(Operator(gate_op[0]()).data)




def matrix_distance_squared(A, B):
    """
    Returns:
        Float : A single value between 0 and 1, representing how closely A and B match.  A value near 0 indicates that A and B are the same unitary, up to an overall phase difference.
    """
    # optimized implementation
    return np.abs(1 - np.abs(np.sum(np.multiply(A, np.conj(B)))) / A.shape[0])



def replace_param_gates_with_clifford(circuit):
    new_circuit = QuantumCircuit(*circuit.qregs)
    for inst, qargs, cargs in circuit.data:
        gate_matrix = Operator(inst).data
        if len(qargs) == 1:
            nearest_gate, nearest_distance = None, float('inf')
            for clifford_gate, clifford_matrix in clifford_1q_gates:
                distance = matrix_distance_squared(gate_matrix, clifford_matrix)
                if distance < nearest_distance:
                    nearest_gate = clifford_gate
                    nearest_distance = distance
            new_circuit.append(nearest_gate(), qargs, cargs)
        elif len(qargs) == 2:
            # print(inst.name)
            assert inst.name == 'cx'
            new_circuit.append(inst, qargs, cargs)
        
        # if inst.name in replacement_mapping:
        #     new_circuit.append(replacement_mapping[inst.name], qargs, cargs)
        # else:
        #     new_circuit.append(inst, qargs, cargs)
            
    return new_circuit



# def replace_param_gates_with_clifford(circuit):
#     replacement_mapping = {
#         'rx': XGate(),  # Replace RX with X gate
#         'ry': YGate(),  # Replace RY with Y gate
#         'rz': ZGate()   # Replace RZ with Z gate
#         # Add more replacements if necessary
#     }
    
#     # print(Operator(XGate()).data)    
    
#     new_circuit = QuantumCircuit(*circuit.qregs)
#     for inst, qargs, cargs in circuit.data:
#         if inst.name in replacement_mapping:
#             new_circuit.append(replacement_mapping[inst.name], qargs, cargs)
#         else:
#             new_circuit.append(inst, qargs, cargs)
#     return new_circuit


def generate_input_circuits(n_qubits, n_states=8):
    states = []
    for _ in range(n_states):
        cllifordgate = random_clifford(n_qubits)
        state = cllifordgate.to_circuit()
        states.append(state)
    
    # Alternative method for generating input states
    # for _ in range(n_states):
    #     state = QuantumCircuit(n_qubits)
    #     for qubit in range(n_qubits):
    #         if random.random() < 0.25:
    #             state.h(qubit)
    #         if 0.25 <random.random() < 0.5:
    #             state.cx(qubit,(qubit+1)%n_qubits)
    #         if 0.5 <random.random() < 0.75:
    #             state.cx((qubit+1)%n_qubits,qubit)
    #         if random.random() > 0.75:
    #             state.x(qubit)
    #     states.append(state)
    return states



def generate_input_circuits(n_qubits, n_states=8):
    circuits = []
    for _ in range(n_states):
        cllifordgate = random_clifford(n_qubits)
        state = cllifordgate.to_circuit()
        circuits.append(state)
    
    # Alternative method for generating input states
    # for _ in range(n_states):
    #     state = QuantumCircuit(n_qubits)
    #     for qubit in range(n_qubits):
    #         if random.random() < 0.25:
    #             state.h(qubit)
    #         if 0.25 <random.random() < 0.5:
    #             state.cx(qubit,(qubit+1)%n_qubits)
    #         if 0.5 <random.random() < 0.75:
    #             state.cx((qubit+1)%n_qubits,qubit)
    #         if random.random() > 0.75:
    #             state.x(qubit)
    #     states.append(state)
    return circuits



def generate_input_states(n_qubits, n_states=8):
    states = []
    for _ in range(n_states):
        cllifordgate = random_clifford(n_qubits)
        qc = cllifordgate.to_circuit()
        backend = Aer.get_backend('statevector_simulator')
        job = backend.run(qc)
        output_state = job.result().get_statevector(qc)
        states.append(output_state)
    return states

def generate_input_states_quito(n_qubits, n_states=8):
    states = []
    for _ in range(n_states):
        cllifordgate = random_clifford(n_qubits)
        qc = cllifordgate.to_circuit()
        backend = Aer.get_backend('statevector_simulator')
        job = backend.run(qc)
        output_state = job.result().get_statevector(qc)
        states.append(output_state)
    return states

def apply_circuit(circuit, input_state):
    output_states = []
    if isinstance(input_state, list):
        input_states = input_state
    else:
        input_states = [input_state]
        
    backend = Aer.get_backend('statevector_simulator')
    
    for input_state in input_states:
        qc = QuantumCircuit(*circuit.qregs)
        qc.initialize(input_state, range(circuit.num_qubits))
        qc.compose(circuit, inplace=True)
        job = backend.run(qc)
        output_state = job.result().get_statevector(qc)
        output_states.append(output_state.data)
    return output_states

def calculate_distance(output_state, correct_output):
    return 1 - np.abs(np.vdot(output_state,correct_output))

def replace_clifford_with_param_gates(structure, param_gates):
    for i, (inst, qargs, cargs) in enumerate(structure.data):
        if inst == XGate():
            structure.data[i] = (RXGate(Parameter(f'θ_{i}')), qargs, cargs)
        elif inst == YGate():
            structure.data[i] = (RYGate(Parameter(f'θ_{i}')), qargs, cargs)
        elif inst == ZGate():
            structure.data[i] = (RZGate(Parameter(f'θ_{i}')), qargs, cargs)
        # if inst == CXGate():
        #     structure.data[i] = (CRXGate(Parameter(f'θ_{i}')), qargs, cargs)
        elif inst == SGate():
            structure.data[i] = (RZGate(Parameter(f'θ_{i}')), qargs, cargs)
        else:
            continue
    return structure

def optimize_parameters(circuit, input_states, correct_output_states, max_iterations=100):
    from qiskit_algorithms.optimizers import COBYLA,ADAM
    
    params = circuit.parameters
    param_dict = {param: random.uniform(0, 2 * np.pi) for param in params}
    
    def objective_function(param_values):
        param_dict = dict(zip(params, param_values))
        bound_circuit = circuit.bind_parameters(param_dict)
        distance = 0
        for input_state_idx, correct_output in correct_output_states.items():
            output_state = apply_circuit(bound_circuit, input_states[input_state_idx])
            distance += calculate_distance(output_state, correct_output)
        return distance/len(correct_output_states)
    
    # iteration_count = 0
    # pbar = tqdm(total=max_iterations)
    # def callback(params):
    #     nonlocal iteration_count
    #     nonlocal pbar
    #     iteration_count += 1
    #     if iteration_count % 10 == 0:
    #         pbar.update(10)
    #         pbar.set_description(f"objective: {objective_function(params)}")
    
    optimizer = COBYLA()
    initial_values = list(param_dict.values())
    result = optimizer.minimize(objective_function, initial_values)
    optimized_param_dict = dict(zip(params, result.x))
    optimized_circuit = circuit.bind_parameters(optimized_param_dict)
    
    return optimized_circuit

# Genetic Algorithm for optimizing circuit structure
def genetic_algorithm_optimize(circuit, input_states,correct_output_states ,population_size=10, generations=20, mutation_rate=0.1):
    def create_population(circuit, size):
        return [generate_bugged_circuit(circuit, n_errors= len(circuit)// 2) for _ in range(size)]

    def evaluate_fitness(circuit, correct_output_states):
        total_distance = 0
        output_states = apply_circuit(circuit, input_states)
        for output_state, correct_output in zip(output_states, correct_output_states):
            total_distance += calculate_distance(output_state, correct_output)
        return total_distance


    def select_parents(population, fitnesses):
        fitness_sum = sum(fitnesses)
        probs = [f / fitness_sum for f in fitnesses]
        parents_indices = np.random.choice(range(len(population)), size=2, p=probs)
        return population[parents_indices[0]], population[parents_indices[1]]

    def crossover(parent1, parent2):
        crossover_point = random.randint(0, len(parent1.data) - 1)
        child1_data = parent1.data[:crossover_point] + parent2.data[crossover_point:]
        child2_data = parent2.data[:crossover_point] + parent1.data[crossover_point:]
        child1 = QuantumCircuit(*parent1.qregs)
        child2 = QuantumCircuit(*parent2.qregs)
        child1.data = child1_data
        child2.data = child2_data
        return child1, child2

    def mutate(circuit, mutation_rate):
        if random.random() < mutation_rate:
            circuit = generate_bugged_circuit(circuit, n_errors=len(circuit)//10)
        return circuit

    population = create_population(circuit, population_size)
    best_circuit = None
    best_fitness = float('inf')

    for generation in range(generations):
        fitnesses = [evaluate_fitness(individual, correct_output_states) for individual in population]
        for i, fitness in enumerate(fitnesses):
            if fitness < best_fitness:
                best_fitness = fitness
                best_circuit = population[i]

        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population.extend([child1, child2])
        
        population = new_population

    return best_circuit

def clliford_repair(circuit: QuantumCircuit,  right_circuit: QuantumCircuit, input_statelen: int, d_max: int):
    
    n_qubits = circuit.n_qubits   
    wrong_program = CllifordProgram.from_circuit(circuit)
    correcter = CllifordCorrecter(n_qubits,d_max,wrong_program)
    for _ in range(input_statelen):
        right_program = CllifordProgram.from_circuit(right_circuit)
        input_stabilizer_table, output_stabilizer_table = generate_inout_stabilizer_tables(n_qubits,right_program)
        correcter.add_iout(input_stabilizer_table,output_stabilizer_table)
    fix_program = correcter.solve()
    # wrong_program.extend(fix_program)
    return fix_program.to_circuit()

if __name__ == "__main__":
    # Example usage
    correct_circuit = QuantumCircuit(2)
    param = 1.4
    correct_circuit.rx(param, 0)
    correct_circuit.rz(param, 1)
    correct_circuit.cz(0, 1)
    correct_circuit.rx(param, 0)
    correct_circuit.rz(param, 1)
    correct_circuit.cx(1, 0)
    print("Correct Circuit:")
    print(correct_circuit)

    bugged_circuit = generate_bugged_circuit(correct_circuit.copy(), error_rate=0.3)
    print("Bugged Circuit:")
    print(bugged_circuit)
    print("_"*10+'start bug fixing'+"_"*10)
    clifford_circuit = replace_param_gates_with_clifford(bugged_circuit)
    print("Clifford Circuit:")
    print(clifford_circuit)
    print('original circuit')
    print(correct_circuit)
    correct_clliford = replace_param_gates_with_clifford(correct_circuit)
    input_states = generate_input_states(2)
    correct_output_states = {i: apply_circuit(correct_circuit, state) for i,state in enumerate(input_states)}

    # fix_structure = clliford_repair(clifford_circuit, correct_clliford, 8, 2)
    fix_structure = correct_clliford

    param_gates = {
        'rx': correct_circuit.data[0][0],  # Assuming RX is the first gate in the original correct circuit
        'ry': correct_circuit.data[1][0],  # Assuming RY is the second gate in the original correct circuit
        'rz': correct_circuit.data[2][0]   # Modify as needed for actual gates in the circuit
    }

    optimized_structure = replace_clifford_with_param_gates(fix_structure, param_gates)
    optimized_circuit = optimize_parameters(optimized_structure,input_states, correct_output_states)

    print("Optimized Circuit:")
    print(optimized_circuit)
