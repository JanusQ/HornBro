
from .utills import custom_random_circuit,generate_bugged_circuit 
from qiskit import transpile
import pickle
import os
from mqt.bench import get_benchmark
import argparse
import ray
from .grover import create_circuit as grover_circuit

@ray.remote
def get_error_right_circuit_pair(n_qubits,n_errors,algoname="random", gate_set=['h', 'cx', 'rx', 'ry', 'rz']):
    if algoname == "random":
        correct_circuit = custom_random_circuit(n_qubits,5,gate_set = gate_set)
    if algoname == "grover":
        correct_circuit = grover_circuit(n_qubits)
        correct_circuit = transpile(correct_circuit, basis_gates=['h','x','y','z','u3','cx','cz'], optimization_level=1)
        correct_circuit.remove_final_measurements()
                ## remove barrier
        for data in correct_circuit.data:
            if data[0].name == 'barrier':
                correct_circuit.data.remove(data)
    else:
        correct_circuit =  get_benchmark(benchmark_name=algoname, level="indep", circuit_size=n_qubits)
        correct_circuit = transpile(correct_circuit, basis_gates=['h','x','y','z','u3','cx','cz'], optimization_level=1)
        correct_circuit.remove_final_measurements()
                ## remove barrier
        for data in correct_circuit.data:
            if data[0].name == 'barrier':
                correct_circuit.data.remove(data)
    bugged_circuit = generate_bugged_circuit(correct_circuit.copy(), n_errors=n_errors)
    
    return correct_circuit,bugged_circuit

def generate_dataset(algorithm_names, n_qubits, n_errors,root_dir_name = 'dataset',n_cases=50):
    for algorithm_name in algorithm_names:
        for n_qubit in n_qubits:
            for n_error in n_errors:
                qc_correct, qc_error = [],[]
                print(f"generate {algorithm_name} {n_qubit} {n_error} dataset")
                results = ray.get([get_error_right_circuit_pair.remote(n_qubit,n_error,algoname=algorithm_name) for _ in range(n_cases)])
                for result in results:
                    qc_correct.append(result[0])
                    qc_error.append(result[1])
                qcs = qc_correct + qc_error
                qcs_trans = transpile(qcs, basis_gates=['h','u3','cx'], optimization_level=1)
                dirname = f'{root_dir_name}/{algorithm_name}/{n_qubit}/{n_error}/'
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                with open(os.path.join(dirname , 'qcs.pkl'),'wb') as f:
                    pickle.dump(qcs_trans,f)

def main():
    algos_supported = [
                        #     'ae',
                        #  'dj',
                        'ghz',
                        'graphstate',
                        'qaoa',
                        'qft',
                        'qnn',
                        'qpeexact',
                        'grover',
                        'qwalk-noancilla',
                        'vqe',
                        'wstate',
                        'shor',
                        'groundstate',
                        'routing',
                        'tsp']
    parser = argparse.ArgumentParser(description="Generate dataset including right circuit and buggy circuit")
    parser.add_argument('--qubits', type=int, nargs='+', required=True, help='A list of number of qubits to generate dataset for.')
    parser.add_argument('--errors', type=int, nargs='+', required=True, help='A list of number of errors to generate dataset for.')
    parser.add_argument('--algorithms', type=str, nargs='+', required=True, help='A list of number of errors to generate dataset for.', choices=algos_supported)
    parser.add_argument('--cases', type=int, default=50, help='Number of cases to generate for each algorithm and qubit number.')
    parser.add_argument('--dirname', type=str, default='dataset', help='Root directory name to store the dataset.')
    args = parser.parse_args()
    print(f"generate the dataset in {args.algorithms}!")
    generate_dataset(args.algorithms, args.qubits, args.errors, n_cases=args.cases, root_dir_name=args.dirname)
    print(f"finish the generation of dataset in {args.algorithms}!")