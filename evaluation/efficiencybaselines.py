import pickle 
import qiskit
from qiskit_aer import Aer
from qiskit import QuantumCircuit
import numpy as np 
from qiskit.synthesis.unitary import qsd
import json
from bqskit import compile
import time
from openai import OpenAI
import sys
from time import perf_counter
import os
from hornbro.circuit import Circuit
from hornbro.dataset import load_dataset
from hornbro.paramizefix.qiskit import apply_circuit, generate_input_states, genetic_algorithm_optimize

# Dictionary to hold the registered functions
baselines = {}

# Decorator to assign name and register the function
def register_function(name):
    def decorator(func):
        func.name = name  # Assign the name attribute
        baselines[name] = func  # Register the function
        return func
    return decorator

def generate_python_source(qc):
    code = "from qiskit import QuantumCircuit\n\n"
    code += f"circuit = QuantumCircuit({qc.num_qubits}, {qc.num_clbits})\n"
    
    for instruction, qargs, cargs in qc.data:
        gate_name = instruction.name
        if gate_name == 'u3':
            gate_name = 'u'
        qargs_str = ', '.join([f"{q._index}" for q in qargs])
        cargs_str = ', '.join([f"{c._index}" for c in cargs])
        params = ', '.join([str(param) for param in instruction.params])

        if gate_name == 'measure':
            code += f"circuit.measure([{qargs_str}], {cargs_str})\n"
        elif params:
            code += f"circuit.{gate_name}({params}, {qargs_str})\n"
        else:
            code += f"circuit.{gate_name}({qargs_str})\n"
    
    code += "print(circuit)\n"
    print(code)
    return code

def convert_python_code_to_qiskit(code):
    ## get the text start from "```python" to "```"
    code = code.split("```python")[1].split("```")[0].replace('u3','u')
    ## remove the "import" and "from" lines
    # code = "\n".join([line for line in code.split("\n") if not line.startswith("import") and not line.startswith("from")])
    local_namespace = {}
    # print(code)
    # 执行代码并将其结果保存在 local_namespace 中
    exec(code, {}, local_namespace)
    
    # 尝试获取 circuit 变量
    circuit = local_namespace.get('circuit')
    
    if circuit is None:
        raise NameError("The code does not define a 'circuit' variable.")
     
    return circuit

@register_function('chatgpt')
def chatgpt_baseline(algoname,qc_correct, qc_bug,n_qubits, n_errors,i):
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    res = {}
    source_text = generate_python_source(qc_bug)
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"There are some bugs in the Grover quantum circuit represented by python code using qiskit.  Can you help me fix it"},
            {"role": "system", "content": "Ignore missing measurements and decimal point accuracy issues, return only python code for the repaired circuit"},
            {"role": "user", "content": source_text}
        ]
        )
    answer = completion.choices[0].message.content
    # print("answer: "+ answer)
    

    res['correct'] = generate_python_source(qc_correct)
    res['bug'] = source_text
    res['fixed'] = answer
    repaired_circuit = convert_python_code_to_qiskit(answer)
    res['repaired_circuit']= Circuit(repaired_circuit).to_json()
    dist = 1- check_circuit(qc_correct, convert_python_code_to_qiskit(answer))
    res['distance'] = dist
    res['state'] = 'success' if dist < 0.1 else 'failed'
    print(f"{i}th dist: {dist}")
    save_path = f'results/chatgpt/{algoname}/qubits_{n_qubits}_errors_{n_errors}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    json.dump(res, open(save_path+f"metrics{i}.json","w"))
    
    
@register_function('qsd')
def qsd_baseline(algoname,qc_correct, qc_bug,n_qubits, n_errors,i):
    backend = Aer.get_backend('unitary_simulator')
    res = {}
    U_correct = backend.run(qc_correct).result().get_unitary()
    U_bug = backend.run(qc_bug).result().get_unitary()
    U_correct = np.array(U_correct)
    U_bug = np.array(U_bug)
    X = np.linalg.solve(U_bug, U_correct)
    
    start_time = time.time()
    qc_syn = qsd.qs_decomposition(X)
    total_time  = time.time() - start_time
    
    res['n_gates'] = len(qc_syn)
    res['depth'] = qc_syn.depth()
    res['total_time'] = total_time
    res['repaired_circuit']= Circuit(qc_syn).to_json()
    dist = 0
    res['distance'] =  dist
    res['state'] = 'success' if dist < 0.1 else 'failed'
    save_path = f'results/qsd/{algoname}/qubits_{n_qubits}_errors_{n_errors}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    json.dump(res, open(save_path+f"metrics{i}.json","w"))
        

@register_function('qfast')
def qfast_baseline(algoname,qc_correct, qc_bug,n_qubits, n_errors,i):
    # qcs = load_dataset(n_qubits, n_errors)
    # assert len(qcs) == 100
    # qc_corrects, qc_bugs = qcs[:50], qcs[50:]
    # i=0
    backend = Aer.get_backend('unitary_simulator')
    # for qc_correct, qc_bug in zip(qc_corrects, qc_bugs):
    res = {}
    U_correct = backend.run(qc_correct).result().get_unitary()
    U_bug = backend.run(qc_bug).result().get_unitary()
    U_correct = np.array(U_correct)
    U_bug = np.array(U_bug)
    X = np.linalg.solve(U_bug, U_correct)
    
    start_time = time.time()
    qc_syn = compile(X, max_synthesis_size=5)
    total_time  = time.time() - start_time
    
    # res['qc_correct'] = qc_correct.__str__()
    # res['qc_bug'] = qc_bug.__str__()
    # res['U_correct'] = U_correct.tolist().__str__()
    # res['U_bug'] = U_bug.tolist().__str__()
    # res['U'] = X.tolist().__str__()
    # res['qc_syn'] = qc_syn.__str__()
    res['n_gates'] = len(qc_syn)
    res['depth'] = qc_syn.depth()
    res['total_time'] = total_time
    res['repaired_circuit']= Circuit(qc_syn).to_json()
    dist = 0
    res['distance'] =  dist
    res['state'] = 'success' if dist < 0.1 else 'failed'
    save_path = f'results/qfast/{algoname}/qubits_{n_qubits}_errors_{n_errors}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    json.dump(res, open(save_path+f"metrics{i}.json","w"))
    
def check_circuit(qc_correct, qc_syn):
    if qc_syn.num_qubits != qc_correct.num_qubits or qc_syn.num_clbits != qc_correct.num_clbits:
        return False
    minnor_circuit = qc_correct.copy()
    ## append the gates of qc_syn reversily to the qc_correct and check if the output state is the same as the input state
    # for gate in qc_syn.data[::-1]:
    #     minnor_circuit.data.append(gate)
    syn_qubits = {bit: i for i, bit in enumerate(qc_syn.qubits)}
    correct_qubits = minnor_circuit.qubits
    
    # 遍历 qc_syn 中的门操作，手动映射到 minnor_circuit
    for gate in qc_syn.data[::-1]:
        # 获取门操作作用的比特
        qubits = [correct_qubits[syn_qubits[bit]] for bit in gate.qubits]
        ## change the gate u3 to u
        if gate.operation.name == 'u3':
            gate.operation.name = 'u'
        # 将门操作应用到 minnor_circuit 的映射比特上
        minnor_circuit.append(gate.operation, qubits)
    # excute the circuit and get the output state
    minnor_circuit.measure_all()
    backend = Aer.get_backend('qasm_simulator')
    job = backend.run(minnor_circuit)
    output_state = job.result().get_counts()
    ## check if the output state is the same as the input state
    return getattr(output_state,'0'*qc_correct.num_qubits,0)
    

@register_function('kimi')
def kimi_baseline(algoname,qc_correct, qc_bug,n_qubits, n_errors,i):
    res = {}
    source_text = generate_python_source(qc_bug)
    api_key = os.getenv("MOONSHOT_API_KEY") # 
    client = OpenAI(
        api_key= api_key, 
        base_url="https://api.moonshot.cn/v1",
    )
    completion = client.chat.completions.create(
        model = "moonshot-v1-128k",
        messages = [
            {"role": "system", "content": "以下代码所表示量子电路中存在bug，你能帮我修复它吗"},
            {"role": "system", "content": "忽略缺失测量和小数点精度问题,仅返回修复后的电路的python代码"},
            {"role": "user", "content": source_text}
        ],
        temperature = 0.3,
    )
    
    answer = completion.choices[0].message.content
    
    res['correct'] = generate_python_source(qc_correct)
    res['bug'] = source_text
    res['fixed'] = answer
    repaired_circuit = convert_python_code_to_qiskit(answer)
    res['repaired_circuit']= Circuit(repaired_circuit).to_json()
    dist = 1 - check_circuit(qc_correct, convert_python_code_to_qiskit(answer))
    res['distance'] =  dist
    res['state'] = 'success' if dist < 0.1 else 'failed'
    save_path = f'results/kimi/{algoname}/qubits_{n_qubits}_errors_{n_errors}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    json.dump(res, open(save_path+f"metrics{i}.json","w"))

@register_function('mutation')
def mutation_baseline(qc_correct, qc_bug, n_qubits, n_errors, i):
    metrics = {}
    n_qubits = qc_correct.num_qubits
    input_states = generate_input_states(n_qubits)
    output_states = apply_circuit(qc_correct, input_states)
    repair_start = perf_counter()
    best_circuit  = genetic_algorithm_optimize(qc_bug, output_states, input_states)
    metrics['total_time'] = perf_counter() - repair_start
    metrics['repaired_circuit']= Circuit(best_circuit).to_json()
    dist = check_circuit(qc_correct, best_circuit)
    metrics['distance'] = 1- dist
    metrics['state'] = 'success' if metrics['distance'] < 0.1 else 'failed'
    save_path = f'results/mutation/{algoname}/qubits_{n_qubits}_errors_{n_errors}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    json.dump(metrics, open(save_path+f"metrics{i}.json","w"))

    
    

import ray
import traceback
@ray.remote
def baseline(baselinefunc: callable, *args, **kwargs):
    try:
        # raise Exception('test')
        return baselinefunc(*args, **kwargs)
    except Exception as e:
        with open(f'error{baselinefunc.name}.log', 'a') as f:
            traceback.print_exc(file=f)
        return None
    
if __name__ == "__main__":
    futures = []
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=str, default='chatgpt')
    parser.add_argument('--datasetdir', type=str, default='all')
    args = parser.parse_args()
    for data in load_dataset(args.datasetdir):
        algoname = data['algorithm']
        n_errors = data['n_errors']
        n_qubits = data['n_qubits']
        qc_corrects, qc_bugs = data['right_circuits'],data['wrong_circuits']
        baselinefunc = baselines[args.baseline]
        [baselinefunc(algoname,qc_correct, qc_bug,n_qubits, n_errors,i)  for i, (qc_correct, qc_bug) in enumerate(zip(qc_corrects, qc_bugs))]
        # futures+= [baseline.remote(baselinefunc, algoname,qc_correct, qc_bug,n_qubits, n_errors,i)  for i, (qc_correct, qc_bug) in enumerate(zip(qc_corrects, qc_bugs))]

    # ray.get(futures)
    
    