## Introduction
hornbro is a tool for automated program repair using Homotopy-like Method for Automated Quantum Program Repair.
## Setup
- Runtime environment: Linux ubuntu or MacOS.
- Hardware requirements: CPU with at least 8 cores and 32GB memory. 1TB disk space is recommended.
- Software requirements: Python 3.10, pip, conda, git.
- Network requirements: Internet connection is required to download the required packages.
## Installation
hornbro can be installed using conda and pip:
create a virtual environment with conda
```bash
conda create -n hornbroenv python==3.10
```
```bash
conda activate hornbroenv
```

install hornbro locally using pip

```bash
pip install -e .
```

## Dataset for quantum program repair
Hornbro generate a dataset for quantum program repair by randomly add buggy code into the original quantum programs, which is the third crafted dataset in the paper. The dataset can be generated by the following command:
```bash
gendataset --qubits 5 10 15 --errors 1 2 4 8 --algorithms ghz grover --cases 50 --dirname dataset
```
Here is the meaning of each parameter, which can be also found by running `gendataset --help`:
- `--qubits`: the number of qubits in the quantum program.
- `--errors`: the number of errors to add to the original program.
- `--algorithms`: the quantum algorithms to use for the program repair. the supported algorithms are [ghz,graphstate,qaoa,qft,qnn,qpeexact,grover,qwalk-noancilla,vqe,wstate,shor,groundstate,routing,tsp].
- `--cases`: the number of programs to generate for each combination of qubits and errors.
- `--dirname`: the directory to save the generated dataset.

The generated dataset will be saved in the directory `dataset` with the following structure:
```
dataset/
├── grover
│   ├── 10
│   │   ├── 2
│   │   │   └── qcs.pkl
│   │   ├── 4
│   │   │   └── qcs.pkl
│   │   └── 6
│   │       └── qcs.pkl
│   ├── 15
│   │   ├── 2
│   │   │   └── qcs.pkl
│   │   ├── 4
│   │   │   └── qcs.pkl
│   │   └── 6
│   │       └── qcs.pkl
```
For a path with `dataset/grover/10/2/qcs.pkl`, the program is a grover algorithm with 10 qubits and 2 errors. The programs with qiskit QuantumCircuit objects are saved in a pickle file named `qcs.pkl`.
The first half of `qcs.pkl` contains the original programs, and the second half contains the buggy programs.
In python script, we can load the dataset using the following code:
```python
from hornbro.dataset import load_dataset
for data in load_dataset(dirname):
    algoname = data['algorithm']
    n_qubits = data['n_qubits']
    n_errors = data['n_errors']
    qc_corrects, qc_bugs = data['right_circuits'],data['wrong_circuits']
    for qc_correct, qc_buggy in zip(qc_corrects, qc_bugs):
        # do something with the program pair
        pass
```

## Evaluation
The experimental evaluation of hornbro is done using the following command:
### RQ1 \& RQ2: Effectiveness and efficiency of HornBro
Run the following command to evaluate the effectiveness of HornBro on the generated dataset:
```bash
python evaluation/efficiencyHornBro.py --datasetdir dataset
```
This command will evaluate the efficiency of HornBro on the generated dataset in `dataset` directory. The results will be saved in the directory `results/`.

For baseline methods, we use the following methods:
- chatgpt: the LLM-QAPR method using the ChatbotGPT model. To use it, you should first set the environment variable `OPENAI_API_KEY` to your ChatbotGPT API key.
```bash
export OPENAI_API_KEY="<your_api_key>"
```
- kimi: the LLM-QAPR method using the KIMI model. Set the environment variable `MOONSHOT_API_KEY` to your KIMI API key.
```bash
export MOONSHOT_API_KEY="<your_api_key>"
```
- qsd: the Syn-QSD method in the paper.
- qfast: the QFAST method in the paper.
- mutation: the mutation-based method in the paper.
To evaluate them, run the following command:
```bash
python evaluation/efficiencybaselines.py --datasetdir dataset --baseline chatgpt
```
the `--baseline` parameter can be replaced with `kimi`, `qsd`, `qfast`, or `mutation` to evaluate the corresponding baseline method.

The results will be saved in the directory `results/{baseline}/`.

### RQ3: Effects of Techniques in Each Stages
For the ablation study of the SMT-based localization, we compares the performance of HornBro with a local search method. The local search method can be run using the following command:
```bash
python evaluation/localsearch.py --datasetdir dataset
```
The results will be saved in the directory `results/localsearch/`.

### RQ4: Effects of Configurable Hyper-parameters
To evaluate the effects of configurable hyper-parameters, we use the following command:
```bash
hornbrorapair --datasetdir dataset --n_samples 20 --max_epochs 1000 --n_layers 3 --lr 0.1 --batch_size 100
```
the parameters discriptions can be found by running 
```bash
hornbrorapair -h
```
which includes:
- `--datasetdir`: the directory of the dataset.
- `--n_samples`: the number of samples to use for the SMT model.
- `--max_epochs`: the maximum number of epochs to train the gradient repair model.
- `--n_layers`: the number of anxiliary layers in the SMT model.
- `--lr`: the learning rate of the gradient repair model.
- `--batch_size`: the batch size of the gradient repair model.

The results will be saved in the directory `results/hornbro_inputs_{n_samples}_layers_{n_layers}_lr_{lr}_max_epochs_{max_epochs}_batch_size_{batch_size}/`.
