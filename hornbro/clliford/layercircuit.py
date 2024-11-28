import pennylane as qml

from qiskit import QuantumCircuit
import copy

from qiskit.circuit import ClassicalRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.dagcircuit.dagnode import DAGNode,DAGOpNode
from math import pi
from collections.abc import Iterable
import numpy as np
from .utills import StabilizerTable
import random

def layer_to_dag(circuit: QuantumCircuit):
    instructions = []
    dagnodes = []

    dagcircuit = DAGCircuit()
    dagcircuit.name = circuit.name
    dagcircuit.global_phase = circuit.global_phase
    dagcircuit.calibrations = circuit.calibrations
    dagcircuit.metadata = circuit.metadata

    dagcircuit.add_qubits(circuit.qubits)
    dagcircuit.add_clbits(circuit.clbits)

    for register in circuit.qregs:
        dagcircuit.add_qreg(register)

    for register in circuit.cregs:
        dagcircuit.add_creg(register)

    # for instruction, qargs, cargs in circuit.data:
    for instruction in circuit.data:
        operation = instruction.operation

        dag_node = dagcircuit.apply_operation_back(
            operation, instruction.qubits, instruction.clbits
        )
        if operation.name == 'barrier':
            continue
        instructions.append(instruction)  # TODO: 这个里面会不会有barrier
        dagnodes.append(dag_node)
        operation= operation.to_mutable()   
        operation._index = len(dagnodes) - 1
        assert instruction.operation.name == dag_node.op.name

    dagcircuit.duration = circuit.duration
    dagcircuit.unit = circuit.unit
    return dagcircuit, instructions, dagnodes

def get_layered_instructions(circuit):
    dagcircuit, instructions, nodes = layer_to_dag(circuit)
    # dagcircuit.draw(filename = 'dag.svg')
    graph_layers = dagcircuit.multigraph_layers()

    layer2operations = []  # Remove input and output nodes
    for layer in graph_layers:
        layer = [node for node in layer if isinstance(
            node, DAGOpNode) and node.op.name not in ('barrier', 'measure')]
        if len(layer) != 0:
            layer2operations.append(layer)

    layer2instructions = []
    instruction2layer = [None] * len(nodes)
    for layer, operations in enumerate(layer2operations):  # 层号，该层操作
        layer_instructions = []
        for node in operations:  # 该层的一个操作
            assert node.op.name != 'barrier'
            # print(node.op.name)
            index = nodes.index(node)
            layer_instructions.append(instructions[index])
            instruction2layer[index] = layer  # instruction在第几层
        layer2instructions.append(layer_instructions)

    return layer2instructions  #, instruction2layer, instructions, dagcircuit, nodes


def qiskit_to_layer_gate(instruction):
    name = instruction.operation.name
    parms = list(instruction.operation.params)
    return {
        'name': name,
        'qubits': [qubit._index for qubit in instruction.qubits],
        'params': [ele if isinstance(ele, float) else float(ele) for ele in parms],
    }


class LayerCllifordProgram(list):
    def __init__(self, n_qubits: int,data: Iterable[Iterable[dict]] = None):
        super().__init__()
        self.n_qubits = n_qubits
        if data is not None:
            for layer in data:
                self.append(layer)

    def __setitem__(self, index, item):
        super().__setitem__(index, str(item))

    def __getitem__(self, index):
        cls = type(self)
        if isinstance(index, slice):
            return cls(self.n_qubits, super().__getitem__(index))
        else:
            return super().__getitem__(index)
    
    def insert(self, index, item):
        super().insert(index, item)

    def copy(self):
        new_program = LayerCllifordProgram(self.n_qubits)
        from copy import deepcopy
        for item in self:
            new_program.append(deepcopy(item))
        return new_program

    def extend(self, other):
        if isinstance(other, type(self)):
            super().extend(other)
        else:
            raise TypeError(
                "Can only extend CllifordProgram with another type")

    def append_single_qubit_gate(self, gate_name: str, qubit: int):
        last_layer = self[-1] if self else None
        from functools import reduce
        if last_layer is not None:
            last_layer_qubits = reduce(lambda x, y: x + y['qubits'],last_layer,[])
            if qubit in last_layer_qubits:
                self.append([{'name': gate_name, 'qubits': [qubit]}])
            else:
                self[-1].append({'name': gate_name, 'qubits': [qubit]})
        else:
            self.append([{'name': gate_name, 'qubits': [qubit]}])
        
    def append_two_qubit_gate(self, gate_name: str, qubits: tuple):
        last_layer = self[-1] if self else None
        from functools import reduce
        if last_layer is not None:
            last_layer_qubits = reduce(lambda x, y: x + y['qubits'],last_layer,[])
            if all(q in last_layer_qubits for q in qubits):
                self.append([{'name': gate_name, 'qubits': list(qubits)}])
            else:
                self[-1].append({'name': gate_name, 'qubits': list(qubits)})
        else:
            self.append([{'name': gate_name, 'qubits': list(qubits)}])
        
    def h(self, qubit):
        self.append_single_qubit_gate('h', qubit)

    def s(self, qubit):
        self.append_single_qubit_gate('s', qubit)

    def sdg(self, qubit):
        for _ in range(3):
            self.append_single_qubit_gate('s', qubit)
        
    def id(self, qubit):
        self.append_single_qubit_gate('id', qubit)

    def x(self, qubit):
        # X = HZH = HSSH
        self.append_single_qubit_gate('x', qubit)

    def y(self, qubit):
        self.append_single_qubit_gate('y', qubit)

    def z(self, qubit):
        self.append_single_qubit_gate('z', qubit)

    def cnot(self, control, target):
        self.cx(control, target)
    def cx(self, control, target):
        self.append_two_qubit_gate('cx', (control, target))
    def cz(self, control, target):
        self.append_two_qubit_gate('cz', (control, target))
    
    def depth(self):
        """
        This function calculates the depth of the Clliford program.
        """
        return len(self)
                
                
    def output_stablizers(self, input_stabilizer_table: StabilizerTable = None):
        """
        This function takes the input stabilizer table and outputs the stabilizers and destabilizers of the Clliford program.
        """
        if input_stabilizer_table is None:
            output_stabilizer_table = StabilizerTable(self.n_qubits)
        else:
            output_stabilizer_table = input_stabilizer_table.copy()
        for layer in self:
            for gate in layer:
                if gate['name'] == 'h':
                    output_stabilizer_table.apply_hadamard(gate['qubits'][0])
                elif gate['name'] == 's':
                    output_stabilizer_table.apply_phase(gate['qubits'][0])
                elif gate['name'] == 'cnot' or gate['name'] == 'cx':
                    output_stabilizer_table.apply_cnot(gate['qubits'][0], gate['qubits'][1])
                elif gate['name'] == 'y':
                    output_stabilizer_table.apply_y(gate['qubits'][0])
                elif gate['name'] == 'swap':
                    output_stabilizer_table.apply_swap(gate['qubits'][0], gate['qubits'][1])
                elif gate['name'] == 'z':
                    output_stabilizer_table.apply_z(gate['qubits'][0])
                elif gate['name'] == 'x':
                    output_stabilizer_table.apply_x(gate['qubits'][0])
                elif gate['name'] == 'cz':
                    output_stabilizer_table.apply_cz(gate['qubits'][0], gate['qubits'][1])
                elif gate['name'] == 'sdg':
                    output_stabilizer_table.apply_sdg(gate['qubits'][0])
                elif gate['name'] == 'id':
                    pass
                else:
                    raise ValueError("Invalid gate type: {}".format(gate['name']))
                # print(output_stabilizer_table.to_string())
        return output_stabilizer_table

    @staticmethod
    def from_qiskit_circuit(qiskit_circuit: QuantumCircuit):
        layer2qiskit_instructions = get_layered_instructions(qiskit_circuit)
        layer_circuit = []
        for layer_index, layer_instructions in enumerate(layer2qiskit_instructions):
            layer_instructions = [qiskit_to_layer_gate(
                instruction) for instruction in layer_instructions]
            layer_circuit.append(layer_instructions)
        return LayerCllifordProgram(qiskit_circuit.num_qubits, layer_circuit)
    
    def to_qml_circuit(self):
        """ convert the layer_circuit to qml circuit

        Args:
            layer_cirucit (Iterable[Iterable[dict]]):  the layer_circuit to be converted
            Define the structure of the layer_circuit:
                1. layer_circuit is a list of layers
                2. each layer is a list of gates
                3. each gate is a dict with the following keys:
                    'name': the name of the gate
                    'qubits': the qubits the gate acts on
                    'params': the parameters of the gate (if any)
                    there are some notes for the gates:
                    (1) If the gate is single qubits gate without params,like H,X,Y,Z, Measure, Tracepoint.
                    the qubits is a list of qubit index, e.g. [0,1],when the gate acts on qubit 0 and 1.
                    (2) If the gate is single qubits gate with params, the qubits is a list of qubit index, e.g. [0,1],when the gate acts on qubit 0 and 1. and if the params is a float, the qubits will be apply the gate with same param. if the params is a list, the qubits will be apply the gate with respective param.
                    (3) If the gate is multi qubits gate, the qubits is a list of qubit index, e.g. [0,1], which means the gate acts on qubit 0 and 1, and the control qubit is the last qubit in the list.

        Returns:
            qml.QNode: the qml circuit
        Raises:
            Exception: Unkown gate type
        """
        gate_map = {
            'h': qml.Hadamard,
            'x': qml.PauliX,
            'y': qml.PauliY,
            'z': qml.PauliZ,
            's': qml.S,
            't': qml.T,
            'u3': qml.U3,
            'u2': qml.U2,
            'u1': qml.U1,
            'rx': qml.RX,
            'ry': qml.RY,
            'rz': qml.RZ,
            'cx': qml.CNOT,
            'cz': qml.CZ
        }
        for layer in self:
            for gate in layer:
                gate_qubits = gate['qubits']
                if gate['name'] in ['h', 'x', 'y', 'z', 's', 't']:
                    if isinstance(gate_qubits, Iterable):
                        for q in gate_qubits:
                            gate_map[gate['name'].lower()](wires=q)
                    else:
                        gate_map[gate['name'].lower()](wires=gate_qubits)
                elif gate['name'] == 'measure':
                    qml.sample(qml.PauliZ(wires=gate_qubits))
                elif gate['name'] == 'sdg':
                    if isinstance(gate_qubits, Iterable):
                        for q in gate_qubits:
                            qml.PhaseShift(-pi/2, wires=q)
                    else:
                        qml.PhaseShift(-pi/2, wires=gate_qubits)
                
                elif gate['name'] == 'u3' or gate['name'] == 'u':
                    theta, phi, lam = gate['params']  # TODO: 这个地方的参数位置需要检查一下
                    qml.U3(theta, phi, lam, wires=gate_qubits)
                elif gate['name'] == 'u2':
                    phi, lam = gate['params']
                    qml.U2(phi, lam, wires=gate_qubits)
                elif gate['name'] == 'u1':
                    phi = gate['params'][0]
                    qml.U1(phi, wires=gate_qubits)
                elif gate['name']== 'flipkey':
                    key = np.array([int(i) for i in list(gate['params'])])
                    qml.ctrl(qml.FlipSign(key, wires=gate['ctrled_qubits']),control = gate['ctrl_qubits'][0])
                elif gate['name'] in ['rx', 'ry', 'rz']:
                    if isinstance(gate['params'],Iterable):
                        if isinstance(gate_qubits, Iterable):
                            for i,q in enumerate(gate_qubits):
                                gate_map[gate['name'].lower()](gate['params'][i],wires=q)
                        else:
                            gate_map[gate['name'].lower()](gate['params'][0],wires=gate_qubits)
                    else:
                        if isinstance(gate_qubits, Iterable):
                            for q in gate_qubits:
                                gate_map[gate['name'].lower()](gate['params'],wires=q)
                        else:
                            gate_map[gate['name'].lower()](gate['params'],wires=gate_qubits)

                elif gate['name'] == 'cx':
                    qml.CNOT(wires=gate_qubits)
                elif gate['name'] == 'cz':
                    qml.CZ(wires=gate_qubits)
                elif gate['name'] == 'cond_x':
                    m_0 = qml.measure(gate_qubits[0])
                    qml.cond(m_0, qml.PauliX)(wires=gate_qubits[1])
                elif gate['name'] == 'cond_z':
                    m_0 = qml.measure(gate_qubits[0])
                    qml.cond(m_0, qml.PauliZ)(wires=gate_qubits[1])
                elif gate['name'] == 'mcz':
                    qml.MultiControlledZ(wires=gate_qubits[:-1], control_wires=gate_qubits[-1])
                elif gate['name'] == 'initialize':
                    qml.QubitStateVector(gate['params'], wires=gate_qubits)
                elif gate['name'] == 'unitary':
                    unitary = gate['params']
                    qml.QubitUnitary(unitary, wires=gate_qubits)
                elif gate['name'] == 'wirecut' or gate['name'] == 'tracepoint':
                    pass
                elif gate['name'] == 'ctrl':
                    operation = gate['params']
                    operation = qml.QubitUnitary(operation, wires=gate['ctrled_qubits'])
                    qml.ctrl(operation, control=gate['ctrl_qubits'])
                elif gate['name'] == 'swap':
                    qml.SWAP(wires=gate_qubits)
                elif gate['name'] == 'channel':
                    channel = gate['params']
                    qml.QubitChannel(channel, wires=gate_qubits)
                else:
                    raise Exception('Unkown gate type', gate)

        
    def to_qiskit_circuit(self):               
        qiskit_circuit = QuantumCircuit(self.n_qubits)
        for layer in self:
            for gate in layer:
                n_qubits = gate['qubits']
                if gate['name'] == 'u3' or gate['name'] == 'u':
                    theta, phi, lam = gate['params']  # TODO: 这个地方的参数位置需要检查一下
                    qiskit_circuit.u(theta, phi, lam, qubit=n_qubits)
                elif gate['name'] == 'u2':
                    phi, lam = gate['params']
                    qiskit_circuit.u(pi/2,phi, lam, qubit=n_qubits)
                elif gate['name'] == 'u1':
                    lam = gate['params'][0]
                    qiskit_circuit.p(lam, qubit=n_qubits)
                elif gate['name'] == 'h':
                    qiskit_circuit.h(qubit=n_qubits)
                elif gate['name'] == 'x':
                    qiskit_circuit.x(qubit=n_qubits)
                elif gate['name'] == 'y':
                    qiskit_circuit.y(qubit=n_qubits)
                elif gate['name'] == 'z':
                    qiskit_circuit.z(qubit=n_qubits)
                elif gate['name'] == 'sdg':
                    qiskit_circuit.sdg(qubit=n_qubits)
                elif gate['name'] == 's':
                    qiskit_circuit.s(qubit=n_qubits)
                elif gate['name'] == 'swap':
                    qiskit_circuit.swap(qubit1=n_qubits[0], qubit2=n_qubits[1])
                elif gate['name'] == 'rz':
                    qiskit_circuit.rz(gate['params'][0], qubit=n_qubits)
                elif gate['name'] == 'cond_x':
                    classcial_register = ClassicalRegister(1)
                    qiskit_circuit.add_register(classcial_register)
                    qiskit_circuit.measure(n_qubits[0], classcial_register)
                    qiskit_circuit.x(qubit=n_qubits[1]).c_if(classcial_register, 1)
                elif gate['name'] == 'cond_z':
                    classcial_register = ClassicalRegister(1)
                    qiskit_circuit.add_register(classcial_register)
                    qiskit_circuit.measure(n_qubits[0], classcial_register)
                    qiskit_circuit.z(qubit=n_qubits[1]).c_if(classcial_register, 1)
                elif gate['name'] == 'cx' or gate['name'] == 'cnot':
                    qiskit_circuit.cx(control_qubit=n_qubits[0], target_qubit=n_qubits[1])
                elif gate['name'] == 'cz':
                    qiskit_circuit.cz(control_qubit=n_qubits[0], target_qubit=n_qubits[1])
                elif gate['name'] == 'mcz':
                    qiskit_circuit.mcp(gate['params'][0], control_qubits=n_qubits[:-1], target_qubit=n_qubits[-1])
                elif gate['name'] == 'ctrl':
                    unitary = gate['params']
                    operation_circuit = QuantumCircuit(len(gate['ctrled_qubits']))
                    operation_circuit.unitary(unitary, qubits=list(range(len(gate['ctrled_qubits']))))
                    custom_control = operation_circuit.to_gate().control(len(gate['ctrl_qubits']))
                    qiskit_circuit.append(custom_control, gate['ctrl_qubits'] + gate['ctrled_qubits'])
                elif gate['name'] == 'initialize':
                    qiskit_circuit.initialize(gate['params'],qubits= n_qubits)
                elif gate['name'] == 'unitary':
                    unitary = gate['params']
                    qiskit_circuit.unitary(unitary,qubits= n_qubits)
                elif gate['name'] == 'channel':
                    channel = gate['params']
                    qiskit_circuit.channel(channel, qubits= n_qubits)
                elif gate['name'] == 'measure':
                    classcial_register = ClassicalRegister(len(n_qubits))
                    qiskit_circuit.add_register(classcial_register)
                    qiskit_circuit.measure(n_qubits, classcial_register)
                elif gate['name'] == 'cswap':
                    qiskit_circuit.cswap(n_qubits[0], n_qubits[1], n_qubits[2])
                elif gate['name'] == 'id':
                    qiskit_circuit.id(qubit=n_qubits)
                elif gate['name'] == 'tracepoint':
                    pass
                else:
                    raise Exception('Unkown gate type', gate)
        return qiskit_circuit


    def to_list(self):
        """ convert the layer_circuit to a list of gates
        Returns:
            list: the list of gates
        """
        layer_list = []
        for layer in self:
            layer_list.extend(layer)
        return layer_list



    @classmethod
    def random_clifford_program(cls, n_qubits: int, depth: int, gate_portion: dict = {'h': 0.2, 's': 0.2, 'cnot': 0.6}):
        """
        This function generates a random Clliford program with the given depth.
        """
        program = cls(n_qubits)
        for _ in range(depth):
            qubits = [i for i in range(program.n_qubits)]
            while qubits:
                randp = np.random.rand()
                if randp < gate_portion['h']:
                    q = random.choice(qubits)
                    qubits.remove(q)
                    program.h(q)
                elif randp < gate_portion['s']+gate_portion['h']:
                    q = random.choice(qubits)
                    qubits.remove(q)
                    program.s(q)

                elif randp < gate_portion['cnot']+gate_portion['s']+gate_portion['h']:
                    control = random.choice(qubits)
                    qubits.remove(control)
                    if not qubits:
                        break
                    target = random.choice(qubits)
                    program.cnot(control, target)
                    qubits.remove(target)
                else:
                    q = random.choice(qubits)
                    qubits.remove(q)
        return program

    def generate_bugged_program(self, n_errors: int):
        bugged_program = self.copy()
        import random
        # n_gates = len(correct_circuit.data)
        # n_errors = int(n_gates * error_rate)
        gates = ['h', 'x', 'y', 'z', 'cx', 'cz']  # CZGate(),
        single_qubit_gates = ['h', 'x', 'y', 'z', 's']
        two_qubit_gates = ['cx', 'cz']  # , CZGate()
        # param_gates = [RXGate(Parameter('θ')), RYGate(Parameter('θ')), RZGate(Parameter('θ'))]
        for _ in range(n_errors):
            error_type = random.choice(['add', 'delete', 'replace'])
            qubits = list(range(self.n_qubits))
            if error_type == 'add':
                gate = random.choice(gates)
                qargs = [random.choice(qubits)]
                if gate in two_qubit_gates:
                    qargs.append(random.choice([q for q in qubits if q != qargs[0]]))
                bugged_program[random.choice(range(len(bugged_program)))].append({'name': gate, 'qubits': qargs, 'params': []})
            elif error_type == 'delete' and len(bugged_program) > 0:
                index = random.randint(0, len(bugged_program) - 1)
                if len(bugged_program[index]) > 0:
                    bugged_program[index].pop(random.randint(0, len(bugged_program[index]) - 1))
            elif error_type == 'replace' and len(bugged_program) > 0:
                layerindex = random.randint(0, len(bugged_program) - 1)
                gateindex = random.randint(0, len(bugged_program[layerindex]) - 1)
                qargs = bugged_program[layerindex][gateindex]['qubits']
                if len(qargs) == 1:
                    gate = random.choice(single_qubit_gates)
                else:
                    gate = random.choice(two_qubit_gates)
                bugged_program[layerindex][gateindex]['name'] = gate
        return bugged_program


