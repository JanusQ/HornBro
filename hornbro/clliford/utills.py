import numpy as np
from z3 import *
from typing import List, Tuple
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import random_clifford,Clifford

class StabilizerTable:
    def __init__(self, n_qubits):
        self.n = n_qubits
        self.X = np.zeros((n_qubits, n_qubits), dtype=bool)
        self.Z = np.zeros((n_qubits, n_qubits), dtype=bool)
        self.P = np.zeros(n_qubits, dtype=bool)
        # Initialize the stabilizer table with Z operators
        for i in range(n_qubits):
            self.Z[i][i] = True  # Z on each qubit
        
    def copy(self):
        new = StabilizerTable(self.n)
        new.X = self.X.copy()
        new.Z = self.Z.copy()
        new.P = self.P.copy()
        return new
    
    def apply_cz(self, control, target):
        # clifford.phase ^= x0 & x1 & (z0 ^ z1)
        # z1 ^= x0
        # z0 ^= x1
        # Apply CZ gate from control to target
        for i in range(self.n):
            self.P[i] ^= (self.X[i, control] & self.X[i, target]  & (self.Z[i, control] ^ self.Z[i, target]))
        self.Z[:, target] ^= self.X[:, control]
        self.Z[:, control] ^= self.X[:, target]
    
    def apply_swap(self, qubit1, qubit2):
        self.Z[:, qubit1], self.Z[:, qubit2] = self.Z[:, qubit2].copy(), self.Z[:, qubit1].copy()
        self.X[:, qubit1], self.X[:, qubit2] = self.X[:, qubit2].copy(), self.X[:, qubit1].copy()

    def apply_cnot(self, control, target):
        # Apply CNOT gate from control to target
        for i in range(self.n):
            self.P[i] ^= (self.X[i, control] & self.Z[i, target] & (
                self.X[i, target] ^ self.Z[i, control] ^ True))
        self.X[:, target] ^= self.X[:, control]
        self.Z[:, control] ^= self.Z[:, target]
        
    def apply_hadamard(self, qubit):
        # Swap X and Z for the given qubit
        self.P ^= (self.X[:, qubit] & self.Z[:, qubit])
        self.X[:, qubit], self.Z[:, qubit] = self.Z[:,qubit].copy(), self.X[:, qubit].copy()
        # P \xor X[qubit]*Z[qubit]

    def apply_phase(self, qubit):
        # S gate (phase gate) on a qubit
        self.P ^= (self.X[:, qubit] & self.Z[:, qubit])
        self.Z[:, qubit] ^= self.X[:, qubit]
        
    def apply_sdg(self, qubit):
        # Sdg gate (inverse phase gate) on a qubit
        self.P ^= (self.X[:, qubit] & ~(self.Z[:, qubit]))
        self.Z[:, qubit] ^= self.X[:, qubit]
    
    def apply_x(self, qubit):
        # X gate on a qubit
        self.P ^= self.Z[:, qubit]
    
    def apply_y(self, qubit):
        # Y gate on a qubit
        self.P ^= (self.X[:, qubit] ^ self.Z[:, qubit])
    def apply_sx(self, qubit):
        self.P ^= ~self.X[:, qubit] & self.Z[:, qubit]
        self.X[:, qubit] ^= self.Z[:, qubit]

    def apply_sxdg(self, qubit):
        self.P ^= self.X[:, qubit] & self.Z[:, qubit]
        self.X[:, qubit] ^= self.Z[:, qubit]
        

    def apply_z(self, qubit):
        # Z gate on a qubit
        self.P ^= self.X[:, qubit]
        
    def apply_clifford(self, other):
        """
        Applies another stabilizer table (Clifford operator) to this one.
        """
        if self.n != other.n:
            raise ValueError("Stabilizer tables must have the same number of qubits to be composed.")
        
        composed = StabilizerTable(self.n)
        
        # Compose the symplectic matrices
        composed.X = np.logical_xor(self.X @ other.X, self.Z @ other.Z)
        composed.Z = np.logical_xor(self.X @ other.Z, self.Z @ other.X)
        
        # Update the phase vector
        composed.P = np.logical_xor(self.P, np.logical_xor(other.P, (self.X @ other.P) & (self.Z @ other.P)))
        
        return composed
    @property
    def table(self):
        """
        Generates the full stabilizer table as a 2N x (2N+1) numpy boolean array.
        """
        full_table = np.zeros((2 * self.n, 2 * self.n + 1), dtype=bool)

        # Fill the full table with X and Z parts and the phase P
        full_table[:self.n, :self.n] = self.X  # Fill X part
        full_table[:self.n, self.n:2 * self.n] = self.Z  # Fill Z part
        # Fill P part for the stabilizers
        full_table[:self.n, 2 * self.n] = self.P

        # For destabilizers, the format is usually (I -Z), (X I)
        full_table[self.n:, :self.n] = self.Z  # X -> Z part
        full_table[self.n:, self.n:2 * self.n] = self.X  # Z -> X part
        full_table[self.n:, 2 * self.n] = self.P  # Phase information

        return full_table

    def is_eq(self, other):
        """
        Checks if two stabilizer tables are equal.
        """
        return (self.X == other.X).all() and (self.Z == other.Z).all() and (self.P == other.P).all()

    def to_string(self, label='S'):
        """
        Converts the stabilizer table to a human-readable form including both stabilizers and destabilizers.
        """
        stabilizer_strings = []
        destabilizer_strings = []

        # Generate stabilizers
        for i in range(self.n):
            s = ""
            s += " " + ("+" if not self.P[i] else "-")
            for j in range(self.n):
                if self.X[i, j] and self.Z[i, j]:
                    s += "Y"
                elif self.X[i, j]:
                    s += "X"
                elif self.Z[i, j]:
                    s += "Z"
                else:
                    s += "I"

            stabilizer_strings.append(s)

        # Generate destabilizers
        for i in range(self.n):
            d = ""
            d += " " + ("+" if not self.P[i] else "-")
            for j in range(self.n):
                if self.Z[i, j] and self.X[i, j]:
                    d += "Y"
                elif self.Z[i, j]:
                    d += "X"
                elif self.X[i, j]:
                    d += "Z"
                else:
                    d += "I"

            destabilizer_strings.append(d)

        stabilizers_section = "Stabilizers: " + " ".join(stabilizer_strings)
        destabilizers_section = "Destabilizers: " + \
            " ".join(destabilizer_strings)
        if label == 'S':
            return " ".join(stabilizer_strings)
        elif label == 'D':
            return " ".join(destabilizer_strings)
        else:
            return "\n".join([stabilizers_section, destabilizers_section])

    def to_dict(self):
        """
        Converts the StabilizerTable to a dictionary for JSON serialization.
        """
        return {
            'n': self.n,
            'X': self.X.tolist(),
            'Z': self.Z.tolist(),
            'P': self.P.tolist()
        }
        
        
class CllifordProgram(list):
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits

    def __setitem__(self, index, item):
        super().__setitem__(index, str(item))

    def insert(self, index, item):
        super().insert(index, str(item))

    def copy(self):
        new_program = CllifordProgram(self.n_qubits)
        for item in self:
            new_program.append(item)
        return new_program

    def extend(self, other):
        if isinstance(other, type(self)):
            super().extend(other)
        else:
            raise TypeError(
                "Can only extend CllifordProgram with another type")

    def h(self, qubit):
        self.append(['H', qubit])

    def s(self, qubit):
        self.append(['S', qubit])

    def sdg(self, qubit):
        for _ in range(3):
            self.append(['S', qubit])
    def id(self, qubit):
        self.append(['I', qubit])
    def cz(self, control, target):
        self.append(['H', target])
        self.append(['CNOT', (control, target)])
        self.append(['H', target])

    def x(self, qubit):
        # X = HZH = HSSH
        self.append(['H', qubit])
        self.append(['S', qubit])
        self.append(['S', qubit])
        self.append(['H', qubit])

    def y(self, qubit):
        # Y = XZ
        self.append(['Y', qubit])

    def z(self, qubit):
        # Z = HS = HSH
        for _ in range(2):
            self.append(['S', qubit])

    def cnot(self, control, target):
        self.append(['CNOT', (control, target)])

    def depth(self):
        if self.depth is None:
            self.depth = 0
            ## reorganize the gates into layers
            self.layers = []
            for gate in self:
                layer = []
                qubits = [i for i in range(self.n_qubits)]
                while qubits:
                    if gate[0] == 'CNOT':
                        if all(gate[1][i] not in qubits for i in range(2)):
                            layer.append(gate)
                            qubits.remove(gate[1][0])
                            qubits.remove(gate[1][1])
                        else:
                            qubits = []
                        
                    elif gate[0] == 'CNOT' and gate[1][1] == qubits[0]:
                        layer.append(gate)
                        qubits.remove(gate[1][1])
                        qubits.remove(gate[1][0])
                        
                    else:
                        qubits.remove(qubits[0])
                        
                if gate[0] == 'H':
                    layer.append(gate)
                
                
        return self.depth
    def output_stablizers(self, input_stabilizer_table: StabilizerTable = None):
        """
        This function takes the input stabilizer table and outputs the stabilizers and destabilizers of the Clliford program.
        """
        if input_stabilizer_table is None:
            output_stabilizer_table = StabilizerTable(self.n_qubits)
        else:
            output_stabilizer_table = input_stabilizer_table.copy()
        for gate in self:
            if gate[0] == 'H':
                output_stabilizer_table.apply_hadamard(gate[1])
            elif gate[0] == 'S':
                output_stabilizer_table.apply_phase(gate[1])
            elif gate[0] == 'CNOT':
                output_stabilizer_table.apply_cnot(gate[1][0], gate[1][1])
            elif gate[0] == 'Y':
                output_stabilizer_table.apply_y(gate[1])
            elif gate[0] == 'I':
                pass
            else:
                raise ValueError("Invalid gate type: {}".format(gate[0]))
            # print(output_stabilizer_table.to_string())
        return output_stabilizer_table

    def to_circuit(self):
        """
        This function converts the Clliford program to a circuit.
        """
        circuit = QuantumCircuit(self.n_qubits)
        for gate in self:
            if gate[0] == 'H':
                circuit.h(gate[1])
            elif gate[0] == 'S':
                circuit.s(gate[1])
            elif gate[0] == 'CNOT':
                circuit.cx(gate[1][0], gate[1][1])
            elif gate[0] == 'Y':
                circuit.y(gate[1])
            elif gate[0] == 'I':
                circuit.id(gate[1])
            else:
                raise ValueError("Invalid gate type: {}".format(gate[0]))
        return circuit

    @classmethod
    def from_circuit(cls, circuit: QuantumCircuit, n_qubits: int = None):  # type: ignore
        """
        This function takes a circuit and converts it to a Clliford program.
        """
        if n_qubits is None:
            n_qubits = circuit.num_qubits
        program: CllifordProgram = cls(n_qubits)

        for inst, qargs, cargs in circuit.data:
            qubitindex = circuit.find_bit(qargs[0])[0]
            if inst.name == "h":
                program.h(qubitindex)
            elif inst.name.lower() == "s":
                program.s(qubitindex)
            elif inst.name.lower() == "sdg":
                program.sdg(qubitindex)
            elif inst.name.lower() == "x":
                program.x(qubitindex)
            elif inst.name.lower() == "y":
                program.y(qubitindex)
            elif inst.name.lower() == "z":
                program.z(qubitindex)
            elif inst.name.lower() == "id":
                program.id(qubitindex)
            elif inst.name.lower() == "cz":
                program.cz(qubitindex, circuit.find_bit(qargs[1])[0])
            elif inst.name.lower() == "cx" or inst.name.lower() == "cnot":
                program.cnot(qubitindex, circuit.find_bit(qargs[1])[0])
            elif inst.name.lower() == "swap":
                program.append(['CNOT', (circuit.find_bit(qargs[0])[
                               0], circuit.find_bit(qargs[1])[0])])
                program.append(['CNOT', (circuit.find_bit(qargs[1])[
                               0], circuit.find_bit(qargs[0])[0])])
                program.append(['CNOT', (circuit.find_bit(qargs[0])[
                               0], circuit.find_bit(qargs[1])[0])])
                
            elif inst.name.lower() in ("barrier"):
                pass
            else:
                raise ValueError("Invalid gate type: {}".format(inst.name))
        return program

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
                    q = np.random.choice(qubits)
                    qubits.remove(q)
                    program.h(q)
                elif randp < gate_portion['s']+gate_portion['h']:
                    q = np.random.choice(qubits)
                    qubits.remove(q)
                    program.s(q)

                elif randp < gate_portion['cnot']+gate_portion['s']+gate_portion['h']:
                    control = np.random.choice(qubits)
                    qubits.remove(control)
                    if not qubits:
                        break
                    target = np.random.choice(qubits)
                    program.cnot(control, target)
                    qubits.remove(target)
                else:
                    q = np.random.random.choice(qubits)
                    qubits.remove(q)
            return program


def generate_input_stabilizer_table(n_qubits: int):
    """
    Generates a random Clliford program with the given number of qubits and depth.
    """

    cliff = random_clifford(n_qubits)
    circuit = cliff.to_circuit()
    program = CllifordProgram.from_circuit(circuit)
    out_stab = program.output_stablizers()
    return out_stab


def generate_inout_stabilizer_tables(n_qubits: int, program: CllifordProgram):
    """
    Generates the input and output stabilizer tables for the given Clliford program.
    """
    input_stabilizer_table = generate_input_stabilizer_table(n_qubits)
    # print('output stabilizer table producing')
    output_stabilizer_table = program.output_stablizers(input_stabilizer_table)
    return input_stabilizer_table, output_stabilizer_table


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


if __name__ == '__main__':
    table1 = StabilizerTable(2)
    table1.apply_hadamard(1)
    table1.apply_hadamard(0)
    table1.apply_cnot(0, 1)

    table2 = StabilizerTable(2)
    table2.apply_phase(0)
    table2.apply_cnot(1, 0)
    
    composed_table = table1.apply_clifford(table2)

    # Print the resulting stabilizer table
    print(composed_table.to_string())
    
    table1.apply_phase(0)
    table1.apply_cnot(1, 0)
    print(table1.to_string())
    