
from z3 import *
from .utills import StabilizerTable, CllifordProgram


class CllifordCorrecter:
    def __init__(self, n_qubits, d_max: int = 4):
        self.n = n_qubits
        self.d_max = d_max
        self.Svars, self.Hvars, self.Ivars, self.CNOTvars = self.define_variables()
        self.constraints = []
        self.iout_idx = 0

    def define_variables(self):
        """
        Defines the variables used in the Clliford solver.
        """
        Svars = []
        for q in range(self.n):
            Svars.append([Bool("S_{}_{}".format(q, d))
                         for d in range(self.d_max)])
        Hvars = []
        for q in range(self.n):
            Hvars.append([Bool("H_{}_{}".format(q, d))
                         for d in range(self.d_max)])
        Ivars = []
        for q in range(self.n):
            Ivars.append([Bool("I_{}_{}".format(q, d))
                         for d in range(self.d_max)])
        CNOTvars = {}
        for q in range(self.n):
            CNOTvars[q] = {}
            for t in range(self.n):
                if q != t:
                    CNOTvars[q][t] = [Bool("CNOT_{}_{}_{}".format(
                        q, t, d)) for d in range(self.d_max)]

        return Svars, Hvars, Ivars, CNOTvars

    def define_tables(self):
        """
        Defines the tables used in the Clliford solver.
        """
        X = []
        Z = []
        P = []
        for i in range(self.n):
            X.append([Bool("X_{}_{}_{}".format(i, q, self.iout_idx))
                     for q in range(self.n)])
            Z.append([Bool("Z_{}_{}_{}".format(i, q, self.iout_idx))
                     for q in range(self.n)])
            P.append(Bool("P_{}_{}".format(i, self.iout_idx)))
        self.X, self.Z, self.P = X, Z, P
        self.iniX, self.iniZ, self.iniP = [], [], []
        self.applyed_gates = False
        for i in range(self.n):
            self.iniX.append([])
            for q in range(self.n):
                self.iniX[i].append(self.X[i][q])
            self.iniZ.append([])
            for q in range(self.n):
                self.iniZ[i].append(self.Z[i][q])
            self.iniP.append(self.P[i])

    def apply_hadamard(self, qubit, Hvar):
        # Swap X and Z for the given qubit
        for i in range(self.n):
            temp = self.X[i][qubit] & self.Z[i][qubit]
            self.P[i] = If(Hvar, self.P[i] ^ temp, self.P[i])
            tempX = self.X[i][qubit]
            self.X[i][qubit] = If(Hvar, self.Z[i][qubit], self.X[i][qubit])
            self.Z[i][qubit] = If(Hvar, tempX, self.Z[i][qubit])

    def apply_identity(self, qubit, Ivar, d):
        # Apply identity gate on a qubit
        pass
        # for i in range(self.n):
        #     # self.constraints.append(Implies(Ivar, self.Ps[d+1][i] == self.Ps[d][i]))
        #     self.constraints.append(Implies(Ivar, self.Xs[d+1][i][qubit] == self.Xs[d][i][qubit]))
        #     self.constraints.append(Implies(Ivar, self.Zs[d+1][i][qubit] == self.Zs[d][i][qubit]))

    def apply_cnot(self, control, target, CNOTvar):
        # Apply CNOT gate from control to target
        for i in range(self.n):
            temp = ((self.X[i][target] ^ True) ^ self.Z[i]
                    [control]) & self.X[i][control] & self.Z[i][target]
            # self.P[i] = Xor(self.P[i], temp)
            self.P[i] = If(CNOTvar, self.P[i] ^ temp, self.P[i])
            # self.X[i][target] =  Xor(self.X[i][target], self.X[i][control] & CNOTvar)
            self.X[i][target] = If(
                CNOTvar, self.X[i][target] ^ self.X[i][control], self.X[i][target])
            # self.Z[i][control] =  Xor(self.Z[i][control], self.Z[i][target] & CNOTvar)
            self.Z[i][control] = If(
                CNOTvar, self.Z[i][control] ^ self.Z[i][target], self.Z[i][control])

    def apply_phase(self, qubit, PHASEvar):
        # S gate (phase gate) on a qubit
        for i in range(self.n):
            self.P[i] = Xor(self.P[i], self.X[i][qubit] &
                            self.Z[i][qubit] & PHASEvar)
            self.Z[i][qubit] = Xor(
                self.Z[i][qubit], self.X[i][qubit] & PHASEvar)

    def apply_gates(self):
        """
        Applies  gates 
        """
        self.applyed_gates = True
        print('Applying gates... can only used once')
        for d in range(self.d_max):
            for q in range(self.n):
                self.apply_hadamard(q, self.Hvars[q][d])
                self.apply_phase(q, self.Svars[q][d])
                for t in range(self.n):
                    if q != t:
                        self.apply_cnot(q, t, self.CNOTvars[q][t][d])
                        # self.apply_cnot(t, q,self.CNOTvars[t][q][d])

    def unique_gate(self):
        """
        one depth and one qubit at a time
        """
        for d in range(self.d_max):
            for q in range(self.n):
                self.constraints.append(Sum([self.Ivars[q][d], self.Hvars[q][d], self.Svars[q][d], *[self.CNOTvars[q][t][d]
                                        for t in range(self.n) if q != t], *[self.CNOTvars[t][q][d] for t in range(self.n) if q != t]]) == 1)

    def inverse_cancel(self):
        for d in range(self.d_max-1):
            for q in range(self.n):
                self.constraints.append(
                    Implies(self.Hvars[q][d], Not(self.Hvars[q][d+1])))
                for t in range(self.n):
                    if q != t:
                        self.constraints.append(
                            Implies(self.CNOTvars[q][t][d], Not(self.CNOTvars[q][t][d+1])))

    def set_in(self, stabilizer_table: StabilizerTable):
        """
        Returns the expression for the equality of the stabilizer table at depth d.
        """
        self.X, self.Z, self.P = [], [], []
        for i in range(self.n):
            self.X.append([BoolVal(True) if stabilizer_table.X[i]
                          [j] else BoolVal(False) for j in range(self.n)])
            self.Z.append([BoolVal(True) if stabilizer_table.Z[i]
                          [j] else BoolVal(False) for j in range(self.n)])
            self.P.append(
                BoolVal(True) if stabilizer_table.P[i] else BoolVal(False))

    def add_iout(self, input_stabilizer_table: StabilizerTable, output_stabilizer_table: StabilizerTable):
        """
        Adds the input and output stabilizer tables to the program.
        """
        self.set_in(input_stabilizer_table)
        self.apply_gates()
        self.set_out(output_stabilizer_table)

    def set_out(self, stabilizer_table: StabilizerTable):
        """
        Returns the expression for the equality of the stabilizer table at depth d.
        """
        for i in range(self.n):
            for j in range(self.n):
                self.constraints.append(self.X[i][j] == BoolVal(
                    bool(stabilizer_table.X[i][j])))
                self.constraints.append(self.Z[i][j] == BoolVal(
                    bool(stabilizer_table.Z[i][j])))
            self.constraints.append(
                self.P[i] == BoolVal(bool(stabilizer_table.P[i])))

    def solve(self, timeout=10000):
        """
        Solves the Clliford solver.
        """
        self.unique_gate()
        self.inverse_cancel()
        s = Solver()
        for c in self.constraints:
            s.add(c)

        s.set("timeout", timeout)
        print("Solving...")
        print(s.check())
        fix_program = CllifordProgram(self.n)
        # print(s.model())
        if s.check() == unknown:
            print(s.reason_unknown())
        elif s.check() == sat:
            m = s.model()
            for d in range(self.d_max):
                for q in range(self.n):
                    repeat = 0
                    if m[self.Hvars[q][d]]:
                        fix_program.h(q)
                        repeat += 1
                    if m[self.Svars[q][d]]:
                        fix_program.s(q)
                        repeat += 1
                    for t in range(self.n):
                        if q != t:
                            if m[self.CNOTvars[q][t][d]]:
                                repeat += 1
                                fix_program.cnot(q, t)
                    if repeat > 1:
                        raise ValueError(
                            "Multiple gates applied to the same qubit at depth {}".format(d))
            return fix_program

    def inference(self, input_stabilizer_table: StabilizerTable, program):
        self.set_in(input_stabilizer_table)
        for layeridx, gate in enumerate(program):
            # print(gate)
            if gate[0] == 'H':
                self.constraints.append(self.Hvars[gate[1]][layeridx] == True)
            elif gate[0] == 'S':
                self.constraints.append(self.Svars[gate[1]][layeridx] == True)
            elif gate[0] == 'CNOT':
                self.constraints.append(
                    self.CNOTvars[gate[1][0]][gate[1][1]][layeridx] == True)
            if isinstance(gate[1], int):
                self.constraints.append(
                    self.Ivars[1-gate[1]][layeridx] == True)
        self.unique_gate()
        self.apply_gates()
        self.inverse_cancel()
        s = Solver()
        for c in self.constraints:
            s.add(c)
        # s.set("timeout", 10000)

        fix_program = CllifordProgram(self.n)
        print("Solving...")
        print(s.check())
        if s.check() == unknown:
            print(s.reason_unknown())
        if s.check() == sat:
            m = s.model()
            result_tab = StabilizerTable(self.n)
            for q in range(self.n):
                for d in range(self.d_max):
                    if m[self.Hvars[q][d]]:
                        fix_program.h(q)
                    if m[self.Svars[q][d]]:
                        fix_program.s(q)
                    for t in range(self.n):
                        if q != t:
                            if m[self.CNOTvars[q][t][d]]:
                                fix_program.cnot(q, t)
            print(fix_program.to_circuit())
            for i in range(self.n):
                for q in range(self.n):
                    # print(m.evaluate(self.X[i][q]),m.evaluate(self.Z[i][q]),m.evaluate(self.P[i]))
                    result_tab.X[i][q] = m.evaluate(self.X[i][q])
                    result_tab.Z[i][q] = m.evaluate(self.Z[i][q])
                result_tab.P[i] = m.evaluate(self.P[i])

            return result_tab
        else:
            return None


if __name__ == "__main__":
    from clliford import generate_inout_stabilizer_tables
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Clifford, random_clifford
    n_qubits = 5
    circuit = random_clifford(n_qubits).to_circuit()
    circuit.data = circuit.data[:10]
    program = CllifordProgram.from_circuit(circuit)
    # program = program[:10]
    print('test program\n', program.to_circuit())
    correcter = CllifordCorrecter(n_qubits, len(program))
    inputs, outputs = [], []
    for _ in range(1):
        input_stabilizer_table, output_stabilizer_table = generate_inout_stabilizer_tables(
            n_qubits, program)
        inputs.append(input_stabilizer_table)
        outputs.append(output_stabilizer_table)
        correcter.add_iout(input_stabilizer_table, output_stabilizer_table)
    find_program = correcter.solve()
    print('find program\n', find_program.to_circuit())
    for input_stabilizer_table, output_stabilizer_table in zip(inputs, outputs):
        predict_out = find_program.output_stablizers(input_stabilizer_table)
        assert predict_out.is_eq(output_stabilizer_table)
