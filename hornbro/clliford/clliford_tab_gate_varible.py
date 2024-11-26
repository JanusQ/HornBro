import numpy as np
from z3 import *
from typing import List, Tuple
from clliford import StabilizerTable,CllifordProgram

# def Xor(x, y):
#     """
#     Computes the bitwise XOR of two boolean arrays.
#     """
#     return (1-x)*y + (1-y)*x

class CllifordCorrecter:
    def __init__(self, n_qubits,d_max:int = 4):
        self.n = n_qubits
        self.d_max = d_max
        self.Svars, self.Hvars,self.Ivars, self.CNOTvars = self.define_variables()
        # self.X, self.Z, self.P = self.define_tables()
        # self.iniX, self.iniZ, self.iniP = [], [], []
        # self.applyed_gates = False
        # for i in range(self.n):
        #     self.iniX.append([])
        #     for q in range(self.n):
        #         self.iniX[i].append(self.X[i][q])
        #     self.iniZ.append([])
        #     for q in range(self.n):
        #         self.iniZ[i].append(self.Z[i][q])
        #     self.iniP.append(self.P[i])
        self.constraints = []
        self.iout_idx = 0
    def define_variables(self):
        """
        Defines the variables used in the Clliford solver.
        """
        Svars = []
        for q in range(self.n):
            Svars.append([Bool("S_{}_{}".format(q, d)) for d in range(self.d_max)])
        Hvars = []
        for q in range(self.n):
            Hvars.append([Bool("H_{}_{}".format(q, d)) for d in range(self.d_max)])
        Ivars = []
        for q in range(self.n):
            Ivars.append([Bool("I_{}_{}".format(q, d)) for d in range(self.d_max)])
        CNOTvars = {}
        for q in range(self.n):
            CNOTvars[q] = {}
            for t in range(self.n):
                if q!= t:
                    CNOTvars[q][t] = [Bool("CNOT_{}_{}_{}".format(q, t, d)) for d in range(self.d_max)]

        return Svars, Hvars, Ivars, CNOTvars

    def define_tables(self):
        """
        Defines the tables used in the Clliford solver.
        """
        X = []
        Z = []
        P = []
        for i in range(self.n):
            X.append([Bool("X_{}_{}_{}".format(i,q,self.iout_idx)) for q in range(self.n)])
            Z.append([Bool("Z_{}_{}_{}".format(i,q,self.iout_idx)) for q in range(self.n)])
            P.append(Bool("P_{}_{}".format(i,self.iout_idx)))
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

    def apply_hadamard(self, qubit,Hvar):
        # Swap X and Z for the given qubit
        for i in range(self.n):
            temp = self.X[i][qubit]&self.Z[i][qubit]
            self.P[i]  =  If(Hvar, self.P[i]^temp, self.P[i])
            tempX = self.X[i][qubit]
            self.X[i][qubit] = If(Hvar, self.Z[i][qubit], self.X[i][qubit])
            self.Z[i][qubit] = If(Hvar, tempX, self.Z[i][qubit])
            
    def apply_identity(self, qubit, Ivar,d):
        # Apply identity gate on a qubit
        pass
        # for i in range(self.n):
        #     # self.constraints.append(Implies(Ivar, self.Ps[d+1][i] == self.Ps[d][i]))
        #     self.constraints.append(Implies(Ivar, self.Xs[d+1][i][qubit] == self.Xs[d][i][qubit]))
        #     self.constraints.append(Implies(Ivar, self.Zs[d+1][i][qubit] == self.Zs[d][i][qubit]))
        
 
    def apply_cnot(self, control, target, CNOTvar):
        # Apply CNOT gate from control to target
        for i in range(self.n):
            temp = (self.X[i][target]^ self.Z[i][control]^ True) & self.X[i][control] & self.Z[i][target]
            # self.P[i] = Xor(self.P[i], temp)
            self.P[i] = If(CNOTvar, self.P[i]^temp, self.P[i])
            # self.X[i][target] =  Xor(self.X[i][target], self.X[i][control] & CNOTvar)
            self.X[i][target] =  If(CNOTvar, self.X[i][target]^self.X[i][control], self.X[i][target])
            # self.Z[i][control] =  Xor(self.Z[i][control], self.Z[i][target] & CNOTvar)
            self.Z[i][control] =  If(CNOTvar, self.Z[i][control]^self.Z[i][target], self.Z[i][control])

    def apply_phase(self, qubit, PHASEvar):
        # S gate (phase gate) on a qubit
        for i in range(self.n):
            self.P[i] =  Xor(self.P[i], self.X[i][qubit] & self.Z[i][qubit] & PHASEvar)
            self.Z[i][qubit] =  Xor(self.Z[i][qubit], self.X[i][qubit]& PHASEvar)

    
    def apply_gates(self):
        """
        Applies  gates 
        """
        self.applyed_gates = True
        print('Applying gates... can only used once')
        for d in range(self.d_max):
            for q in range(self.n):
                self.apply_hadamard(q,self.Hvars[q][d])
                self.apply_phase(q,self.Svars[q][d])
                for t in range(self.n):
                    if q!= t:
                        self.apply_cnot(q, t,self.CNOTvars[q][t][d])
                        # self.apply_cnot(t, q,self.CNOTvars[t][q][d])
    
    def unique_gate(self):
        """
        one depth and one qubit at a time
        """
        for d in range(self.d_max):
            for q in range(self.n):
                self.constraints.append(Sum([self.Ivars[q][d], self.Hvars[q][d], self.Svars[q][d], *[self.CNOTvars[q][t][d] for t in range(self.n) if q!= t], *[self.CNOTvars[t][q][d] for t in range(self.n) if q!= t]]) == 1)
                
        
    def inverse_cancel(self):
        for d in range(self.d_max-1):
            for q in range(self.n):
                self.constraints.append(Implies(self.Hvars[q][d], Not(self.Hvars[q][d+1])))
                for t in range(self.n):
                    if q!= t:
                        self.constraints.append(Implies(self.CNOTvars[q][t][d], Not(self.CNOTvars[q][t][d+1])))

    def set_out(self,stabilizer_table:StabilizerTable):
        """
        Returns the expression for the equality of the stabilizer table at depth d.
        """
        expr = []
        for i in range(self.n):
            for j in range(self.n):
                expr.append( self.X[i][j] ==  bool(stabilizer_table.X[i][j]))
                expr.append( self.Z[i][j] ==  bool(stabilizer_table.Z[i][j]))
            expr.append( self.P[i] == bool(stabilizer_table.P[i]))
        return And(*expr)
    def set_in(self,stabilizer_table:StabilizerTable):
        """
        Returns the expression for the equality of the stabilizer table at depth d.
        """
        expr = []
        for i in range(self.n):
            for j in range(self.n):
                expr.append( self.iniX[i][j] ==  bool(stabilizer_table.X[i][j]))
                expr.append( self.iniZ[i][j] ==  bool(stabilizer_table.Z[i][j]))
            expr.append( self.iniP[i] == bool(stabilizer_table.P[i]))
        return And(*expr)
        
    def add_iout(self,input_stabilizer_table:StabilizerTable,output_stabilizer_table:StabilizerTable):
        """
        Adds the input and output stabilizer tables to the program.
        """
        self.iout_idx += 1
        # if not self.applyed_gates:
        self.define_tables()
        self.apply_gates()
        self.unique_gate()
        self.inverse_cancel()
        inexpr = self.set_in(input_stabilizer_table)
        
        outexpr = self.set_out(output_stabilizer_table)
        # self.constraints.append(Implies(inexpr, outexpr))
        self.constraints.append(outexpr==True)
        self.constraints.append(inexpr==True)
        
    def solve(self):
        """
        Solves the Clliford solver.
        """
        
        s = Solver()
        for c in self.constraints:
            s.add(c)


        # s.set("parallel.enable", True)
        # s.set("parallel.threads", 50)
        print("Solving...")
        print(s.check())
        fix_program = CllifordProgram(self.n)
        print(s.model())
        if s.check() == sat:
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
                        if q!= t:
                            if m[self.CNOTvars[q][t][d]]:
                                repeat += 1
                                fix_program.cnot(q,t)
                    if repeat > 1:
                        raise ValueError("Multiple gates applied to the same qubit at depth {}".format(d))
            # print(fix_program.to_circuit())
            # return self.result_tabs[-1]
            return fix_program
        
        
    def inference(self,input_stabilizer_table:StabilizerTable,program):
        self.define_tables()
        for i in range(self.n):
            for q in range(self.n):
                self.constraints.append(self.iniX[i][q] == bool(input_stabilizer_table.X[i][q]))   
                self.constraints.append(self.iniZ[i][q] == bool(input_stabilizer_table.Z[i][q]))
            self.constraints.append(self.iniP[i] == bool(input_stabilizer_table.P[i]))
        ## add program
        # print('max depth: ',self.d_max)
        for layeridx,gate in enumerate(program):
            # print(gate)
            if gate[0] == 'H':
                self.constraints.append(self.Hvars[gate[1]][layeridx]==True)
            elif gate[0] == 'S':
                self.constraints.append(self.Svars[gate[1]][layeridx]==True)
            elif gate[0] == 'CNOT':
                self.constraints.append(self.CNOTvars[gate[1][0]][gate[1][1]][layeridx]==True)
            if isinstance(gate[1],int):
                self.constraints.append(self.Ivars[1-gate[1]][layeridx]==True)
        self.unique_gate()
        self.apply_gates()
        self.inverse_cancel()
        s = Solver()
        for c in self.constraints:
            s.add(c)
        s.set("timeout", 10000)
        fix_program = CllifordProgram(self.n)
        print("Solving...")
        print(s.check())
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
                        if q!= t:
                            if m[self.CNOTvars[q][t][d]]:
                                fix_program.cnot(q,t)
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


