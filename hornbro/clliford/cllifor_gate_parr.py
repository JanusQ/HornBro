
from z3 import *
from .utills import StabilizerTable,CllifordProgram
from typing import List, Dict, Tuple
import ray
import os
class ConstraintsGenerator:
    def __init__(self, d_max, n):
 
        self.d_max = d_max
        self.n = n
        self.soft_constraints = []
        self.Svars , self.Hvars ,self.Ivars, self.CNOTvars = self.define_variables()
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

        
    def set_in(self,stabilizer_table:StabilizerTable):
        """
        Returns the expression for the equality of the stabilizer table at depth d.
        """
        X, Z, P = [], [], []
        for i in range(self.n):
            X.append([BoolVal(True) if stabilizer_table['X'][i][j] else BoolVal(False) for j in range(self.n) ])
            Z.append([BoolVal(True) if stabilizer_table['Z'][i][j] else BoolVal(False) for j in range(self.n) ])
            P.append(BoolVal(True) if stabilizer_table['P'][i] else BoolVal(False))
        self.X = X
        self.Z = Z
        self.P = P
    def set_out(self,stabilizer_table:StabilizerTable):
        """
        Returns the expression for the equality of the stabilizer table at depth d.
        """
        for i in range(self.n):
            for j in range(self.n):
                self.soft_constraints.append( self.X[i][j] ==  BoolVal(bool(stabilizer_table['X'][i][j])))
                self.soft_constraints.append( self.Z[i][j] ==  BoolVal(bool(stabilizer_table['Z'][i][j])))
            self.soft_constraints.append( self.P[i] == BoolVal(bool(stabilizer_table['P'][i])))
        print('finished setting out')
        from z3 import Solver, Optimize

        solver = Optimize()
        for cons in self.soft_constraints:
            solver.add_soft(cons)
        
        # Export constraints to SMT-LIB format
        constraints_smtlib = solver.sexpr()
        ## write constraints to file
        import uuid

        idx = uuid.uuid1()
        if not os.path.exists(f"data/constraints/"):
            os.makedirs(f"data/constraints/")
        with open(f"data/constraints/constraints{idx}.smt2", "w") as f:
            f.write(constraints_smtlib)
        return f"data/constraints/constraints{idx}.smt2"
        # return 1
    
    
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
        
    def apply_X(self, qubit, Xvar):
        # Apply X gate on a qubit
        for i in range(self.n):
            self.X[i][qubit] = Xvar
            self.Z[i][qubit] = Not(Xvar)
 
    def apply_cnot(self, control, target, CNOTvar):
        # Apply CNOT gate from control to target
        for i in range(self.n):
            temp = ((self.X[i][target]^ True) ^ self.Z[i][control]) & self.X[i][control] & self.Z[i][target]
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
        # print('Applying gates... can only used once')
        for d in range(self.d_max):
            for q in range(self.n):
                self.apply_hadamard(q,self.Hvars[q][d])
                self.apply_phase(q,self.Svars[q][d])
                for t in range(self.n):
                    if q!= t:
                        self.apply_cnot(q, t,self.CNOTvars[q][t][d])
                        # self.apply_cnot(t, q,self.CNOTvars[t][q][d])


@ray.remote
def add_iout_parrelell(input_stabilizer_table:StabilizerTable,output_stabilizer_table:StabilizerTable,d_max,n):
    """
    Adds the input and output stabilizer tables to the program.
    """
    print('adding iout')
    generator = ConstraintsGenerator(d_max,n)
    generator.set_in(input_stabilizer_table)
    generator.apply_gates()
    return generator.set_out(output_stabilizer_table)
 
class CllifordCorrecter:
    def __init__(self, n_qubits,d_max:int = 4,time_out_eff:int = 10):
        self.n = n_qubits
        self.d_max = d_max
        self.Svars, self.Hvars,self.Ivars, self.CNOTvars = self.define_variables()
        self.constraints = []
        self.soft_constraints = []
        self.iout_idx = 0
        self.time_out_eff = time_out_eff
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

    
    
    
        
    def add_iout(self,input_stabilizer_tables:List[StabilizerTable],output_stabilizer_tables:List[StabilizerTable]):
        """
        Adds the input and output stabilizer tables to the program.
        """
        # self.constraintssmt2 = []
        # ray.init()
        self.constraintssmt2= ray.get([add_iout_parrelell.remote(input_stabilizer_tables[i].to_dict(),output_stabilizer_tables[i].to_dict(),self.d_max,self.n) for i in range(len(input_stabilizer_tables))])
        # from concurrent.futures import ProcessPoolExecutor, TimeoutError
        # num_cpu = os.cpu_count()
        # with ProcessPoolExecutor(max_workers=num_cpu - 2) as executor:
        #     futures = []
        #     for i in range(len(input_stabilizer_tables)):
        #         future = executor.submit(add_iout_parrelell, input_stabilizer_tables[i].to_dict(),output_stabilizer_tables[i].to_dict(),self.d_max,self.n)
        #     futures.append(future)
        # for future in futures:
        #     try:
        #         self.constraintssmt2.append(future.result())
        #     except TimeoutError:
        #         print("TimeoutError")
                
        
        
    
    
    def solve(self):
        """
        Solves the Clliford solver.
        """
        self.unique_gate()
        self.inverse_cancel()
        s = Optimize()
        for c in self.constraints:
            s.add(c)
        
        for file in self.constraintssmt2:
            try:
                s.from_file(file)
            except:
                print("Error parsing SMT-LIB:")
            ## delete the smt2 file
        for file in self.constraintssmt2:
            print(file)
            os.remove(file)
        # for c in self.soft_constraints:
        #     s.add_soft(c)
        s.set("timeout", self.time_out_eff*self.n**5)
        print("Solving...")
        print(s.check())
        fix_program = CllifordProgram(self.n)
        # print(s.model())
        if s.check() == unknown or s.check() == sat:
            m = s.model()
            ## evaluate the model
            satisfied_num = 0
            for cons_soft in self.soft_constraints:
                if m.evaluate(cons_soft):
                    satisfied_num += 1
            # print("Satisfied soft constraints portion(%): ", (satisfied_num/len(self.soft_constraints))**100)
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
            return fix_program
