
from z3 import *
from .utills import StabilizerTable
from .layercircuit import LayerCllifordProgram
from typing import List, Dict, Tuple
import ray
import traceback
import os
import numpy as np
def _n_half_pis(param) -> int:
    try:
        param = float(param)
        epsilon = (abs(param) + 0.5 * 1e-10) % (np.pi / 2)
        if epsilon > 1e-10:
            raise ValueError(f"{param} is not to a multiple of pi/2")
        multiple = int(np.round(param / (np.pi / 2)))
        return multiple % 4
    except TypeError as err:
        raise ValueError(f"{param} is not bounded") from err

class ConstraintsGenerator:
    __slot__ = ['program', 'n', 'd_max', 'is_soft','singleq_gates', 'twoq_gates', 'basis_gates','soft_constraints', 'X', 'Z', 'P',
                'Xvars','Zvars','Yvars','CZvars','CNOTvars','Svars','Hvars','SXvars','Sdgvars']
    def __init__(self,num_qubits:int,singleq_gates:List[str]=['X','S','H'], twoq_gates:List[str]=['CNOT'], is_soft: bool=True):
        self.soft_constraints = []
        self.n = num_qubits
        self.program = []
        self.d_max = 2
        self.singleq_gates = singleq_gates
        self.is_soft = is_soft
        self.twoq_gates = twoq_gates
        self.basis_gates = self.singleq_gates + self.twoq_gates
        self.define_variables()
    def define_variables(self):
        """
        Defines the variables used in the Clliford solver.
        """
        for gate in self.singleq_gates+['I']:
            SingleQgates = []
            for q in range(self.n):
                SingleQgate_q =  [Bool("{}_{}_{}".format(gate, q, d)) for d in range(self.d_max)]
                SingleQgates.append(SingleQgate_q)
            setattr(self, gate + "vars",SingleQgates)
        for gate in self.twoq_gates:
            twoqvars = {}
            for q in range(self.n):
                twoqvars[q] = {}
                for t in range(self.n):
                    if q!= t:
                        twoqvars[q][t] = [Bool("{}_{}_{}_{}".format(gate, q, t, d)) for d in range(self.d_max)]
            setattr(self, gate+"vars",twoqvars)
        
    def apply_gate_var(self, gate,qubits,var):
        """
        Applies a gate to the Clliford solver.
        """
        if gate.lower() == 'cnot':
            gate = 'cx'
        method = getattr(self, 'apply_'+gate.lower()+'_var', None)
        if method is not None:
            method(qubits,var)
        else:
            raise ValueError('Gate {} is not supported.'.format(gate))
    def apply_gate(self, gate,qubits):
        """
        Applies a gate to the Clliford solver.
        """
        method = getattr(self, 'apply_'+gate.lower(), None)
        if method is not None:
            method(qubits)
        else:
            raise ValueError('Gate {} is not supported.'.format(gate))
    
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
        if self.is_soft:
            solver = Optimize()
            for cons in self.soft_constraints:
                solver.add_soft(simplify(cons))
        else:
            solver = Solver()
            for cons in self.soft_constraints:
                solver.add(simplify(cons))
        
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
    
    def simplify_tab(self):
        """
        Simplifies the stabilizer table.
        """
        for i in range(self.n):
            for j in range(self.n):
                self.X[i][j] = simplify(self.X[i][j])
                self.Z[i][j] = simplify(self.Z[i][j])
            self.P[i] = simplify(self.P[i])
    # def _append_rz(clifford, qubit, multiple):
    #     """Apply an Rz gate to a Clifford.

    #     Args:
    #         clifford (Clifford): a Clifford.
    #         qubit (int): gate qubit index.
    #         multiple (int): z-rotation angle in a multiple of pi/2

    #     Returns:
    #         Clifford: the updated Clifford.
    #     """
    #     if multiple % 4 == 1:
    #         return _append_s(clifford, qubit)
    #     if multiple % 4 == 2:
    #         return _append_z(clifford, qubit)
    #     if multiple % 4 == 3:
    #         return _append_sdg(clifford, qubit)

    #     return clifford
    # def apply_u_var(self, qubits,Uvar):
    #    # If u gate, check if it is a Clifford, and if so, apply it
    #     try:
    #         theta, phi, lambd = tuple(_n_half_pis(par) for par in gate.params)
    #     except ValueError as err:
    #         raise QiskitError("U gate angles must be multiples of pi/2 to be a Clifford") from err
    #     if theta == 0:
    #         clifford = _append_rz(clifford, qargs[0], lambd + phi)
    #     elif theta == 1:
    #         clifford = _append_rz(clifford, qargs[0], lambd - 2)
    #         clifford = _append_h(clifford, qargs[0])
    #         clifford = _append_rz(clifford, qargs[0], phi)
    #     elif theta == 2:
    #         clifford = _append_rz(clifford, qargs[0], lambd - 1)
    #         clifford = _append_x(clifford, qargs[0])
    #         clifford = _append_rz(clifford, qargs[0], phi + 1)
    #     elif theta == 3:
    #         clifford = _append_rz(clifford, qargs[0], lambd)
    #         clifford = _append_h(clifford, qargs[0])
    #         clifford = _append_rz(clifford, qargs[0], phi + 2)
    #     return clifford
    def apply_gates(self):
        """
        Applies  gates 
        """
        self.applyed_gates = True
        # print('Applying gates... can only used once')
        for didx in range(self.d_max):
            for q in range(self.n):
                for gate in self.singleq_gates:
                    self.apply_gate_var(gate,[q],getattr(self, gate + "vars")[q][didx])
                # self.simplify_tab()
                for t in range(self.n):
                    if q!= t:
                        for gate in self.twoq_gates:
                            self.apply_gate_var(gate,[q,t],getattr(self, gate + "vars")[q][t][didx])
                # self.simplify_tab()
    def apply_h(self, qubits):
        # Swap X and Z for the given qubit
        qubit = qubits[0]
        for i in range(self.n):
            self.P[i]  ^=  self.X[i][qubit]&self.Z[i][qubit]
            tempX = self.X[i][qubit]
            self.X[i][qubit] =  self.Z[i][qubit]
            self.Z[i][qubit] =tempX
        
    def apply_h_var(self, qubits,Hvar):
        # Swap X and Z for the given qubit
        qubit = qubits[0]
        for i in range(self.n):
            temp = self.X[i][qubit]&self.Z[i][qubit]
            # self.P[i]  =  If(Hvar, self.P[i]^temp, self.P[i])
            self.P[i]  =  simplify(Xor(self.P[i], Hvar & temp))
            tempX = self.X[i][qubit]
            # self.X[i][qubit] = If(Hvar, self.Z[i][qubit], self.X[i][qubit])
            self.X[i][qubit] = simplify(Or(self.Z[i][qubit]& Hvar, self.X[i][qubit] & ~Hvar))
            # self.Z[i][qubit] = If(Hvar, tempX, self.Z[i][qubit])
            self.Z[i][qubit] = simplify(Or(self.Z[i][qubit] & ~Hvar, tempX & Hvar))
    
    def apply_s_var(self, qubits, PHASEvar):
        # S gate (phase gate) on a qubit
        qubit = qubits[0]
        for i in range(self.n):
            self.P[i] =  simplify(Xor(self.P[i], self.X[i][qubit] & self.Z[i][qubit] & PHASEvar))
            self.Z[i][qubit] =  simplify(Xor(self.Z[i][qubit], self.X[i][qubit]& PHASEvar))
    
    def apply_s(self, qubits):
        qubit = qubits[0]
        for i in range(self.n):
            self.P[i] ^=  self.X[i][qubit]&self.Z[i][qubit]
            self.Z[i][qubit] ^= self.X[i][qubit]
    
    
    def apply_sdg_var(self, qubits, PHASEvar):
        # S gate (phase gate) on a qubit
        qubit = qubits[0]
        for i in range(self.n):
            self.P[i] =  simplify(Xor(self.P[i], self.X[i][qubit] & Not(self.Z[i][qubit]) & PHASEvar))
            self.Z[i][qubit] =  simplify(Xor(self.Z[i][qubit], self.X[i][qubit]& PHASEvar))
    
    def apply_sdg(self, qubits):
        qubit = qubits[0]
        for i in range(self.n):
            self.P[i] ^=  self.X[i][qubit]& Not(self.Z[i][qubit])
            self.Z[i][qubit] ^= self.X[i][qubit]
        
    def apply_sx_var(self, qubits, SXvar):
        # SX gate (sqrt(X) gate) on a qubit
        qubit = qubits[0]
        for i in range(self.n):
            self.P[i]  =   simplify(Xor(self.P[i], Not(self.X[i][qubit]) & self.Z[i][qubit] & SXvar))
            self.X[i][qubit] = simplify(self.X[i][qubit]^(self.Z[i][qubit] & SXvar))
        
    def apply_swap_var(self, qubits, SWAPvar):
        qubit1 , qubit2 = qubits
        for i in range(self.n):
            self.X[i][qubit1] , self.X[i][qubit2] = If(SWAPvar, self.X[i][qubit2], self.X[i][qubit1]), If(SWAPvar, self.X[i][qubit1], self.X[i][qubit2])
            self.Z[i][qubit1] , self.Z[i][qubit2] = If(SWAPvar, self.Z[i][qubit2], self.Z[i][qubit1]), If(SWAPvar, self.Z[i][qubit1], self.Z[i][qubit2])
    
    def apply_swap(self, qubits):
        qubit1 , qubit2 = qubits
        for i in range(self.n):
            self.X[i][qubit1] , self.X[i][qubit2] = self.X[i][qubit2], self.X[i][qubit1]
            self.Z[i][qubit1] , self.Z[i][qubit2] = self.Z[i][qubit2], self.Z[i][qubit1]
            
    def apply_sx(self, qubits):
        qubit = qubits[0]
        for i in range(self.n):
            self.P[i]  ^=  Not(self.X[i][qubit] )& self.Z[i][qubit]
            self.Z[i][qubit] ^=  self.X[i][qubit]
    
    def apply_sxdg_var(self, qubits, Sxdgvar):
        """Apply an SXdg gate to a Clifford.
        Args:
            qubit (int): gate qubit index.    
        """
        qubit = qubits[0]
        
        for i in range(self.n):
            self.P[i]  =  Xor(self.P[i], self.X[i][qubit] & self.Z[i][qubit] & Sxdgvar)
            self.X[i][qubit] =  self.X[i][qubit]^(self.Z[i][qubit] & Sxdgvar)
            
    def apply_sxdg(self, qubits):
        qubit = qubits[0]
        
        for i in range(self.n):
            self.P[i]  =  self.P[i]^(self.X[i][qubit] & self.Z[i][qubit])
            self.X[i][qubit] =  self.X[i][qubit]^self.Z[i][qubit]
            
            
    def apply_x_var(self, qubits, Xvar):
        """Apply an X gate to a Clifford.
        Args:
            
            qubit (int): gate qubit index.
                    
        """
        qubit = qubits[0]
        for i in range(self.n):
            # self.P[i]  =  If(Xvar, self.P[i]^self.Z[i][qubit],self.P[i])
            self.P[i] = simplify(Xor(self.P[i], Xvar & self.Z[i][qubit]))

    
    def apply_x(self, qubits):
        """Apply an X gate to a Clifford.
        Args:
            
            qubit (int): gate qubit index.
                    
        """
        qubit = qubits[0]
        for i in range(self.n):
            self.P[i]  =  self.P[i]^self.Z[i][qubit]


    def apply_y(self, qubits):
        """Apply a Y gate to a Clifford.
        Args:
            
            qubit (int): gate qubit index.
                    
        """
        qubit = qubits[0]
        for i in range(self.n):
            self.P[i]  =  self.P[i]^self.X[i][qubit]^self.Z[i][qubit]
            
    def apply_y_var(self, qubits, Yvar):
        """Apply a Y gate to a Clifford.
        Args:
            
            qubit (int): gate qubit index.
                    
        """
        qubit = qubits[0]
        for i in range(self.n):
            # self.P[i]  =  If(Yvar,self.P[i]^self.X[i][qubit]^self.Z[i][qubit], self.P[i])
            self.P[i]  =  self.P[i]^(Yvar &(self.X[i][qubit]^self.Z[i][qubit]))
        
        
        
    def apply_z(self, qubits):
        qubit = qubits[0]
        for i in range(self.n):
            self.P[i]  =  self.P[i]^self.X[i][qubit]
    
    def apply_z_var(self, qubits, Zvar):
        qubit = qubits[0]
        for i in range(self.n):
            # self.P[i]  =  If(Zvar, self.P[i]^self.X[i][qubit], self.P[i])
            self.P[i]  =  self.P[i]^(self.X[i][qubit] & Zvar)
    
    
    def apply_cz_var(self, qubits, CZvar):
        
        """Apply a CZ gate to a Clifford.
        Args:
            
            control (int): gate control qubit index.
            target (int): gate target qubit index.
        """
        control, target = qubits
        for i in range(self.n):
            # self.P[i]  =  If(CZvar, self.P[i]^(self.X[i][control] & self.X[i][target] & (self.Z[i][control] ^ self.Z[i][target]), self.P[i]))
            self.P[i] = simplify(Xor(self.P[i], CZvar & self.X[i][control] & self.X[i][target] & (self.Z[i][control] ^ self.Z[i][target])))
            # self.Z[i][target] =  If(CZvar,self.Z[i][target]^self.X[i][control],self.Z[i][target])
            self.Z[i][target] = Xor(self.Z[i][target], self.X[i][control] & CZvar)
            
            # self.Z[i][control] =  If(CZvar,  self.Z[i][control]^self.X[i][target],self.Z[i][control])
            self.Z[i][control] = Xor(self.Z[i][control], self.X[i][target] & CZvar)
        
    def apply_cz(self, qubits):
        """Apply a CZ gate to a Clifford.
        Args:
            control (int): gate control qubit index.
            target (int): gate target qubit index.   
        """
        control, target = qubits
        for i in range(self.n):
            self.P[i]  =  self.P[i]^(self.X[i][control] & self.X[i][target] & (self.Z[i][control] ^ self.Z[i][target]))
            self.Z[i][target] =  self.X[i][target]^self.Z[i][control]
            self.Z[i][control] =  self.Z[i][control]^self.X[i][target]
    
    
    
 
 
    def apply_cx_var(self,qubits,  CXvar):
        control, target = qubits
        # Apply CNOT gate from control to target
        for i in range(self.n):
            temp = simplify(((self.X[i][target]^ True) ^ self.Z[i][control]) & self.X[i][control] & self.Z[i][target])
            self.P[i] = Xor(self.P[i], temp& CXvar)
            # self.P[i] = simplify(If(CXvar, self.P[i]^temp, self.P[i]))
            self.X[i][target] =  Xor(self.X[i][target], self.X[i][control] & CXvar)
            # self.X[i][target] =  If(CXvar, self.X[i][target]^self.X[i][control], self.X[i][target])
            self.Z[i][control] =  Xor(self.Z[i][control], self.Z[i][target] & CXvar)
            # self.Z[i][control] =  If(CXvar, self.Z[i][control]^self.Z[i][target], self.Z[i][control])
    def apply_cx(self, qubits):
        control, target = qubits
        # Apply CNOT gate from control to target
        for i in range(self.n):
            temp = ((self.X[i][target]^ True) ^ self.Z[i][control]) & self.X[i][control] & self.Z[i][target]
            self.P[i] = self.P[i]^temp
            self.X[i][target] =  self.X[i][target]^self.X[i][control]
            self.Z[i][control] =  self.Z[i][control]^self.Z[i][target]
        
   

@ray.remote
def add_iout_parrelell(*args, **kwargs):
    try:
        # raise Exception('test')
        return _add_iout(*args, **kwargs)
    except Exception as e:
        raise e
    

def _add_iout(input_stabilizer_table:StabilizerTable,output_stabilizer_table:StabilizerTable, **kwargs):
    """
    Adds the input and output stabilizer tables to the program.
    """
    print('adding iout')
    generator = ConstraintsGenerator(**kwargs)
    generator.set_in(input_stabilizer_table)
    generator.apply_gates()
    return generator.set_out(output_stabilizer_table)
 
 
  
 
class CllifordCorrecter:
    __slot__ = ['program', 'n', 'd_max', 'is_soft','singleq_gates', 'twoq_gates', 'basis_gates','soft_constraints', 'X', 'Z', 'P',
                'Xvars','Zvars','Yvars','CZvars','CNOTvars','Svars','Hvars','SXvars','Sdgvars']
    def __init__(self, program:LayerCllifordProgram, signleq_gates:List[str]=['X','S','H'], twoq_gates:List[str]=['CNOT'],  is_soft : bool = False,time_out_eff:int = 10, insert_layer_indexes:List[int]=None):
        self.program = program
        self.n = program.n_qubits
        depth = program.depth()
        self.is_soft = is_soft
        if insert_layer_indexes is None:
            insert_layer_indexes = np.random.choice(depth, 1, replace=False)
        self.d_max = 2
        self.insert_layer_indexes = insert_layer_indexes
        # print(self.insert_layer_indexes)
        self.constraints = []
        self.soft_constraints = []
        self.iout_idx = 0
        self.time_out_eff = time_out_eff
        self.singleq_gates = signleq_gates
        self.twoq_gates = twoq_gates
        self.define_variables()
    def define_variables(self):
        """
        Defines the variables used in the Clliford solver.
        """
        for gate in self.singleq_gates+['I']:
            SingleQgates = []
            for q in range(self.n):
                SingleQgate_q =  [Bool("{}_{}_{}".format(gate, q, d)) for d in range(self.d_max)]
                SingleQgates.append(SingleQgate_q)
            setattr(self, gate + "vars",SingleQgates)
        for gate in self.twoq_gates:
            twoqvars = {}
            for q in range(self.n):
                twoqvars[q] = {}
                for t in range(self.n):
                    if q!= t:
                        twoqvars[q][t] = [Bool("{}_{}_{}_{}".format(gate, q, t, d)) for d in range(self.d_max)]
            setattr(self, gate+"vars",twoqvars)
    
    def unique_gate(self):
        """
        one depth and one qubit at a time
        """
        for d in range(self.d_max):
            for q in range(self.n):
                gatelist = [getattr(self, gate + "vars")[q][d] for gate in self.singleq_gates+['I']]
                twogatelist = [getattr(self, gate + "vars")[q][t][d] for t in range(self.n) for gate in self.twoq_gates if q!= t]
                gatelist.extend(twogatelist)
                twogatelist = [getattr(self, gate + "vars")[t][q][d] for t in range(self.n) for gate in self.twoq_gates if q!= t]
                gatelist.extend(twogatelist)
                self.constraints.append(Sum(gatelist) == 1)
                
        
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
        insert_index = self.insert_layer_indexes[0]
        beforegates = self.program[:insert_index]
        aftergates = self.program[insert_index:]
        aftergates.reverse()
        mid_inputs = [beforegates.output_stablizers(input_tab) for input_tab in input_stabilizer_tables]
        mid_outputs = [aftergates.output_stablizers(output_tab) for output_tab in output_stabilizer_tables]
        self.constraintssmt2= ray.get([add_iout_parrelell.remote(mid_inputs[i].to_dict(),mid_outputs[i].to_dict(),num_qubits=self.n, is_soft= self.is_soft,singleq_gates= self.singleq_gates, twoq_gates= self.twoq_gates) for i in range(len(input_stabilizer_tables))])
        
        # self.constraintssmt2= [add_iout_parrelell(input_stabilizer_tables[i].to_dict(),output_stabilizer_tables[i].to_dict(),self.is_soft,self.program,self.singleq_gates,self.twoq_gates) for i in range(len(input_stabilizer_tables))]
        
                
        
        
    
    
    def solve(self):
        """
        Solves the Clliford solver.
        """
        self.unique_gate()
        self.inverse_cancel()
        if self.is_soft:
            s = Optimize()
        else:
            s = Solver()
        for c in self.constraints:
            s.add(c)
        
        for file in self.constraintssmt2:
            try:
                s.from_file(file)
                print('read smt2 file successfully')
            except Exception as e:
                print("Error parsing SMT-LIB:")
                raise e
            ## delete the smt2 file
        for file in self.constraintssmt2:
            print(file)
            os.remove(file)
        # for c in self.soft_constraints:
        #     s.add_soft(c)
        s.set("timeout", self.time_out_eff*self.n**5)
        print("Solving...")
        print(s.check())
        fix_program = LayerCllifordProgram(self.n)
        # print(s.model())
        if s.check() == unknown or s.check() == sat:
            m = s.model()
            ## evaluate the model
            # if s.check() == unknown:
            #     satisfied_num = 0
            #     for cons_soft in self.soft_constraints:
            #         if m.evaluate(cons_soft):
            #             satisfied_num += 1
                # print("Satisfied soft constraints portion(%): ", (satisfied_num/len(self.soft_constraints))**100)
            for d in range(len(self.program)):
                fix_program.append(self.program[d])
                if d not in self.insert_layer_indexes:
                    continue
                d = self.insert_layer_indexes.tolist().index(d)
                for d in range(self.d_max):
                    for q in range(self.n):
                        for gate in self.singleq_gates:
                            method = getattr(fix_program, gate.lower())
                            if m[getattr(self, gate + "vars")[q][d]]:
                                method(q)
                        for t in range(self.n):
                            if q!= t:
                                for gate in self.twoq_gates:
                                    method = getattr(fix_program, gate.lower())
                                    if m[getattr(self, gate + "vars")[q][t][d]]:
                                        method(q,t)

            return fix_program
