
if __name__ == "__main__":
    from clliford.utills import generate_inout_stabilizer_tables,CllifordProgram
    from clliford.cllifor_gate_parr import CllifordCorrecter
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Clifford,random_clifford
    ## 测试要逼近的电路
    n_qubits = 20
    program = CllifordProgram.random_clifford_program(n_qubits,depth=1)
    print('test program\n',program.to_circuit())
    correcter = CllifordCorrecter(n_qubits,1)
    inputs, outputs = [],[]
    ## 生成输入和输出
    for _ in range(20):
        input_stabilizer_table, output_stabilizer_table = generate_inout_stabilizer_tables(n_qubits,program)
        inputs.append(input_stabilizer_table)
        outputs.append(output_stabilizer_table)
    correcter.add_iout(inputs,inputs)
    ## 求解得到的程序
    find_program = correcter.solve()
    print('find program\n',find_program.to_circuit())
    for input_stabilizer_table, output_stabilizer_table in zip(inputs,outputs):
        print('input')
        print(input_stabilizer_table.to_string())
        print('real output')
        solver_out = program.output_stablizers(input_stabilizer_table)
    # return input_stabilizer_table, output_stabilizer_table
        assert solver_out.is_eq(output_stabilizer_table)
    








    