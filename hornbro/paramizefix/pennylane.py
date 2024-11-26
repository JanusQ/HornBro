import pennylane as qml

import jax
from jax import numpy as jnp
from jax import grad, jit
import numpy as np
import optax
from jax.config import config
from qiskit import QuantumCircuit
from sklearn.utils import shuffle
from bugfix.circuit import Circuit
from tqdm import tqdm
from random import randint

config.update("jax_enable_x64", True)


class CircuitTape:
    def __init__(self, circuit_data: list[dict], n_qubits: int):
        self.circuit_data = circuit_data
        self.n_qubits: int = n_qubits

        dev = qml.device("default.qubit", wires=n_qubits)

        @jax.jit
        @qml.qnode(dev, interface="jax")  # interface如果没有jax就不会有梯度
        def run(input_state, params=None):
            return self.to_pennylane_circuit(input_state, params)
        self.run = run

    def __call__(self, input_states: jnp.ndarray, params: jnp.ndarray = None) -> jnp.ndarray:

        results = []
        for input_state in tqdm(input_states):
            results.append(self.run(input_state, params))

        return results

        return jax.vmap(self.run, in_axes=(0, None))(input_states, params)
        # dev = qml.device("default.qubit", wires=self.n_qubits)
        # return qml.QNode(self.to_pennylane_circuit, dev)(input_state, params)

    """ the following methods are used to add gates to the circuit_data """

    def x(self, qubits: list):
        self.circuit_data.append({"name": "x", "qubits": qubits})
        return

    def y(self, qubits: list):
        self.circuit_data.append({"name": "y", "qubits": qubits})
        return

    def z(self, qubits: list):
        self.circuit_data.append({"name": "z", "qubits": qubits})
        return

    def h(self, qubits: list):
        self.circuit_data.append({"name": "h", "qubits": qubits})
        return

    def rx(self, qubits: list, params: list):
        """add a rx gate to the circuit_data

        Args:
            qubits (list): the qubits to apply the gate
            params (list): the rotation angles for each qubit, unit: rad
        """
        self.circuit_data.append(
            {"name": "rx", "qubits": qubits, "params": params})
        return

    def ry(self, qubits: list, params: list):
        """add a ry gate to the circuit_data

        Args:
            qubits (list): the qubits to apply the gate
            params (list): the rotation angles for each qubit, unit: rad
        """
        self.circuit_data.append(
            {"name": "ry", "qubits": qubits, "params": params})
        return

    def rz(self, qubits: list, params: list):
        """add a rz gate to the circuit_data

        Args:
            qubits (list): the qubits to apply the gate
            params (list): the rotation angle
        """
        self.circuit_data.append(
            {"name": "rz", "qubits": qubits, "params": params})
        return

    def s(self, qubits: list):
        """add a s gate to the circuit_data
        Args:
            qubits (list): the qubits to apply the gate
        """
        self.circuit_data.append({"name": "s", "qubits": qubits})
        return

    def cnot(self, qubits: list):
        """add a cnot gate to the circuit_data

        Args:
            qubits (list): [control_qubit, target_qubit]
        """
        self.circuit_data.append({"name": "cx", "qubits": qubits})
        return

    def cx(self, qubits: list):
        """add a cnot gate to the circuit_data


        Args:
            qubits (list): [control_qubit, target_qubit]
        """
        self.circuit_data.append({"name": "cx", "qubits": qubits})
        return

    def cz(self, qubits: list):
        """add a cz gate to the circuit_data

        Args:
            qubits (list): [control_qubit, target_qubit]
        """
        self.circuit_data.append({"name": "cz", "qubits": qubits})
        return

    def swap(self, qubits: list):
        """add a swap gate to the circuit_data


        Args:
            qubits (list): [qubit1, qubit2]
        """
        self.circuit_data.append({"name": "swap", "qubits": qubits})
        return

    def t(self, qubits: list):
        self.circuit_data.append({"name": "t", "qubits": qubits})
        return

    def tdg(self, qubits: list):
        self.circuit_data.append({"name": "tdg", "qubits": qubits})
        return

    def u1(self, qubits: list, params: list):
        """add a u1 gate to the circuit_data

        Args:
            qubits (list): the qubits to apply the gate
            params (list): the phase angles for each qubit, unit: rad
        """
        self.circuit_data.append(
            {"name": "u1", "qubits": qubits, "params": params})
        return

    def u2(self, qubits: list, params: list):
        """add a u2 gate to the circuit_data

        Args:
            qubits (list): the qubits to apply the gate
            params (list): the [phi, lambda] angles for each qubit, unit: rad
        """
        self.circuit_data.append(
            {"name": "u2", "qubits": qubits, "params": params})
        return

    def u3(self, qubits, params):
        self.circuit_data.append(
            {"name": "u3", "qubits": qubits, "params": params})
        return

    def unitary(self, qubits, params):
        self.circuit_data.append(
            {"name": "unitary", "qubits": qubits, "params": params}
        )
        return

    def mcx(self, qubits):
        self.circuit_data.append({"name": "mcx", "qubits": qubits})
        return

    def mcp(self, qubits):
        self.circuit_data.append({"name": "mcp", "qubits": qubits})
        return

    def get_n_params(self):
        """gate paramize size"""
        n_params = 0

        for gate in self.circuit_data:
            if gate["name"] in ('rx', 'x', 'ry', 'y', 'rz', 'z'):
                n_params += 1
            elif gate["name"] in ('u3',):
                n_params += 3

        return n_params

    def gate_params(self):
        """parameterize the gates"""

        params = []
        for gate in self.circuit_data:
            if gate["name"] in ("rx", "ry", "rz", "u3"):
                params += gate["params"]
            if gate["name"] in ('x', 'y', 'z'):
                params.append(jnp.pi)

        return params

    def to_pennylane_circuit(self, input_state=None, params=None):
        if input_state is not None:
            qml.QubitStateVector(input_state, wires=range(self.n_qubits))

        n_params = 0

        for gate in self.circuit_data:
            if params is not None:
                if gate["name"] == "rx" or gate["name"] == "x":
                    qml.RX(params[n_params], wires=gate["qubits"][0])
                    n_params += 1
                elif gate["name"] == "ry" or gate["name"] == "y":
                    qml.RY(params[n_params], wires=gate["qubits"][0])
                    n_params += 1
                elif gate["name"] == "rz" or gate["name"] == "z":
                    qml.RZ(params[n_params], wires=gate["qubits"][0])
                    n_params += 1
                elif gate["name"] == "u3":
                    qml.U3(*params[n_params:n_params + 3],
                           wires=gate["qubits"][0])
                    n_params += 3
                else:
                    self.to_pennylane_gate(gate)
            else:
                for gate in self.circuit_data:
                    self.to_pennylane_gate(gate)

        return qml.probs()

    @classmethod
    def from_clifford_program(cls, clifford_program: list[tuple[str, list]]):
        """convert the clifford program to circuit_data"""
        n_qubits = clifford_program.n_qubits
        tape: CircuitTape = cls([], n_qubits)

        for inst, qubits in clifford_program:
            if inst.lower() == "x":
                tape.x([qubits])
            elif inst.lower() == "y":
                tape.y([qubits])
            elif inst.lower() == "z":
                tape.z([qubits])
            elif inst.lower() == "h":
                tape.h([qubits])
            elif inst.lower() == "s":
                tape.s([qubits])
            elif inst.lower() == "cnot":
                tape.cnot(qubits)
            elif inst.lower() == "cx":
                tape.cx(qubits)
            else:
                raise ValueError("Invalid gate type: {}".format(inst.name))
        return tape

    def to_pennylane_gate(self, gate):
        """convert the circuit_data to pennylane circuit"""
        if gate["name"] == "x":
            qml.PauliX(wires=gate["qubits"][0])

        elif gate["name"] == "y":
            qml.PauliY(wires=gate["qubits"][0])

        elif gate["name"] == "z":
            qml.PauliZ(wires=gate["qubits"][0])

        elif gate["name"] == "h":
            qml.Hadamard(wires=gate["qubits"][0])

        elif gate["name"] == "s":
            qml.S(wires=gate["qubits"][0])

        elif gate["name"] == "rx":
            qml.RX(gate["params"][0], wires=gate["qubits"][0])

        elif gate["name"] == "ry":
            qml.RY(gate["params"][0], wires=gate["qubits"][0])

        elif gate["name"] == "rz":
            qml.RZ(gate["params"][0], wires=gate["qubits"][0])

        elif gate["name"] == "cx":
            qml.CNOT(wires=gate["qubits"])

        elif gate["name"] == "cz":
            qml.CZ(wires=gate["qubits"])

        elif gate["name"] == "swap":
            qml.SWAP(wires=gate["qubits"])

        elif gate["name"] == "t":
            qml.T(wires=gate["qubits"][0])

        elif gate["name"] == "tdg":
            qml.T(wires=gate["qubits"][0]).inv()

        elif gate["name"] == "u1":
            qml.U1(gate["params"][0], wires=gate["qubits"][0])

        elif gate["name"] == "u2":
            qml.U2(gate["params"][0], gate["params"]
                   [1], wires=gate["qubits"][0])

        elif gate["name"] == "u3":
            # raise Exception('qiskit 和 pennylane 的顺序不对')
            qml.U3(
                gate["params"][0],
                gate["params"][1],
                gate["params"][2],
                wires=gate["qubits"][0],
            )

        elif gate["name"] == "unitary":
            qml.QubitUnitary(gate["params"][0], wires=gate["qubits"])

        elif gate["name"] == "mcx":
            qml.MultiControlledX(
                gate["qubits"], control_values="111", wires=range(self.n_qubits)
            )

        else:
            raise ValueError("Gate not recognized")

    def to_qiskit_circuit(self):
        """
        Purpose:  convert the circuit_data to qiskit circuit
        """
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(n_wires)
        for gate in self.circuit_data:
            if gate["name"] == "x":
                qc.x(gate["qubits"][0])

            elif gate["name"] == "y":
                qc.y(gate["qubits"][0])

            elif gate["name"] == "z":
                qc.z(gate["qubits"][0])

            elif gate["name"] == "h":
                qc.h(gate["qubits"][0])

            elif gate["name"] == "rx":
                qc.rx(gate["params"][0], gate["qubits"][0])

            elif gate["name"] == "ry":
                qc.ry(gate["params"][0], gate["qubits"][0])

            elif gate["name"] == "rz":
                qc.rz(gate["params"][0], gate["qubits"][0])

            elif gate["name"] == "cx":
                qc.cx(gate["qubits"][0], gate["qubits"][1])

            elif gate["name"] == "cz":
                qc.cz(gate["qubits"][0], gate["qubits"][1])

        return qc.reverse_bits()

    @staticmethod
    def from_qiskit_circuit(qiskit_circuit: QuantumCircuit):
        return CircuitTape(Circuit(qiskit_circuit.reverse_bits()).gates, qiskit_circuit.num_qubits)

    def optimize_target_output_states(self, input_states, target_output_states, n_epochs=50):
        loss_history = []
        n_params = self.get_n_params()
        n_qubits = self.n_qubits
        n_batch = 10
        print("n_params:", n_params)

        opt = optax.adam(learning_rate=0.1)
        params = jax.random.uniform(jax.random.PRNGKey(
            randint(0, 200)), (n_params,), minval=0, maxval=jnp.pi * 2)
        # params = jnp.array(circuit_tape.gate_params())
        opt_state = opt.init(params)

        def parameterized_circuit(input_state, params):
            """Quantum circuit ansatz"""
            qml.QubitStateVector(input_state, wires=range(n_qubits))
            circuit_tape.to_pennylane_circuit(params)
            return qml.probs()

        @jax.jit
        def loss_fn(params, input_states, target_output_states):
            loss = 0
            dev = qml.device("default.qubit", wires=n_qubits)
            for input_state, target_output_state in zip(input_states, target_output_states):
                output_state = qml.QNode(
                    parameterized_circuit, dev)(input_state, params)
                loss += jnp.mean(jnp.abs(output_state - target_output_state))
            return loss

        with tqdm(total=n_epochs) as pbar:
            for epoch in range(n_epochs):

                input_states, target_output_states = shuffle(
                    input_states, target_output_states)

                loss_val, grads = jax.value_and_grad(loss_fn)(
                    params, input_states[:n_batch], target_output_states[n_batch:])
                updates, opt_state = opt.update(grads, opt_state)
                params = optax.apply_updates(params, updates)

                pbar.update(1)
                loss_val = round(float(loss_val), 7)
                pbar.set_description(f"Loss: {loss_val}")
                loss_history.append(loss_val)

        return loss_history, params


def generate_input_states(n_qubits, n_states=8):
    states = []
    from qiskit.quantum_info import random_clifford

    for _ in range(n_states):
        cllifordgate = random_clifford(n_qubits)
        matrix = cllifordgate.to_matrix()
        init_state = jnp.zeros(2**n_qubits)
        init_state = init_state.at[0].set(1)
        state = jnp.dot(matrix, init_state)
        states.append(state)
    return states


# if __name__ == "__main__":
#     def target_circuit(input_state, n_wires):
#         """Quantum circuit ansatz"""

#         qml.QubitStateVector(input_state, wires=range(n_wires))
#         circuit_tape.to_pennylane_circuit()
#         # we use a sum of local Z's as an observable since a
#         # local Z would only be affected by params on that qubit.
#         # return qml.state()
#         return qml.probs()

#     def parameterized_circuit(input_state, params):
#         """Quantum circuit ansatz"""

#         qml.QubitStateVector(input_state, wires=range(n_wires))
#         circuit_tape.to_pennylane_circuit(params)
#         # we use a sum of local Z's as an observable since a
#         # local Z would only be affected by params on that qubit.
#         # return qml.state()
#         return qml.probs()

#     @jax.jit
#     def loss_fn(params, n_wires):
#         loss = 0
#         dev = qml.device("default.qubit", wires=n_wires)

#         target_output_states = []
#         input_states = generate_input_states(n_wires)

#         for input_state in input_states:
#             target_output_states.append(target_circuit(input_state))

#         for input_state, target_state in zip(input_states, target_output_states):
#             output_state = qml.QNode(
#                 parameterized_circuit, dev)(input_state, params)
#             loss += jnp.mean(jnp.abs(output_state - target_state))
#         return loss

#     def update_step(
#         opt,
#         params,
#         opt_state,
#         n_wires
#     ):
#         @jax.jit
#         def loss_fn(params, input_states, target_output_states):
#             loss = 0
#             dev = qml.device("default.qubit", wires=n_wires)

#             for input_state, target_state in zip(input_states, target_output_states):
#                 output_state = qml.QNode(
#                     parameterized_circuit, dev)(input_state, params)
#                 loss += jnp.mean(jnp.abs(output_state - target_state))
#             return loss

#         input_states = generate_input_states(n_wires)

#         target_output_states = []
#         for input_state in input_states:
#             target_output_states.append(target_circuit(input_state))

#         loss_val, grads = jax.value_and_grad(loss_fn)(params)
#         updates, opt_state = opt.update(grads, opt_state)
#         params = optax.apply_updates(params, updates)
#         return params, opt_state, loss_val

#     n_wires = 3
#     circuit_tape = CircuitTape()
#     circuit_tape.h([0])
#     circuit_tape.h([1])
#     circuit_tape.cx([0, 1])
#     circuit_tape.x([0])
#     circuit_tape.y([1])
#     circuit_tape.z([2])
#     circuit_tape.rx([0], [0.5])
#     circuit_tape.cx([1, 0])

#     from random import randint

#     circuit_tape.to_pennylane_circuit(jnp.ones(9))
#     opt = optax.adam(learning_rate=0.1)

#     # target_output_states = []
#     # input_states = generate_input_states(n_wires)
#     # for input_state in input_states:
#     #     target_output_states.append(target_circuit(input_state))

#     # for input_state, target_state in zip(input_states, target_output_states):
#     #     parm_output_state = parameterized_circuit(input_state, params)
#     #     output_state = target_circuit(input_state)

#     #     print(np.round(parm_output_state, 2))
#     #     print(np.round(output_state, 2))
#     #     # print(np.round(output_state - parm_output_state, 2))
#     #     print(jnp.mean(jnp.abs(parm_output_state - output_state)))
#     #     print()

#     loss_history = []
#     from tqdm import tqdm

#     # params = jnp.zeros(n_params)
#     # params = jax.random.normal(jax.random.PRNGKey(randint(0, 200)), (n_params,))
#     params = jax.random.uniform(jax.random.PRNGKey(
#         randint(0, 200)), (n_params,), minval=0, maxval=jnp.pi * 2)
#     # params = jnp.array(circuit_tape.gate_params())
#     print("n_params:", n_params)
#     opt_state = opt.init(params)

#     with tqdm(total=1000) as pbar:
#         for i in range(1000):
#             params, opt_state, loss_val = update_step(
#                 opt, params, opt_state, n_wires)
#             pbar.update(1)
#             loss_val = round(float(loss_val), 7)
#             pbar.set_description(f"Loss: {loss_val}")
#             loss_history.append(loss_val)
#     print(params)
#     print(loss_history)
