import numpy as np
from QUBIT_circuit_components import(makeGate, makeState)

def BValg_circuit(s_string, toffoli=0):
    '''
    Create a circuit of the 'n'-bit Bernstein-Vazirani algorithm.
    Bernstein-Vazirani algorithm
    - For a given function f(x) = x.s, find the string s using the function
    once.
    [INPUT]
    - s_string: the string s in the function f(x)
    - toffoli: if toffoli==0, the output circuit is composed of up to
               two-qubit gates,
               if toffoli==1, then the ouput circuit includes the Toffoli
               gates (3-qubit gate).
    '''
    n_bit = len(s_string)
    Toffoli_matrix = np.array([[1., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 1., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 1., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 1., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 1., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 1., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 1.],
                               [0., 0., 0., 0., 0., 0., 1., 0.]])

    ### State
    state_list = []
    for i in range(n_bit): # x-string part
        state_list.append(makeState('0'))
    for s in s_string: # s-string part
        state_list.append(makeState(s))
    state_list.append(makeState('1')) # Oracle part

    ### Gate
    Gate_list = []
    Index_list = []

    # First Hadamard gates to create the n-bit uniform superposition:
    # (1/\sqrt{2^n})\sum_x |x>
    for i in range(n_bit):
        Index_list.append([i])
        Gate_list.append(makeGate('H'))
    # Make the oracle qubit |-> state
    Index_list.append([2*n_bit])
    Gate_list.append(makeGate('H'))
    # O_f part: Toffoli gates part
    for i in range(n_bit):
        print()
        if toffoli:
            Index_list.append([i,i+n_bit,2*n_bit])
            Gate_list.append(Toffoli_matrix)
        else:
            sub_index_list, sub_gate_list = Toffoli_gate_in_2q([i,i+n_bit,
                                                                2*n_bit])
            for i in range(len(sub_gate_list)):
                Index_list.append(sub_index_list[i])
                Gate_list.append(sub_gate_list[i])

    # Last Hadamard gates to the first n qubits.
    for i in range(n_bit):
        Index_list.append([i])
        Gate_list.append(makeGate('H'))

    ### Measurement
    meas_list = []
    for s in s_string: # Measure in s (what the outcome supposed to be)
        meas_list.append(makeState(s))
    for i in range(n_bit+1):
        meas_list.append(np.eye(2))

    circuit = {'state_list': state_list, 'gate_list': Gate_list,
               'index_list': Index_list, 'meas_list': meas_list}

    return circuit

def Toffoli_gate_in_2q(index):
    '''
    Decompose a Toffoli gate into one- and two-qubit gates.
    '''
    c1, c2, t = index

    Gate_list = [makeGate('H'), makeGate('C+'), makeGate('t'), makeGate('C+'),
                 makeGate('T'), makeGate('C+'), makeGate('t'), makeGate('C+'),
                 makeGate('T'), makeGate('T'), makeGate('C+'), makeGate('H'),
                 makeGate('T'), makeGate('t'), makeGate('C+')]
    Index_list = [[t], [c2,t], [t], [c1,t], [t], [c2,t], [t], [c1,t], [c2],
                  [t], [c1,c2], [t], [c1], [c2], [c1,c2]]
    return Index_list, Gate_list







