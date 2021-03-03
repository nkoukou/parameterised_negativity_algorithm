import numpy as np
import matplotlib.pylab as plt
from QUBIT_circuit_components import(makeState, makeGate)
from QUBIT_random_circuit_generator import(random_circuit, show_circuit,compress_circuit)

def string_to_circuit(circuit_string):
    circuit_compressed = compress_circuit(circuit_string)
    state_string_list = circuit_compressed[0]
    gate_compressed_list = circuit_compressed[1]
    meas_string_list = circuit_compressed[2]

    rho_list = []
    for state_string in state_string_list:
        rho_list.append(makeState(state_string))
            
    gate_U2q_list = []
    gate_qudit_index_list = []
    for gate_compressed in gate_compressed_list:
        gate_qudit_index_list.append(gate_compressed[0])
        gate_U2q_list.append(gate_compressed[1])
            
    meas_list = []
    for meas_string in meas_string_list:
        if meas_string=='/':
            continue
        E = makeState(meas_string)
        meas_list.append(E)

    circuit = {'state_list': rho_list, 'gate_list': gate_U2q_list, 'qudit_index_list': gate_qudit_index_list, 'meas_list': meas_list}
    
    return circuit
    
def accum_sum(list):
    accum_sum_list = []
    accum_sum = 0
    for ele in list:
        accum_sum += ele
        accum_sum_list.append(accum_sum)
    return np.array(accum_sum_list)
