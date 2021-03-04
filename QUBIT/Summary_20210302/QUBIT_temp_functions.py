import numpy as np
from QUBIT_circuit_components import(makeState, makeGate, makeMeas)

def string_to_circuit(circuit_string):
    state_string_list = circuit_string[0]
    gate_string_list = circuit_string[1]
    meas_string_list = circuit_string[2]

    state_list = []
    for state_string in state_string_list:
        state_list.append(makeState(state_string))

    gate_list = []
    index_list = []
    for gate in gate_string_list:
        index_list.append(gate[0])
        gate_list.append(makeGate(gate[1]))

    meas_list = []
    for meas_string in meas_string_list:
        meas_list.append(makeMeas(meas_string))

    circuit = {'state_list': state_list, 'gate_list': gate_list,
               'index_list': index_list, 'meas_list': meas_list}

    return circuit

def accum_sum(list):
    accum_sum_list = []
    accum_sum = 0
    for ele in list:
        accum_sum += ele
        accum_sum_list.append(accum_sum)
    return np.array(accum_sum_list)
