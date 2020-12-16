import numpy as np
import numpy.random as nr
from circuit_components import(makeGate)

def random_circuit(qudit_num, C1qGate_num, TGate_num, CSUMGate_num,
                   given_state=None, given_measurement=1):
    ''' Creates a random circuit in the form
        [state_string, gate_sequence, meas_string]
        qudit_num, C1qGate_num, TGate_num, CSUMGate_num - int
        given_state - None or string (state_string)
        given_measurement - string (measurement_string)
                            or int (number of measurement modes)
        Currently gates are completely random.
    '''
    # state_string
    if given_state is None:
        # Full state list: ['0', '1', '2', '+', 'm', 'S', 'N', 'T']
        char1q = ['0', '1', '2', 'T']
        # Equal probability
        prob1q = [1/len(char1q)]*len(char1q)

        state_string = ''
        for i in range(qudit_num):
            state_string += nr.choice(char1q, p=prob1q)

    else:
        if len(given_state)!=qudit_num:
            raise Exception('Number of qubits must be %d'%(qudit_num))
        state_string = given_state

    # gate_sequence
    char1q = ['H', 'S', '1']
    prob1q = [1/len(char1q)]*len(char1q)

    gates_sequence = []
    for i in range(C1qGate_num):
        char = nr.choice(char1q, p=prob1q)
        gate = [[nr.randint(qudit_num)], makeGate(char)]
        gates_sequence.append(gate)
    for i in range(TGate_num):
        gate = [[nr.randint(qudit_num)], makeGate('T')]
        gates_sequence.append(gate)
    for i in range(CSUMGate_num):
        gate = [list(nr.choice(qudit_num, size=2, replace=False)),
                makeGate('C+')]
        gates_sequence.append(gate)
    nr.shuffle(gates_sequence)

    # measurement_string
    if type(given_measurement)==int:
        # Full meas list: ['Z', 'X', 'T']
        char1q = ['Z', 'X', 'T']
        # Equal probability
        prob1q = [1/len(char1q)]*len(char1q)

        measurement = ['1']*qudit_num
        for i in range(given_measurement):
            measurement[i] = nr.choice(char1q, p=prob1q)

        measurement_string = ''
        for m in measurement:
            measurement_string += m
    else:
        if len(given_measurement)!=qudit_num:
            raise Exception('Number of qubits is %d'%(qudit_num))
        measurement_string = given_measurement

    circuit_string = [state_string] + [gates_sequence] + [measurement_string]
    return circuit_string