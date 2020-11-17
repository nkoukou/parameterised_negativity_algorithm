import numpy as np
import numpy.random as nr

def random_circuit_string(n, L, given_state=None, given_measurement=None,
                          p_csum=0.6):
    ''' Creates a random circuit stringin the form
        ['state', 'gate1', 'gate2', ..., 'gateL', 'measurement'].
        n - integer - number of qudits
        L - integer - circuit depth (number of gate levels)
    '''
    # state_string
    if given_state is None:
        char1q = ['0', '1', '2', 'S', 'N', 'T'] # Can also add: '+'
        prob1q = [1/len(char1q)]*len(char1q)

        state_string = ''
        for i in range(n):
            state_string += nr.choice(char1q, p=prob1q)

    else:
        if len(given_state)!=n: raise Exception('Number of qubits is n')
        state_string = given_state

    # gate_string
    prob = [p_csum, 1-p_csum] # Probability of 1-qudit gate or 2-qudit gate
    char1q = ['H', 'S', '1']
    prob1q = [1/len(char1q)]*len(char1q)

    gates_string = []
    for i in range(L):
        gate_string = ''
        q1 = nr.choice([1,0], p=prob)
        if q1:
            for j in range(n):
                gate_string += nr.choice(char1q, p=prob1q)
        else:
            gate = ['1']*n
            c = nr.choice(np.arange(n))
            t = c
            while t==c:
                t = nr.choice(np.arange(n))
            gate[c] = 'C'
            gate[t] = 'T'
            for g in gate:
                gate_string += g
        gates_string.append(gate_string)

    # measurement_string
    if given_measurement is None:
        char1q = ['X', 'Z', 'T']
        prob1q = [0., 0., 1.] # [1/len(char1q)]*len(char1q)
        basis = nr.choice(char1q, p=prob1q)

        measurement = ['1']*n
        basis_index = nr.choice(np.arange(n))
        measurement[basis_index] = basis

        measurement_string = ''
        for m in measurement:
            measurement_string += m
    else:
        if len(given_measurement)!=n: raise Exception('Number of qubits is n')
        measurement_string = given_measurement
    circuit_string = [state_string] + gates_string + [measurement_string]
    return circuit_string