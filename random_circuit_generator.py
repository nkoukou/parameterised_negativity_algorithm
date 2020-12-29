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
        !!! Gates are completely random.
    '''
    ### state_string
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
            raise Exception('Number of qudits must be %d'%(qudit_num))
        state_string = given_state

    ### gate_sequence
    # Full 1q gate list: ['H', 'S', '1']
    char1q = ['H', 'S']
    # Equal probability
    prob1q = [1/len(char1q)]*len(char1q)

    gates_sequence = []
    for i in range(C1qGate_num):
        char = nr.choice(char1q, p=prob1q)
        gate = [[nr.randint(qudit_num)], char]
        gates_sequence.append(gate)
    for i in range(TGate_num):
        gate = [[nr.randint(qudit_num)], 'T']
        gates_sequence.append(gate)
    for i in range(CSUMGate_num):
        gate = [list(nr.choice(qudit_num, size=2, replace=False)),
                'C+']
        gates_sequence.append(gate)
    nr.shuffle(gates_sequence)

    # measurement_string
    if type(given_measurement)==int:
        # Full meas list: ['0', '+', 'T']
        char1q = ['0', '+', 'T']
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
            raise Exception('Number of qudits is %d'%(qudit_num))
        measurement_string = given_measurement

    circuit_string = [state_string] + [gates_sequence] + [measurement_string]
    return circuit_string

def show_circuit(circuit, return_repr=False):
    '''
    '''
    init_state = circuit[0]
    qudit_num = len(init_state)

    circ_repr = [[init_state[i], '> '] for i in range(qudit_num)]

    gate_tracker = {}
    for i in range(qudit_num):
        gate_tracker[i] = 0
    for gate in circuit[1:-1][0]:
        idx, g = gate[0], gate[1]
        if len(idx)==1:
            circ_repr[idx[0]].append(g)
            gate_tracker[idx[0]] +=1
        elif len(idx)==2:
            idx_max = max(gate_tracker[idx[0]], gate_tracker[idx[1]])
            for i in [0,1]:
                circ_repr[idx[i]].extend(['-' for j in
                  range(idx_max - gate_tracker[idx[i]])])
                gate_tracker[idx[i]] += idx_max - gate_tracker[idx[i]]
                circ_repr[idx[i]].append(g[i])
                gate_tracker[idx[i]] +=1
        else: raise Exception('Too many gate indices')
    idx_max = max(gate_tracker.values())
    for i in range(qudit_num):
        circ_repr[i].extend(['-' for j in
                                  range(idx_max - gate_tracker[i])])

    for i in range(qudit_num):
        circ_repr[i].append(' <')
        circ_repr[i].append(circuit[-1][i])


    circ_repr = [''.join(wire) for wire in circ_repr]
    for wire in circ_repr:
        print(wire)
    if return_repr: return circ_repr















