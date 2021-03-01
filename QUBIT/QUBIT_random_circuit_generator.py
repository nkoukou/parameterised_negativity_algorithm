import numpy as np
import numpy.random as nr
from QUBIT_state_functions import evolve
from QUBIT_circuit_components import(makeGate)

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
        # Full state list: ['0', '1', '+', 'm', 'H', 'T']
        char1q = ['0', '1', 'T']
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
    # Full 1q gate list: ['H', 'K', '1', 'Z', 'X']
    char1q = ['H', 'K']
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

        measurement = ['/']*qudit_num
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
    ''' Prints a visual circuit representation.
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

def compress_circuit(circuit):
    ''' Returns an equivalent circuit that contains only 2-qudit gates
        Input: symbolic circuit (i.e. contains gate strings, e.g. 'H', '1',
                                 etc.)
        Output: non-symbolic circuit (i.e. contains gate matrices , e.g.
                                      makeGate('H'), makeGate('1'), etc.)
    '''
    state_string, gate_seq, measurement_string = circuit
    qudit_num = len(state_string)
    gate_num = len(gate_seq)

    gate_seq_compressed = []
    cnot_counts = []
    gate_masked = [0]*gate_num
    disentangled_wires = list(range(qudit_num))

    for count, gate in enumerate(gate_seq):
        if len(gate[0])==1: continue

        gate_seq_compressed.append([gate[0], 1])
        cnot_counts.append(count)
        gate_masked[count] = 1
        for i in [0, 1]:
            if gate[0][i] in disentangled_wires:
                disentangled_wires.remove(gate[0][i])

    for k, cnot in enumerate(gate_seq_compressed):
        u1q = [makeGate('1')]*2
        for i in range(cnot_counts[k]):
            idx, gate = gate_seq[i]
            if gate_masked[i]: continue
            if idx[0] not in cnot[0]: continue

            if idx[0]==cnot[0][0]:
                u1q[0] = np.dot(makeGate(gate), u1q[0])
            if idx[0]==cnot[0][1]:
                u1q[1] = np.dot(makeGate(gate), u1q[1])
            gate_masked[i] +=1

        u2q = np.dot(makeGate('C+'), np.kron(u1q[0], u1q[1]))
        cnot[1] = u2q

    for k, cnot in reversed(list(enumerate(gate_seq_compressed))):
        u1q = [makeGate('1')]*2
        for i in range(gate_num-1, cnot_counts[k], -1):
            idx, gate = gate_seq[i]
            if gate_masked[i]: continue
            if idx[0] not in cnot[0]: continue

            if idx[0]==cnot[0][0]:
                u1q[0] = np.dot(u1q[0], makeGate(gate))
            if idx[0]==cnot[0][1]:
                u1q[1] = np.dot(u1q[1], makeGate(gate))
            gate_masked[i] +=1

        u2q = np.dot(np.kron(u1q[0], u1q[1]), cnot[1])
        cnot[1] = u2q

    u1qs = [makeGate('1')]*len(disentangled_wires)
    for i, gate in enumerate(gate_seq):
        if gate[0][0] not in disentangled_wires: continue

        idx = disentangled_wires.index(gate[0][0])
        u1qs[idx] = np.dot(makeGate(gate[1]), u1qs[idx])

        gate_masked[i] +=1
    for i, wire in enumerate(disentangled_wires):
        gate_seq_compressed.append([[wire], u1qs[i]])

    duplicates = []
    for i in range(len(gate_seq_compressed)-1):
        gate, gate_next = gate_seq_compressed[i], gate_seq_compressed[i+1]
        if len(gate[0])==1: continue
        if gate[0]!=gate_next[0]: continue

        gate_next[1] = np.dot(gate_next[1], gate[1])

        duplicates.append(i)
    for i in duplicates:
        gate_seq_compressed.remove(gate_seq_compressed[i])

    circuit = [state_string] + [gate_seq_compressed] + [measurement_string]
    return circuit















