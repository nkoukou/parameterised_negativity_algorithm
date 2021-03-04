import numpy as np
import numpy.random as nr
from state_functions import(evolve)
from circuit_components import(makeState, makeGate, makeMeas)

def random_circuit(qudit_num, C1qGate_num, TGate_num, CSUMGate_num,
                   given_state=None, given_measurement=1):
    ''' Inputs:
        qudit_num         - int
        C1qGate_num       - int
        TGate_num         - int
        CSUMGate_num      - int
        given_state       - None or 0 (all zeros) or string
        given_measurement - string or int (number of measurement modes)

        Output:
        circuit = {'state': states, 'gates': gates, 'indices': indices,
                   'meas': measurements}
    '''
    # States
    if given_state is None:
        char = ['0', '1', '2'] # Full list: ['0','1','2','+','m','S','N','T']
        prob = [1/len(char)]*len(char)

        given_state = ''
        for i in range(qudit_num):
            given_state += nr.choice(char, p=prob)
    elif given_state==0:
        given_state = '0'*qudit_num
    else:
        if len(given_state)!=qudit_num:
            raise Exception('Number of qudits must be %d'%(qudit_num))
    states = []
    for s in given_state:
        states.append(makeState(s))

    # Gates
    char1q = ['H', 'S'] # Full list: ['H', 'S', '1']
    prob1q = [1/len(char1q)]*len(char1q)
    gates_seq = []
    for i in range(C1qGate_num):
        gate = makeGate(nr.choice(char1q, p=prob1q))
        index = [nr.randint(qudit_num)]
        gates_seq.append((gate, index))
    for i in range(TGate_num):
        gate = makeGate('T')
        index = [nr.randint(qudit_num)]
        gates_seq.append((gate, index))
    for i in range(CSUMGate_num):
        gate = makeGate('C+')
        index = list(nr.choice(qudit_num, size=2, replace=False))
        gates_seq.append((gate, index))
    nr.shuffle(gates_seq)
    gates, indices = zip(*gates_seq)
    gates, indices = list(gates), list(indices)

    # Measurements
    if type(given_measurement)==int:
        char = ['0'] # Full list: ['0','1','2','+','m','S','N','T']
        prob = [1/len(char)]*len(char)

        meas = ['/']*qudit_num
        for i in range(given_measurement):
            meas[i] = nr.choice(char, p=prob)

        given_measurement = ''
        for m in meas:
            given_measurement += m
    else:
        if len(given_measurement)!=qudit_num:
            raise Exception('Number of qudits is %d'%(qudit_num))
    measurements = []
    for m in given_measurement:
        measurements.append(makeMeas(m))

    circuit = {'state': states, 'gates': gates, 'indices': indices,
               'meas': measurements}
    return circuit

def compress2q_circuit(circuit):
    ''' Returns an equivalent circuit that contains only 2-qudit gates
    '''
    states, gates, indices = circuit['gates'], circuit['indices']

    if isinstance(gates[0], str):
        raise ValueError("Gates should be arrays")

    qudit_num = len(states)
    gate_num = len(gates)

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

def solve_circuit_symbolic(circuit):
    ''' Input  - symbolic circuit
        Output - Born probability, float
    '''
    state_string, gate_seq, measurement_string = circuit

    state = makeState(state_string)

    qudit_num = len(state_string)
    circ_repr = [[] for i in range(qudit_num)]
    gate_tracker = {}
    for i in range(qudit_num):
        gate_tracker[i] = 0
    for gate in gate_seq:
        idx, g = gate[0], gate[1]
        if len(idx)==1:
            circ_repr[idx[0]].append(g)
            gate_tracker[idx[0]] +=1
        elif len(idx)==2:
            idx_max = max(gate_tracker[idx[0]], gate_tracker[idx[1]])
            for i in [0,1]:
                circ_repr[idx[i]].extend(['1' for j in
                  range(idx_max - gate_tracker[idx[i]])])
                gate_tracker[idx[i]] += idx_max - gate_tracker[idx[i]]
                circ_repr[idx[i]].append(g[i])
                gate_tracker[idx[i]] +=1
            idx_max = max(gate_tracker.values())
            for i in range(qudit_num):
                circ_repr[i].extend(['1' for j in
                                       range(idx_max - gate_tracker[i])])
                gate_tracker[i] += (idx_max - gate_tracker[i])
        else: raise Exception('Too many gate indices')
        idx_max = max(gate_tracker.values())
        for i in range(qudit_num):
            circ_repr[i].extend(['1' for j in
                                      range(idx_max - gate_tracker[i])])
            gate_tracker[i] += (idx_max - gate_tracker[i])

    layers = list(map(list, zip(*circ_repr)))
    for layer in layers:
        gate = makeGate(''.join(layer))
        state = evolve(state, gate)

    meas = makeMeas(measurement_string)

    p_born = np.trace(np.dot(state, meas)).real
    return p_born

def solve_circuit_nonsymbolic(circuit):
    ''' Input  - non-symbolic circuit
        Output - Born probability, float
    '''
    state_string, gate_seq, measurement_string = circuit

    # State
    state = 1

    # Gates

    meas = makeMeas(measurement_string)

    p_born = np.trace(np.dot(state, meas)).real
    return p_born












