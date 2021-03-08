from functools import reduce
import numpy as np
import numpy.random as nr
from QUBIT_circuit_components import(makeState, makeGate, makeMeas)
from QUBIT_state_functions import(evolve)

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
        circuit = {'state_list': states, 'gate_list': gates,
                   'index_list': indices, 'meas_list': measurements}
    '''
    # States
    if given_state is None:
        char = ['0', '1'] # Full list: ['0','1','2','+','m','S','N','T']
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
    char1q = ['H', 'K'] # Full list: ['H', 'K', 'X', 'Z', '1']
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

    circuit = {'state_list': states, 'gate_list': gates,
               'index_list': indices, 'meas_list': measurements}
    return circuit

def random_connected_circuit(qudit_num, C1qGate_num, TGate_num, CSUMGate_num,
                   given_state=None, given_measurement=1):
    ''' Inputs:
        qudit_num         - int
        C1qGate_num       - int
        TGate_num         - int
        CSUMGate_num      - int
        given_state       - None or 0 (all zeros) or string
        given_measurement - string or int (number of measurement modes)

        Output:
        circuit = {'state_list': states, 'gate_list': gates,
                   'index_list': indices, 'meas_list': measurements}

        circuit is fully connected, i.e. there are no disentangled wires.
    '''
    if CSUMGate_num < qudit_num/2:
        raise Exception('Not enough CSUMs for circuit to be fully connected.')

    check = False
    while check is False:
        circuit = random_circuit(qudit_num, C1qGate_num, TGate_num,
                                 CSUMGate_num, given_state, given_measurement)
        check = []
        for idx in circuit['index_list']:
            if len(idx)==1: continue
            for i in [0,1]:
                if idx[i] in check: continue
                check.append(idx[i])
        check = (set(check)==set(np.arange(qudit_num)))

    return circuit



def compress2q_circuit(circuit):
    ''' Returns an equivalent circuit that contains only 2-qudit gates
    '''
    qudit_num = len(circuit['state_list'])
    gates, indices = circuit['gate_list'], circuit['index_list']
    gate_num = len(indices)

    if isinstance(gates[0], str):
        raise ValueError("Gates should be arrays")

    gates_compressed = []
    indices_compressed = []
    u2q_counts = []
    gate_masked = [0]*gate_num
    disentangled_wires = list(range(qudit_num))

    for count, gate in enumerate(gates):
        if len(indices[count])==1: continue

        gates_compressed.append(gate)
        indices_compressed.append(indices[count])
        u2q_counts.append(count)
        gate_masked[count] = 1
        for i in [0, 1]:
            if indices[count][i] in disentangled_wires:
                disentangled_wires.remove(indices[count][i])

    for k in range(len(indices_compressed)):
        u2q_gate, u2q_index = gates_compressed[k], indices_compressed[k]
        u1q = [makeGate('1')]*2
        for i in range(u2q_counts[k]):
            idx, gate = indices[i], gates[i]
            if gate_masked[i]: continue
            if idx[0] not in u2q_index: continue

            if idx[0]==u2q_index[0]:
                u1q[0] = np.dot(gate, u1q[0])
            if idx[0]==u2q_index[1]:
                u1q[1] = np.dot(gate, u1q[1])
            gate_masked[i] +=1
        gates_compressed[k] = np.dot(u2q_gate, np.kron(u1q[0], u1q[1]))

    for k in range(len(indices_compressed)-1, 0, -1):
        u2q_gate, u2q_index = gates_compressed[k], indices_compressed[k]
        u1q = [makeGate('1')]*2
        for i in range(gate_num-1, u2q_counts[k], -1):
            idx, gate = indices[i], gates[i]
            if gate_masked[i]: continue
            if idx[0] not in u2q_index: continue

            if idx[0]==u2q_index[0]:
                u1q[0] = np.dot(u1q[0], gate)
            if idx[0]==u2q_index[1]:
                u1q[1] = np.dot(u1q[1], gate)
            gate_masked[i] +=1
        gates_compressed[k] = np.dot(np.kron(u1q[0], u1q[1]), u2q_gate)

    u1qs = [makeGate('1')]*len(disentangled_wires)
    for i in range(len(gates)):
        if indices[0] not in disentangled_wires: continue

        idx = disentangled_wires.index(indices[0])
        u1qs[idx] = np.dot(makeGate(gates[i]), u1qs[idx])

        gate_masked[i] +=1
    for i, wire in enumerate(disentangled_wires):
        gates_compressed.append(u1qs[i])
        indices_compressed.append([wire])

    duplicates = []
    for i in range(len(gates_compressed)-1):
        gate, gate_next = gates_compressed[i], gates_compressed[i+1]
        idx, idx_next = indices_compressed[i], indices_compressed[i+1]
        if set(idx)!=set(idx_next): continue

        if idx==idx_next:
            gate_next = np.dot(gate_next, gate)
        else:
            swap = makeGate('S')
            gate_next = np.dot(gate_next, np.dot(swap, np.dot(gate, swap)))
        gates_compressed[i+1] = gate_next
        duplicates.append(i)
    for i in duplicates[::-1]:
        gates_compressed.pop(i)
        indices_compressed.pop(i)

    circuit_compressed = {'state_list': circuit['state_list'],
                          'gate_list': gates_compressed,
                          'index_list': indices_compressed,
                          'meas_list': circuit['meas_list']}
    return circuit_compressed

def show_connectivity(circuit):
    ''' Prints a visual circuit representation.
    '''
    qudit_num = len(circuit['state_list'])
    indices = circuit['index_list']
    meas = circuit['meas_list']

    circ_repr = [['> '] for i in range(qudit_num)]
    for i in range(len(indices)):
        idx = indices[i]
        if len(idx)==1: continue
        elif len(idx)==2:
            idle_wires = np.delete(np.arange(qudit_num), idx)
            for j in idle_wires: circ_repr[j].extend(['-'])
            circ_repr[idx[0]].append('o')
            circ_repr[idx[1]].append('+')
        else: raise Exception('show_connectivity not implemented for m>2')
    identity = makeGate('1')
    for i in range(qudit_num):
        m = '/' if np.allclose(meas[i], identity) else 'D'
        circ_repr[i].append(' '+m)


    circ_repr = [''.join(wire) for wire in circ_repr]
    for wire in circ_repr:
        print(wire)
    return circ_repr


def string_to_circuit(circuit_string):
    ''' Converts symbolic circuit to:

        circuit = {'state_list': states, 'gate_list': gates,
                   'index_list': indices, 'meas_list': measurements}
    '''
    state_string_list = circuit_string[0]
    gate_string_list  = circuit_string[1]
    meas_string_list  = circuit_string[2]

    state_list = []
    for state_string in state_string_list:
        state_list.append(makeState(state_string))

    gate_list, index_list = [], []
    for gate in gate_string_list:
        index_list.append(gate[0])
        gate_list.append(makeGate(gate[1]))

    meas_list = []
    for meas_string in meas_string_list:
        # if meas_string=='/': continue
        meas_list.append(makeMeas(meas_string))

    circuit = {'state_list': state_list, 'gate_list': gate_list,
               'index_list': index_list, 'meas_list': meas_list}

    return circuit

def solve_qubit_circuit(circuit):
    ''' Solves !!! qubit circuits.
    '''
    state = reduce(np.kron, circuit['state_list'])
    qudit_num = int(np.log(state.shape[0]))

    identity = makeGate('1')
    for i in range(len(circuit['index_list'])):
        idx, gate = circuit['index_list'][i], circuit['gate_list'][i]

        for j in range(qudit_num - len(idx)):
            gate = np.kron(gate, identity)
            pass


    meas = reduce(np.kron, circuit['meas_list'])
    prob = np.trace(np.dot(meas, state))
    return prob

def solve_BV1q_circuit(circuit):
    ''' Solves compressed 1-qubit BV circuit.
    '''
    states = circuit['state_list']
    s0 = np.kron(states[0], np.kron(states[1], states[2]))

    gates = circuit['gate_list']
    index = circuit['index_list']

    meas = circuit['meas_list'][0]

    gates3q = []
    swap = makeGate('S')
    identity = makeGate('1')

    g = np.kron(identity, gates[0])
    gates3q.append(g)
    g = np.kron(identity, gates[1])
    h = np.kron(swap, identity)
    g = evolve(g, h, is_state=0)
    gates3q.append(g)
    g = np.kron(identity, gates[2])
    gates3q.append(g)
    g = np.kron(identity, gates[3])
    h = np.kron(swap, identity)
    g = evolve(g, h, is_state=0)
    gates3q.append(g)
    g = np.kron(gates[4], identity)
    gates3q.append(g)

    s = s0
    for gate in gates3q:
        s = evolve(s, gate)
    meas = np.kron(meas, np.kron(identity, identity))

    prob = np.trace(np.dot(s, meas))
    return prob








