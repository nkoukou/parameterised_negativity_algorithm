from random import(shuffle)
import numpy as np
import numpy.random as nr
from QUBIT_circuit_components import(makeState, makeGate, makeMeas)

def random_connected_circuit(qudit_num, circuit_length, Tgate_prob=1/3,
                   given_state=None, given_measurement=1, method='c'):
    ''' Inputs:
        qudit_num         - int
        circuit_length    - int
        Tgate_prob        - float
        given_state       - None or 0 (all zeros) or string
        given_measurement - string or int (number of measurement modes)

        Output:
        circuit = {'state_list': states, 'gate_list': gates,
                   'index_list': indices, 'meas_list': measurements}

        circuit is fully connected, i.e. there are no disentangled wires.
    '''
    # States
    if given_state is None:
        char = ['0', '1']
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

    # Indices
    indices = get_index_list(circuit_length, qudit_num, method)

    # Gates
    gates = []
    if type(Tgate_prob)==int:
        char = ['1', 'H', 'K']
        prob = [1/len(char)]*len(char)
        chars = ['T']*Tgate_prob + [nr.choice(char, p=prob)
                      for g in range(2*circuit_length - Tgate_prob)]
        shuffle(chars)
        for g in range(circuit_length):
            U1qA = makeGate(chars[2*g])
            U1qB = makeGate(chars[2*g+1])
            U_AB_loc = np.kron(U1qA, U1qB)
            csum = 'C+' if indices[g][0]>indices[g][1] else '+C'
            csum = makeGate(csum)
            U_AB_tot = np.dot(U_AB_loc, csum)
            gates.append(U_AB_tot)
        Tcount = Tgate_prob
    else:
        char = ['1', 'H', 'K', 'T']
        prob_list = [(1-Tgate_prob)/3]*(len(char)-1) + [Tgate_prob]
        Tcount = 0
        for g in range(circuit_length):
            U1qA = nr.choice(char, p=prob_list)
            U1qB = nr.choice(char, p=prob_list)
            Tcount +=(U1qA=='T')+(U1qB=='T')
            U1qA = makeGate(U1qA)
            U1qB = makeGate(U1qB)
            U_AB_loc = np.kron(U1qA, U1qB)
            csum = 'C+' if indices[g][0]>indices[g][1] else '+C'
            csum = makeGate(csum)
            U_AB_tot = np.dot(U_AB_loc, csum)
            gates.append(U_AB_tot)

    # Measurements
    if type(given_measurement)==int:
        char = ['0']
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

    return circuit#, Tcount

def get_index_list(circuit_length, qudit_num, method='r'):
    ''' Creates index_list for circuit of given circuit_length, qudit_num
        with given method ('r': random, 'c': canonical)
    '''
    gate_qudit_index_list = []
    if method=='r':
        for gate_index in range(circuit_length):
            rng = nr.default_rng()
            gate_qudit_index = rng.choice(qudit_num, size=2, replace=False)
            gate_qudit_index_list.append(list(gate_qudit_index))

    elif method=='c':
        qudit_index = 0
        for gate_index in range(circuit_length):
            gate_qudit_index_list.append([qudit_index, qudit_index+1])
            qudit_index += 2
            if qudit_index == qudit_num and qudit_num%2 == 0:
                qudit_index = 1
            elif qudit_index == qudit_num-1 and qudit_num%2 == 0:
                qudit_index = 0
            elif qudit_index == qudit_num and qudit_num%2 == 1:
                qudit_index = 0
            elif qudit_index == qudit_num-1 and qudit_num%2 == 1:
                qudit_index = 1
    return gate_qudit_index_list

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
        elif len(idx) in [2,3]:
            idle_wires = np.delete(np.arange(qudit_num), idx)
            for j in idle_wires: circ_repr[j].extend(['-'])
            if len(idx)==3:
                circ_repr[idx[-3]].append('O')
                circ_repr[idx[-2]].append('|')
                circ_repr[idx[-1]].append('+')
            if len(idx)==2:
                circ_repr[idx[-2]].append('c')
                circ_repr[idx[-1]].append('z')
        else:
            # raise Exception('show_connectivity not implemented for n>3')
            print('\nshow_connectivity not implemented for n>3\n')
            return
    idc = makeGate('1')
    for i in range(qudit_num):
        m = '/' if np.allclose(meas[i], idc) else 'D'
        circ_repr[i].append(' '+m)


    circ_repr = [''.join(wire) for wire in circ_repr]
    for wire in circ_repr:
        print(wire)
    # return circ_repr

