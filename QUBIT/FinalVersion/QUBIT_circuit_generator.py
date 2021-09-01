from random import(shuffle)
import numpy as np
import numpy.random as nr
from numpy.linalg import qr
from QUBIT_circuit_components import(makeState, makeGate, makeMeas)

def qr_haar(d):
    ''' Generates a Haar-random matrix using the QR decomposition.
    '''
    A, B = np.random.normal(size=(d,d)), np.random.normal(size=(d,d))
    Z = A + 1j * B
    Q, R = qr(Z)
    Lambda = np.diag([R[i,i] / np.abs(R[i,i]) for i in range(d)])
    return np.dot(Q, Lambda)

def haar_random_connected_circuit(N, L, n, d=2,
                                  given_state=None, given_meas=1, method='c'):
    ''' Generates a circuit with Haar-random n-qudit gates.
        Input:
            N - number of qudits
            L - number of gates
            n - number of qudits a gate acts on
            d - qudit dimension
            given_state - None or 0 (all zeros) or string
            given_meas  - string or int (number of measurement modes)
        Output:
            circuit = {'state_list': states, 'gate_list': gates,
                       'index_list': indices, 'meas_list': measurements}

        circuit is fully connected, i.e. there are no disentangled wires.
    '''
    if n>N:
        raise Exception("n must be less or equal than N")

    # States
    if given_state is None:
        char = ['0', '1']
        prob = [1/len(char)]*len(char)

        given_state = ''
        for i in range(N):
            given_state += nr.choice(char, p=prob)
    elif given_state==0:
        given_state = '0'*N
    else:
        if len(given_state)!=N:
            raise Exception('Number of qudits must be %d'%(N))
    states = []
    for s in given_state:
        states.append(makeState(s))

    # Gates
    indices = get_index_list(L, N, n, method)
    gates = []
    for i in range(L):
        gates.append(qr_haar(d**n))

    # Measurements
    if type(given_meas)==int:
        char = ['0']
        prob = [1/len(char)]*len(char)

        meas = ['/']*N
        for i in range(given_meas):
            meas[i] = nr.choice(char, p=prob)

        given_measurement = ''
        for m in meas:
            given_measurement += m
    else:
        if len(given_meas)!=N:
            raise Exception('Number of qudits is %d'%(N))
    measurements = []
    for m in given_measurement:
        measurements.append(makeMeas(m))

    circuit = {'state_list': states, 'gate_list': gates,
               'index_list': indices, 'meas_list': measurements}
    return circuit

def get_index_list(L, N, n, method='r'):
    ''' Creates index_list for circuit of given circuit_length, qudit_num
        with given method ('r': random, 'c': canonical)
    '''
    gate_qudit_index_list = []
    if method=='r':
        for gate_index in range(L):
            rng = nr.default_rng()
            gate_qudit_index = rng.choice(N, size=n, replace=False)
            gate_qudit_index_list.append(list(gate_qudit_index))

    elif method=='c':
        qudit_index = 0
        for gate_index in range(L):
            gate_qudit_index_list.append([qudit_index, qudit_index+1])
            qudit_index += 2
            if qudit_index == N and N%2 == 0:
                qudit_index = 1
            elif qudit_index == N-1 and N%2 == 0:
                qudit_index = 0
            elif qudit_index == N and N%2 == 1:
                qudit_index = 0
            elif qudit_index == N-1 and N%2 == 1:
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

