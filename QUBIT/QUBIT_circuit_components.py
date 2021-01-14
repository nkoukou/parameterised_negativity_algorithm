import numpy as np
from QUBIT_state_functions import(DIM, omega, ksi, psi2rho, maxcoh, maxmixed,
                            element, inverse)
# import time
np.set_printoptions(precision=4, suppress=True)

def makeState(state_string):
    ''' Makes a state matrix from the generating state string,
        e.g. '+', '000SS', '012+TSN'. !!! Faster p_Born
    '''
    state = 1
    for s in state_string:
        # print(s)
        temp = makeState1q(s)
        state = np.kron(state, temp)
    return state

def makeGate(gate_string):
    ''' Makes a gate matrix from the generating gate string,
        e.g. 'H', '1HS', '1HST', '1C+11'.
    '''
    if gate_string.find('+')>=0:
        gate = makeCsum(gate_string)
        return gate

    gate = 1
    for g in gate_string:
        # print(g)
        temp = makeGate1q(g)
        gate = np.kron(gate, temp)
    return gate

def makeMeas(meas_string):
    ''' Returns the measurement output and mode from the generating all-qubit
    measurement string ('/' - Trace out).
    '''
    MeasO_list = []
    Meas_mode_list = []
    for meas_index in range(len(meas_string)):
        if meas_string[meas_index]=='/':
            continue
        MeasO_list.append(makeState1q(meas_string[meas_index]))
        Meas_mode_list.append(meas_index)
    return MeasO_list, Meas_mode_list

def makeState1q(state_string, dim=DIM):
    ''' Returns a 1-qudit state matrix from the generating state string:
        '0' - |0><0| (Pauli Z basis |0> state)
        '1' - |1><1| (Pauli Z basis |1> state)
        '+' - |+><+| (maximally coherent state)
        'm' - 1/d 1  (maximally mixed state)
        'H' - |H><H| (H state)
        'T' - |T><T| (T state)
        Followed the notations in the Bravyi&Kitaev paper:quant-ph/0403025.
    '''
    if state_string=='0':
        state = psi2rho(np.array([1,0]))
    elif state_string=='1':
        state = psi2rho(np.array([0,1]))
    elif state_string=='+':
        state = maxcoh(dim)
    elif state_string=='m':
        state = maxmixed(dim)
    elif state_string=='H':
        state = psi2rho(np.array([np.cos(np.pi/8), np.sin(np.pi/8)]))
    elif state_string=='T':
        beta = np.arccos(1/np.sqrt(3))/2
        state = psi2rho(np.array([np.cos(beta), ksi*np.sin(beta)]))
    else:
        raise Exception('Invalid state string')
    return state

def makeGate1q(gate_string, dim=DIM):
    ''' Returns a 1-qubit gate matrix from the generating gate string:
        '1' - Identity
        'H' - Hadamard
        'K' - Pi/4-Phase shift gate ([[1,0],[0,i]]) 
        'T' - qubit T-gate (magic gate)
        Followed the notations in the Bravyi&Kitaev paper:quant-ph/0403025.
    '''
    if gate_string=='1':
        gate = np.eye(dim)
    elif gate_string=='H':
        gate = 1/np.sqrt(dim) * np.array([[1., 1.],
                                          [1., -1.]])
    elif gate_string=='K':
        gate = np.diag([1., 1.j])
    elif gate_string=='Z':
        gate = np.diag(omega**np.arange(dim))
    elif gate_string=='X':
        gate = np.roll(np.eye(dim), 1, axis=0)
    elif gate_string=='T':
        gate = ksi/np.sqrt(dim) * np.array([[1., 1.],
                                            [1.j, -1.j]])
    else:
        raise Exception('Invalid 1-q gate string')
    return gate

def makeCsum(gate_string, dim=DIM):
    ''' Makes a 2-qubit C-SUM gate matrix from the generating gate string,
        e.g. '11+1C'.
        'C' - C-SUM gate (control) between two qubits
        'c' - inverse C-SUM gate (control) between two qubits
        '+' - C-SUM & inverse C-SUM gate (target) between two qubits
    '''
    if not(gate_string.count('+')==1 and
           gate_string.count('1')==len(gate_string)-2 and
           (gate_string.count('C')==1 or gate_string.count('c')==1)):
        raise Exception('Invalid 2-q gate string')
    t = gate_string.find('+')
    c1 = gate_string.find('C')
    c2 = gate_string.find('c')
    c = max(c1, c2)
    if c2<0: dagger = False
    else: dagger = True
    N = len(gate_string)

    idx = np.array([min(c,t), max(c,t) - min(c,t) - 1, N - max(c,t) - 1])
    ids = dim**idx
    temp2 = np.eye(ids[1])

    gate = 0
    for i in range(dim):
        for j in range(dim):
            if c<t:
                temp1 = element(i,i, dim)
                temp3 = element(j+i,j, dim) if not dagger else element(j,
                                                                 j+i, dim)
            else:
                temp1 = element(j+i,j, dim) if not dagger else element(j,
                                                                 j+i, dim)
                temp3 = element(i,i, dim)
            temp = np.kron( np.kron(temp1, temp2), temp3 )
            gate += temp
    gate = np.kron( np.kron(np.eye(ids[0]), gate), np.eye(ids[2]) )
    return gate

# def gate2F(gate_string, invert_ct=False, dim=DIM):
#     ''' Creates index matrix from gate_string, e.g. '1SH', 'C1+'.
#     '''
#     Fs, zs = [], []
#     for g in gate_string:
#         F, z = gate2F_aux(g, invert_ct, dim)
#         Fs.append(F); zs.append(z)
#     return Fs, zs # np.vstack(Fs), np.vstack(zs)

# def gate2F_aux(gate_string, invert_ct=False, dim=DIM):
#     ''' Creates index matrix from gate_strings '1', 'H', 'S', 'C' and '+'.
#     '''
#     if gate_string=='1':
#         F = np.array([[1,0],[0,1]])
#         z = np.array([0,0])
#     if gate_string=='H':
#         F = np.array([[0,-1],[1,0]])
#         z = np.array([0,0])
#     if gate_string=='S':
#         F = np.array([[1,0],[1,1]])
#         z = np.array([0,inverse(2, dim)])
#     if gate_string=='C':
#         if invert_ct:
#             F = np.array([[0,0,1,0],[0,-1,0,1]])
#         else:
#             F = np.array([[1,0,0,0],[0,1,0,-1]])
#         z = np.array([0,0,0,0])
#     if gate_string=='+':
#         if invert_ct:
#             F = np.array([[1,0,1,0],[0,1,0,0]])
#         else:
#             F = np.array([[1,0,1,0],[0,0,0,1]])
#         z = np.array([])
#     return F, z

# def makeMeas1q(meas_string, dim=DIM):
#     ''' Returns the measurement output and mode from the generating 1-qudit
#     measurement string:
#         'Z' - Projection on |0>
#         'X' - Projection on |+>
#         'T' - Projection on |T>
#     '''
#     MeasO = makeState1q(meas_string)
#     Meas_mode = meas_string.find(meas_string)
#     return MeasO, Meas_mode


'''Sample Code'''
# state = makeState('1S+')
# gate1q = makeGate('HS1')
# gate2q = makeGate('1C+')
# symp1q = gate2F('H')
# symp2q = gate2F('C+')
# MeasO_list, Meas_mode_list = makeMeas('TT1T1+01//')
# print(len(Meas_mode_list))