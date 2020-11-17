import numpy as np
from copy_phase_space import CopyPhaseSpace
from make_state import makeState, makeMeas
from random_circuit_generator import random_circuit_string

DIM = 3
ps1 = CopyPhaseSpace(DIM, 1)
ps2 = CopyPhaseSpace(DIM, 2)
ksi = np.exp(2*np.pi*1.j/9)

'''Auxiliary functions'''

def evolve(X, U):
    ''' Returns UXU^\dagger.
    '''
    return np.dot(U, np.dot(X, U.conj().T))

def makeGate(gate_string):
    ''' Makes a gate matrix from the generating gate string,
        e.g. 'H', '1HS', '1HSM', '1CT11'.
    '''
    if gate_string.find('T')>=0:
        gate = makeCsum(gate_string)
        return gate

    gate = 1
    for g in gate_string:
        # print(g)
        temp = makeGate1q(g)
        gate = np.kron(gate, temp)
    return gate

def makeGate1q(gate_string):
    ''' Makes a 1-qudit gate matrix from the generating gate string:
        '1' - Identity
        'H' - Hadamard
        'h' - inverse Hadamard
        'S' - Phase gate
        's' - inverse Phase gate
        'M' - T gate (magic gate)
        'm' - T gate (magic gate)
    '''
    if gate_string=='1':
        gate = np.eye(DIM)
    elif gate_string=='H':
        gate = ps1.clifford( [[[0,-1],[1,0]]], [(0,0)] )
    elif gate_string=='h':
        gate = ps1.clifford( [[[0,-1],[1,0]]], [(0,0)] ).conj().T
    elif gate_string=='S':
        gate = ps1.clifford( [[[1,0],[1,1]]], [(0,ps1.inverse(2))] )
    elif gate_string=='s':
        gate = ps1.clifford( [[[1,0],[1,1]]], [(0,ps1.inverse(2))] ).conj().T
    elif gate_string=='Z':
        gate = ps1.Z
    elif gate_string=='z':
        gate = ps1.Z.conj() #.T
    elif gate_string=='X':
        gate = ps1.X
    elif gate_string=='x':
        gate = ps1.X.conj().T
    elif gate_string=='M':
        gate = np.diag([ksi, 1, 1/ksi])
    elif gate_string=='m':
        gate = np.diag([1/ksi, 1, ksi])
    else:
        raise Exception('Invalid gate string')
    return gate

def makeCsum(gate_string):
    ''' Makes a 2-qudit C-SUM gate matrix from the generating gate string,
        e.g. '11T1C'.
        'C' - C-SUM gate (control) between two qudits
        'c' - inverse C-SUM gate (control) between two qudits
        'T' - C-SUM & inverse C-SUM gate (target) between two qudits
    '''
    t = gate_string.find('T')
    c1 = gate_string.find('C')
    c2 = gate_string.find('c')
    c = max(c1, c2)
    if c2<0: dagger = False
    else: dagger = True
    N = len(gate_string)

    idx = np.array([min(c,t), max(c,t) - min(c,t) - 1, N - max(c,t) - 1])
    ids = DIM**idx
    temp2 = np.eye(ids[1])

    gate = 0
    for i in range(DIM):
        for j in range(DIM):
            if c<t:
                temp1 = ps1.ele_1q(i,i)
                temp3 = ps1.ele_1q(j+i,j) if not dagger else ps1.ele_1q(j,j+i)
            else:
                temp1 = ps1.ele_1q(j+i,j) if not dagger else ps1.ele_1q(j,j+i)
                temp3 = ps1.ele_1q(i,i)
            temp = np.kron( np.kron(temp1, temp2), temp3 )
            gate += temp
    gate = np.kron( np.kron(np.eye(ids[0]), gate), np.eye(ids[2]) )
    return gate

def simulate_circuit(circuit_string):
    ''' Calculates outcome Born probability for given circuit_string.
    '''
    state0 = makeState(circuit_string[0])
    state  = state0
    gates = []
    for i in range(1,len(circuit_string)-1):
        gate = makeGate(circuit_string[i])
        gates.append(gate)
        state = evolve(state, gate)
    measp = makeMeas(circuit_string[-1])
    prob = np.trace( np.dot(measp, state) )

    if not np.isclose(np.imag(prob), 0):
        raise Exception('Probabilty must be real')
    prob = 0 if np.isclose(prob, 0) else np.real(prob)
    return prob


'''Example Code'''

# circuit_string = random_circuit_string(n=2, L=1)
# print(circuit_string)
# p = simulate_circuit(circuit_string)
# print(p)