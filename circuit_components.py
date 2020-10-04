import numpy as np
import itertools as it
import state_functions as sf
# import time
from copy_phase_space import CopyPhaseSpace, evolve
np.set_printoptions(precision=4, suppress=True)
MDIR = 'C:/Users/nkouk/Desktop/PhD/Algo/scripts/files/'

DIM = 3
ps1 = CopyPhaseSpace(DIM, 1)
ps2 = CopyPhaseSpace(DIM, 2)

def makeState(state_string):
    ''' Makes a state matrix from the generating state string,
        e.g. '+', '000SS', '012+TSN'.
    '''
    state = 1
    for s in state_string:
        # print(s)
        temp = makeState1q(s)
        state = np.kron(state, temp)
    return state

def calcWstate(state_string, s):
    ''' Calculates the Wigner distribution from the generating state string and
        smoothing parameter s,
        e.g. '+', '000SS', '012+TSN'.
    '''
    wig = 1
    for i in range(len(state_string)):
        state = makeState(state_string[i])
        coords = it.product(*([range(ps1.dim)]*2))
        temp = []
        for x in coords:
            w = 1/ps1.dim * np.real(np.trace(np.dot(state, ps1.A_1q(x, s[i]))))
            if np.isclose(w,0): w = 0
            temp.append(w)
        wig = np.kron(wig, np.array(temp))
    return wig

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

def calcWgate(gate_string, s, trj):
    ''' Calculates the Wigner distribution from the generating gate string and
        smoothing parameter s,
        e.g. 'H', '1HS', '1HSM', '1CT11'.
    '''
    N = len(gate_string)
    rep = np.base_repr(trj, ps1.dim**2).zfill(N)
    s_in, s_out = [-i for i in s[0]], s[1]
    wig = 1
    for i in range(N):
        gate = makeGate1q(gate_string[i])
        wig_sub = []
        xcoords = it.product(*([range(ps1.dim)]*2))
        for x in xcoords:
            ycoords = it.product(*([range(ps1.dim)]*2))
            for y in ycoords:
                Aev = evolve(ps1.A_1q(x, s_in[i]), gate)
                w = 1/ps1.dim * np.real( np.trace(np.dot(
                                ps1.A_1q(y, s_out[i]), Aev)) )
                if np.isclose(w,0): w = 0
                wig_sub.append(w)
        wig_sub = np.array(wig_sub).reshape((ps1.dim**2, ps1.dim**2))
        wig = np.kron(wig, wig_sub[int(rep[i])])
    return wig

def calcWgate_v2(gate_string, trj):
    ''' DEPRICATED - Calculates the Wigner distribution from the generating
        state string,
        e.g. 'H', '1HS', '1HSM', '1CT11'.
    '''
    N = len(gate_string)

    Fs, zs = [], []
    for i in range(N):
        if gate_string[i] in ['C', 'T']:
            c, t = gate_string.find('C'), gate_string.find('T')
            imin, imax = min(c,t), max(c,t)
            idx = np.array([imin, imax-imin-1, N-imax-1])
            invert_ct = True if c>t else False
            F, z = gate2F(gate_string[i], invert_ct)
            F = np.insert(F, [2]*(2*idx[1]), 0, axis=1)
            F = np.pad(F,((0,0),(2*idx[0],2*idx[2])),'constant',
                       constant_values=0)
        else:
            F, z = gate2F(gate_string[i])
            F = np.pad(F,((0,0),(2*i,2*(N-i-1))),'constant', constant_values=0)
        Fs.append(F)
        zs.append(z)
    F = np.vstack(Fs)
    z = np.concatenate(zs).astype(int)

    # basis = ps1.dim**np.arange(2*N)[::-1]
    # xcoords = it.product(*([range(ps1.dim)]*(2*N)))
    # wig = np.zeros((ps1.dim**(2*N), ps1.dim**(2*N)))
    # for x in xcoords:
    #     x = np.array(x)
    #     X = (x*basis).sum()
    #     Y = ( ((np.dot(F, x)+z)%ps1.dim)*basis ).sum()
    #     wig[Y,X] = 1

    rep = np.base_repr(trj, ps1.dim).zfill(2*N)
    x = np.array([int(rep[i]) for i in range(len(rep))])
    basis = ps1.dim**np.arange(2*N)[::-1]
    Yi = ( ((np.dot(np.linalg.inv(F).astype(int), x-z))%ps1.dim)*basis ).sum()
    Yi = ( ((np.dot(F, x)+z)%ps1.dim)*basis ).sum()
    wig = np.zeros(ps1.dim**(2*N))
    wig[Yi]=1

    return wig

def makeMeas(meas_string):
    ''' Makes a measurement projector matrix,
        e.g. '2xx' (projects qudit 0 on state |2>, 'xxSx' (projects qudit 2 on
        state |2>).
    '''
    N = len(meas_string)
    if meas_string.count('x') != N-1: raise Exception(
         'Measurement projection on more than 1 qudits is not implemented.')
    qudit_num = next((i for i, ch  in enumerate(meas_string)
                      if ch not in ['x']),None)
    outcome = meas_string[qudit_num]
    idx = np.array([qudit_num, N - qudit_num - 1])
    ids = ps1.dim**idx

    measp = makeState(outcome)
    measp = np.kron( np.kron(np.eye(ids[0]), measp), np.eye(ids[1]) )
    return measp

def calcWmeas(meas_string, s, trj):
    ''' Calculates the measurement Wigner distribution from the generating
        measurement string and smoothing parameter s,
        e.g. '2xx' (projects qudit 0 on state |2>, 'xxSx' (projects qudit 2 on
        state |2>).
    '''
    N = len(meas_string)
    if meas_string.count('x') != N-1: raise Exception(
         'Measurement projection on more than 1 qudits is not implemented.')
    qudit_num = next((i for i, ch  in enumerate(meas_string)
                      if ch not in ['x']),None)
    s = [s[qudit_num]] #!!!
    outcome = meas_string[qudit_num]
    idx = np.array([qudit_num, N - qudit_num - 1])
    ids = ps1.dim**(2*idx)

    measp = ps1.dim*calcWstate(outcome, s)
    measp = np.kron( np.kron(np.ones(ids[0]), measp),
                     np.ones(ids[1]) )
    return measp[trj]

def calcWmeas_v2(meas_string, qudit_num, outcome, trj):
    ''' DEPRICATED - Calculates the measurement Wigner distribution from
        inputs given.
    '''
    N = len(meas_string)
    idx = np.array([qudit_num, N - qudit_num - 1])
    ids = ps1.dim**(2*idx)

    i = int(outcome)
    measp = np.array([0]*(3*i)+[1]*3+[0]*(6-3*i))
    measp = np.kron( np.kron(np.ones(ids[0]), measp),
                     np.ones(ids[1]) )
    return measp#[trj]

##### AUXILIARY FUNCTIONS #####

def makeState1q(state_string):
    ''' Makes a 1-qudit state matrix from the generating state string:
        '0' - |0><0|
        '1' - |1><1|
        '2' - |2><2|
        '+' - |+><+| (maximally coherent state)
        'm' - 1/d 1  (maximally mixed state)
        'S' - |S><S| (Strange state)
        'N' - |N><N| (Norrell state)
        'T' - |T><T| (T state)
    '''
    if state_string=='0':
        state = ps1.ele_1q(0,0)
    elif state_string=='1':
        state = ps1.ele_1q(1,1)
    elif state_string=='2':
        state = ps1.ele_1q(2,2)
    elif state_string=='+':
        state = sf.maxcoh(DIM)
    elif state_string=='m':
        state = sf.maxmixed(DIM)
    elif state_string=='S':
        state = sf.strange
    elif state_string=='N':
        state = sf.norrell
    elif state_string=='T':
        state = sf.tmagic
    else:
        raise Exception('Invalid state string')
    return state

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
        gate = np.eye(ps1.dim)
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
        gate = np.diag([sf.ksi, 1, 1/sf.ksi])
    elif gate_string=='m':
        gate = np.diag([1/sf.ksi, 1, sf.ksi])
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
    ids = ps1.dim**idx
    temp2 = np.eye(ids[1])

    gate = 0
    for i in range(ps1.dim):
        for j in range(ps1.dim):
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

def gate2F(gate_string, invert_ct=False):
    ''' Creates index matrix from gate_strings 'H', 'S', '1', 'C' and 'T'.
    '''
    if gate_string=='1':
        F = np.array([[1,0],[0,1]])
        z = np.array([0,0])
    if gate_string=='H':
        F = np.array([[0,-1],[1,0]])
        z = np.array([0,0])
    if gate_string=='S':
        F = np.array([[1,0],[1,1]])
        z = np.array([0,ps1.inverse(2)])
    if gate_string=='C':
        if invert_ct:
            F = np.array([[0,0,1,0],[0,-1,0,1]])
        else:
            F = np.array([[1,0,0,0],[0,1,0,-1]])
        z = np.array([0,0,0,0])
    if gate_string=='T':
        if invert_ct:
            F = np.array([[1,0,1,0],[0,1,0,0]])
        else:
            F = np.array([[1,0,1,0],[0,0,0,1]])
        z = np.array([])
    return F, z


# filename = MDIR+'_ws_'+state_string+'.npy'
# try:
#     wig = np.load(filename)
#     return wig
# except: pass
# np.save(filename, wig)















