import numpy as np
# import itertools as it
# import state_functions as sf

DIM = 3

def gate_sequence2symplectic_form_merged(gate_sequence):
    circuit_length = len(gate_sequence)
    qudit_num = len(gate_sequence[0])
    S_tot = np.eye(2*qudit_num,dtype=int)
    z_tot = np.zeros(2*qudit_num,dtype=int)
    F_seq, z_seq = gate_sequence2symplectic_form(gate_sequence)
    for gate_layer in range(circuit_length):
        S_tot = np.dot(F_seq[gate_layer],S_tot)
        z_tot = np.dot(F_seq[gate_layer],z_tot) + z_seq[gate_layer]
    return S_tot%DIM, np.array(z_tot,dtype=int)%DIM

def gate_sequence2symplectic_form(gate_sequence):
    '''
    A gate sequence has the form ['11CT', 'HSH1', '11SH'].
    Valid characters: '1', 'H', 'S', 'C', 'T'.
    Do not mix with 'C', 'T' with 'H', 'S' in the same gate string/layer.

    Returns list of F and list of z for every gate string/layer.
    '''
    F_seq, z_seq = [], []
    for gate_layer in gate_sequence:
        N = len(gate_layer)

        Fs, zs = [], []
        for i in range(N):
            if gate_layer[i] in ['C', 'T']:
                c, t = gate_layer.find('C'), gate_layer.find('T')
                imin, imax = min(c,t), max(c,t)
                idx = np.array([imin, imax-imin-1, N-imax-1])
                invert_ct = True if c>t else False
                F, z = gate2symp(gate_layer[i], invert_ct)
                F = np.insert(F, [2]*(2*idx[1]), 0, axis=1)
                F = np.pad(F,((0,0),(2*idx[0],2*idx[2])),'constant',
                           constant_values=0)
            else:
                F, z = gate2symp(gate_layer[i])
                F = np.pad(F,((0,0),(2*i,2*(N-i-1))),'constant',
                           constant_values=0)
            Fs.append(F); zs.append(z)
        F, z = np.vstack(Fs), np.concatenate(zs).astype(int)
        F_seq.append(F); z_seq.append(z)
    return F_seq, z_seq

def gate2symp(gate_string, invert_ct=False):
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
        z = np.array([0,get_Inv(2)])
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
    return F%DIM, z%DIM

def get_Inv(p):
    out_list = []
    for index in range(DIM):
        out_list.append(index*p%DIM)
    return out_list.index(1)

def symplectic_inverse(S):
    qudit_num = int(len(S)/2)
    J = makeJ(qudit_num)
    return np.dot(J.T,np.dot(S.T,J))%DIM

def makeJ(N):
    J = np.zeros((2*N,2*N),dtype=int)
    for index in range(N):
        J[2*index,2*index+1] = 1
        J[2*index+1,2*index] = -1
    return J

#from calc_born_prob import makeGate
#from dis_op import D1q
#w = np.array([-1, 1],dtype=int)
#print('w:',w)
#gate = ['S']
#U = makeGate(gate[0])
#D = D1q(w)
#print('U:\n', U)
#print('D(w):\n', D)
#S,z = gate_sequence2symplectic_form(gate)
#print('S\n:', S[0])
#print('z:',z[0])
#
#Sw = np.dot(S[0],w)%DIM
#print('Sw:\t', Sw)
#print('Sz:\t', np.dot(S[0],z[0]))
#
#phase = np.exp(-1.j*2*np.pi/DIM*np.dot(z[0],np.dot(makeJ(1),Sw)))
#
#
#print('Exp[-i(2 pi / DIM ) zJw] D(Sw):\n', phase*D1q(np.dot(S[0],w)))
#print('U.D(w).U^+:\n', np.dot(U,np.dot(D,U.conj().T)))
#
#Diff = phase*D1q(np.dot(S[0],w)) - np.dot(U,np.dot(D,U.conj().T))
#print(Diff)
#
#
#print(np.dot(z[0],np.dot(makeJ(1),Sw)))
#print(np.dot(np.dot(symplectic_inverse(S[0]),z[0]),np.dot(makeJ(1),w)))
#
#print(D.conj().T-D1q(-w))
#
#w1 = np.array([0, 2],dtype=int)
#w2 = np.array([1, 2],dtype=int)
#print(D1q(w1+w2) - np.exp(-1.j*2*np.pi/DIM*np.dot(w1,np.dot(makeJ(1),w2)))*np.dot(D1q(w1),D1q(w2)))


#print(symplectic_inverse(S[0]))
#print(np.dot(symplectic_inverse(S[0]),S[0])%DIM)

#gate_sequence = ['11S11H1','1CT111S']
#SAll, zAll = gate_sequence2symplectic_form(gate_sequence)
#Stot, ztot= gate_sequence2symplectic_form_merged(gate_sequence)
#print(Stot)
#print(ztot)
#
#print(np.dot(SAll[1],SAll[0]))
#
#print(Stot)
#print(symplectic_inverse(Stot))
#print(np.dot(Stot,symplectic_inverse(Stot)))
