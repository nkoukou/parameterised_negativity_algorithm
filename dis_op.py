import numpy as np
from numpy.linalg import matrix_power

'''DIM: Dimension of qudit (for qutrit DIM = 3)'''
DIM = 3

'''Some phase factors'''
tau = -np.exp(1.j*np.pi/DIM)
omega = tau * tau

'''Pauli operators'''
X = np.roll(np.eye(DIM), 1, axis=0)
Z = np.diag(omega**np.arange(DIM))

'''Get 2^{-1}'''
def get_Inv(p):
    out_list = []
    for index in range(DIM):
        out_list.append(index*p%DIM)
    return out_list.index(1)
Inv2 = get_Inv(2)

'''Power of matrix'''
def power(a, p):
#     return matrix_power(a,p%DIM)
    if p==0:
        return np.eye(a.shape[0])
    elif p==1:
        return a
    else:
        return np.dot(a, power(a, (p-1)))

def chi(q):
    return np.exp(1.j*2.*np.pi*q/DIM)

''' Calculates displacement operator D at point w = (p,q).'''
def D1q(w):
    p = w[0]%DIM
    q = w[1]%DIM
    return tau**(p*q) * np.dot(power(X, p), power(Z, q))

import itertools
from make_state import makeState1q

'''Sample Code'''
#x_range = list(range(DIM))
#p_range = list(range(DIM))
#
#rho = makeState1q('1')
#print(rho)
#rhoW = np.zeros((DIM,DIM))
#for w in itertools.product(x_range,repeat=2):
#    w = np.array(w,dtype=int).flatten()
#    rhoW = rhoW + np.trace(np.dot(rho,D1q(-w)))*np.array(D1q(w))
#rhoW = rhoW/DIM
#print(np.real(rhoW))
