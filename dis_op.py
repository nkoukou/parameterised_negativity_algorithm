import autograd.numpy as np
#import numpy as np
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

'''Sample Code'''
#
#import itertools
#from make_state import makeState1q
#from discrete_fourier import FT1q
#from smoothing_function import h
#
#x_range = list(range(DIM))
#p_range = list(range(DIM))
#
#rho = makeState1q('T')
#print(rho)
#rhoW = np.zeros((DIM,DIM))
#for w in itertools.product(x_range,repeat=2):
#    w = np.array(w,dtype=int).flatten()
#    rhoW = rhoW + np.trace(np.dot(rho,D1q(-w)))*np.array(D1q(w))
#rhoW = rhoW/DIM
#print(np.real(rhoW))
#
#wlist = []
#for w in itertools.product(x_range,repeat=2):
#    w = np.array(w,dtype=int).flatten()
#    print(w)
#    wlist.append(np.trace(np.dot(rho,D1q(-w))))
#wlist = np.reshape(wlist,(DIM,DIM))
#print(np.real(wlist))
#print(np.imag(wlist))
#print(np.abs(wlist))
#print(np.angle(wlist))
#print(np.abs(wlist).sum()/DIM)
#print(np.log(np.abs(wlist).sum()/DIM))

#
#print(np.trace(np.dot(rho,D1q([0,2]))))
#
#D = np.zeros((DIM,DIM))
#print(D)
#for w in itertools.product(x_range,repeat=2):
#    D = D + tau**(w[0]*w[1]) * np.dot(power(X, w[0]), power(Z, w[1]))
#print(np.real(D)/DIM)
#print(np.imag(D)/DIM)

#W_out = 0
#for w in itertools.product(x_range,repeat=2):
#    w = np.array(w).flatten()
#    W_out = W_out + h(Cov,S,w)*np.trace(np.dot(rho,D1q(-w)))*FT1q(ll,w)
#print(W_out/DIM/DIM)
#
#Dsum = np.zeros((3,3))
#for w in itertools.product(x_range,repeat=2):
#    Dsum = Dsum + D1q(w)
#Dsum = Dsum/DIM
#print(np.real(Dsum))
#print(np.imag(Dsum))
#
#for w in itertools.product(x_range,repeat=2):
#    print(np.real(np.trace(np.dot(Dsum,D1q(w)))))
#
#for w in itertools.product(x_range,repeat=2):
#    w = np.array(w)
#    print(np.real(np.dot(D1q([w[1],-w[0]]),np.dot(Dsum,D1q([-w[1],w[0]])))))
#    print(np.real(np.dot(D1q(w),Dsum)))
#
#for w in itertools.product(x_range,repeat=2):
#    w = np.array(w)
#    A = np.dot(D1q(w),np.dot(Dsum,D1q(-w)))
#    B = D1q(w)
#    print(np.real(A))
#    print(np.real(B))
#    print(np.imag(A))
#    print(np.imag(B))
#    print(np.abs(A-B))
#
