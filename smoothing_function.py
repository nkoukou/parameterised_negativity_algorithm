import numpy as np
'''DIM: Dimension of qudit (for qutrit DIM = 3)'''
DIM = 3
k_shift = (DIM-1)/2
def innerProd(v,A,w):
    return np.dot(v,np.dot(A,w))

def w_rescale(w):
    return np.array((w + k_shift)%DIM - k_shift,dtype=int)

def h(Cov, S, w):
    v = w_rescale(np.dot(S,w))
    return np.exp(-innerProd(v, Cov, v))
