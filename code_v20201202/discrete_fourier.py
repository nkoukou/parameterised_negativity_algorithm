import numpy as np
'''DIM: Dimension of qudit (for qutrit DIM = 3)'''
DIM = 3

J1q = np.array([[0,1],[-1,0]])

def innerProd(v,A,w):
    return np.dot(v,np.dot(A,w))
    
def FT1q(ll,w):
    return np.exp(-1.0j*(2*np.pi/DIM)*innerProd(ll,J1q,w))
