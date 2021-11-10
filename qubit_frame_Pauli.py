import autograd.numpy as np
import itertools as it

sigma_I = np.array([[1,0],[0,1]],dtype='complex')
sigma_x = np.array([[0,1],[1,0]],dtype='complex')
sigma_y = np.array([[0,-1.j],[1.j,0]],dtype='complex')
sigma_z = np.array([[1,0],[0,-1]],dtype='complex')
Pauli_list = [np.eye(2),sigma_x,sigma_y,sigma_z]

x0 = [0,0,0]
DIM = 2

def F(x):
    [a,b,c] = x
    expa = np.exp(1.j*a)
    expac = np.exp(-1.j*a)
    cb = np.cos(b)
    sb = np.sin(b)
    cc = np.cos(c)
    sc = np.sin(c)

    P0 = sigma_I/DIM
    P1 = np.array([[-sb, expac*cb],[expa*cb,sb]])/DIM
    P2 = np.array([[cb*sc, expac*(sb*sc-1.j*cc)],
                   [expa*(sb*sc+1.j*cc),-cb*sc]])/DIM
    P3 = np.array([[cb*cc, expac*(sb*cc+1.j*sc)],
                   [expa*(sb*cc-1.j*sc),-cb*cc]])/DIM
    return np.array([[P0,P1],[P3,P2]])
#     return np.array([[sigma_I, sigma_x], [sigma_y, sigma_z]])

def G(x):
    [a,b,c] = x
    expa = np.exp(1.j*a)
    expac = np.exp(-1.j*a)
    cb = np.cos(b)
    sb = np.sin(b)
    cc = np.cos(c)
    sc = np.sin(c)

    P0 = sigma_I
    P1 = np.array([[-sb, expac*cb],[expa*cb,sb]])
    P2 = np.array([[cb*sc, expac*(sb*sc-1.j*cc)],
                   [expa*(sb*sc+1.j*cc),-cb*sc]])
    P3 = np.array([[cb*cc, expac*(sb*cc+1.j*sc)],
                   [expa*(sb*cc-1.j*sc),-cb*cc]])
    return np.array([[P0,P1],[P3,P2]])

