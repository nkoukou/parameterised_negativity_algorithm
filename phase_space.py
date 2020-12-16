import autograd.numpy as np
import itertools as it

from state_functions import(DIM, tau, power)
from circuit_components import(makeGate)

x_range = list(range(DIM))
X = makeGate('X')
Z = makeGate('Z')
CSUM = makeGate('C+')

def D1q(w):
    ''' Returns 1-qudit displacement operator at phase space location w.
        w - tuple
    '''
    p = w[0]%DIM
    q = w[1]%DIM
    return tau**(p*q) * np.dot(power(X, p), power(Z, q))

def allD1qs():
    ''' Returns displacement operators at all DIM*DIM phase space locations.
        Output - (DIM,DIM,DIM,DIM) complex ndarray
    '''
    D1q_list = []
    for w in it.product(range(DIM), repeat=2):
        w = np.array(w, dtype=int)
        D1q_list.append(D1q(w))
    D1q_list = np.array(np.reshape(D1q_list,(DIM,DIM,DIM,DIM)),
                        dtype="complex_")
    return D1q_list
D1q_list = allD1qs()

def x2Gamma(x):
    ''' Returns Hermitian matrix Gamma given array of independent parameters x
        with len(x) = 8.
    '''
    return np.array([[ x[0],             x[1] + 1.j*x[2],  x[3] + 1.j*x[4] ],
                     [ x[1] - 1.j*x[2],  x[5],             x[6]+1.j*x[7]   ],
                     [ x[3] - 1.j*x[4],  x[6] - 1.j*x[7],  1-x[0]-x[5]     ]
                    ],dtype = "complex_")

def get_trace_D(Gamma):
    ''' Returns traces of all displacement operators.
        Output - (DIM,DIM) complex ndarray
    '''
    out_list = []
    for w in it.product(range(DIM),repeat=2):
        out_list.append(np.trace(np.dot(D1q_list[w], Gamma)))
    return np.reshape(np.array(out_list,dtype = "complex_"),(DIM,DIM))

def get_F1q0(Gamma):
    D_Gamma_list = get_trace_D(Gamma)
    F1q0 = np.zeros((DIM,DIM), dtype="complex_")
    for w in it.product(range(DIM),repeat=2):
        p,q = w[0],w[1]
        F1q0 += D1q_list[-p,-q]/D_Gamma_list[p,q]
    return np.array(F1q0/DIM, dtype="complex_")

def get_F1q_list(Gamma):
    F1q0 = get_F1q0(Gamma)
    F_list = []
    for ll in it.product(range(DIM), repeat=2):
        p,q = ll[0],ll[1]
        F_list.append(np.dot(np.dot(D1q_list[p,q], F1q0), D1q_list[-p,-q]))
    return np.reshape(np.array(F_list, dtype="complex_"), (DIM,DIM,DIM,DIM))

def get_G1q_list(Gamma):
    G_list = []
    for ll in it.product(range(DIM), repeat=2):
        p,q = ll[0],ll[1]
        G_list.append(np.dot(np.dot(D1q_list[p,q], Gamma),D1q_list[-p,-q]))
    return np.reshape(np.array(G_list, dtype="complex_"), (DIM,DIM,DIM,DIM))

def W_state_1q(rho, Gamma):
    ''' Returns Gamma-distribution of rho.
        Output - (DIM, DIM) complex ndarray
    '''
    w_list = []
    F1q = get_F1q_list(Gamma)
    for ll in it.product(range(DIM), repeat=2):
        p,q = ll[0],ll[1]
        w_list.append(np.trace(np.dot(rho,F1q[p,q])))
    return np.reshape(np.real(w_list)/DIM,(DIM,DIM))

def neg_state_1q(rho, Gamma):
    ''' Calculates sum-negativity of the Gamma-distribution of state rho.
    '''
    return np.abs(W_state_1q(rho, Gamma)).sum()

def W_meas_1q(E, Gamma):
    ''' Returns Gamma-distribution of measurement effect E.
    '''
    w_list = []
    G1q = get_G1q_list(Gamma)
    for ll in it.product(range(DIM), repeat=2):
        p,q = ll[0],ll[1]
        w_list.append(np.trace(np.dot(E, G1q[p,q])))
    return np.reshape(np.real(w_list),(DIM,DIM))

def neg_meas_1q(E, Gamma):
    ''' Calculates sum-negativity of the Gamma-distribution of state rho.
    '''
    return np.max(np.abs(W_meas_1q(E, Gamma)))

def W_gate_1q(U1q, Gamma_in, Gamma_out):
    ''' Returns Gamma-distribution of 1-qudit gate U.
    '''
    w_list = []
    G1q_in = get_G1q_list(Gamma_in)
    F1q_out = get_F1q_list(Gamma_out)
    for ll_in in it.product(range(DIM), repeat=2):
        p_in,q_in = ll_in[0],ll_in[1]
        rho_ev = np.dot(np.dot(U1q,G1q_in[p_in,q_in]),U1q.T.conj())
        for ll_out in it.product(range(DIM), repeat=2):
            p_out, q_out = ll_out[0], ll_out[1]
            w_list.append(np.trace(np.dot(rho_ev,F1q_out[p_out,q_out])))
    return np.reshape(np.real(w_list)/DIM,(DIM,DIM,DIM,DIM))

def neg_gate_1q(U1q, Gamma_in, Gamma_out):
    ''' Calculates sum-negativity of the Gamma-distribution of 1-qudit gate
        U1q.
    '''
    neg_list = []
    G1q_in = get_G1q_list(Gamma_in)
    F1q_out = get_F1q_list(Gamma_out)
    for ll_in in it.product(range(DIM), repeat=2):
        p_in,q_in = ll_in[0],ll_in[1]
        rho_ev = np.dot(np.dot(U1q,G1q_in[p_in,q_in]),U1q.T.conj())
        neg = 0
        for ll_out in it.product(range(DIM), repeat=2):
            p_out, q_out = ll_out[0], ll_out[1]
            neg = neg + np.abs(np.trace(np.dot(rho_ev,F1q_out[p_out,q_out])))
        neg_list.append(neg)
    return np.max(neg_list)/DIM

def W_gate_2q(U2q, Gamma1_in, Gamma2_in, Gamma1_out, Gamma2_out):
    ''' Returns Gamma-distribution of 2-qudit gate U2q.
    '''
    w_list = []
    G1_in = get_G1q_list(Gamma1_in)
    G2_in = get_G1q_list(Gamma2_in)
    F1_out = get_F1q_list(Gamma1_out)
    F2_out = get_F1q_list(Gamma2_out)
    for ll_in in it.product(range(DIM), repeat=4):
        p1_in, q1_in, p2_in, q2_in = ll_in[0], ll_in[1], ll_in[2], ll_in[3]
        G_in = np.kron(G1_in[p1_in,q1_in],G2_in[p2_in,q2_in])
        rho_ev = np.dot(np.dot(U2q,G_in),U2q.T.conj())
        for ll_out in it.product(range(DIM), repeat=4):
            p1_out, q1_out = ll_out[0], ll_out[1]
            p2_out, q2_out = ll_out[2], ll_out[3]
            F_out = np.kron(F1_out[p1_out,q1_out],F2_out[p2_out,q2_out])
            w_list.append(np.trace(np.dot(rho_ev,F_out)))
    return np.reshape(np.real(w_list/DIM/DIM), (DIM,DIM,DIM,DIM,
                                               DIM,DIM,DIM,DIM))

def neg_gate_CSUM(GammaC_in, GammaT_in, GammaC_out, GammaT_out):
    G0_in = np.kron(GammaC_in, GammaT_in)
    FC_out = get_F1q_list(GammaC_out)
    FT_out = get_F1q_list(GammaT_out)
    neg = 0
    for ll in it.product(range(DIM), repeat=4):
        p1,q1,p2,q2 = ll[0],ll[1],ll[2],ll[3]
        rho_ev = np.dot(np.dot(CSUM, G0_in), CSUM.T.conj())
        F_out = np.kron(FC_out[p1,q1], FT_out[p2,q2])
        neg += np.abs(np.trace(np.dot(rho_ev, F_out)))
    return neg/DIM/DIM

def neg_gate_Cliff_2q(U2q, GammaC_in, GammaT_in, GammaC_out, GammaT_out):
    G0_in = np.kron(GammaC_in, GammaT_in)
    FC_out = get_F1q_list(GammaC_out)
    FT_out = get_F1q_list(GammaT_out)
    neg = 0
    for ll in it.product(range(DIM), repeat=4):
        p1,q1,p2,q2 = ll[0],ll[1],ll[2],ll[3]
        rho_ev = np.dot(np.dot(U2q, G0_in), U2q.T.conj())
        F_out = np.kron(FC_out[p1,q1], FT_out[p2,q2])
        neg += np.abs(np.trace(np.dot(rho_ev,F_out)))
    return neg/DIM/DIM

def neg_gate_2q(U2q, Gamma1_in, Gamma2_in, Gamma1_out, Gamma2_out):
    neg_list = []
    G1_in = get_G1q_list(Gamma1_in)
    G2_in = get_G1q_list(Gamma2_in)
    F1_out = get_F1q_list(Gamma1_out)
    F2_out = get_F1q_list(Gamma2_out)
    for ll_in in it.product(x_range,repeat=4):
        p1_in,q1_in,p2_in,q2_in = ll_in[0],ll_in[1],ll_in[2],ll_in[3]
        G_in = np.kron(G1_in[p1_in,q1_in],G2_in[p2_in,q2_in])
        rho_ev = np.dot(np.dot(U2q,G_in), U2q.T.conj())
        neg = 0
        for ll_out in it.product(x_range,repeat=4):
            p1_out,q1_out,p2_out,q2_out = ll_out[0],ll_out[1],ll_out[2],ll_out[3]
            F_out = np.kron(F1_out[p1_out,q1_out],F2_out[p2_out,q2_out])
            neg = neg + np.abs(np.trace(np.dot(rho_ev,F_out)))
        neg_list.append(neg)
    return np.max(neg_list)/DIM/DIM

'''Sample Code'''
#
#import it
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
#for w in it.product(x_range,repeat=2):
#    w = np.array(w,dtype=int).flatten()
#    rhoW = rhoW + np.trace(np.dot(rho,D1q(-w)))*np.array(D1q(w))
#rhoW = rhoW/DIM
#print(np.real(rhoW))
#
#wlist = []
#for w in it.product(x_range,repeat=2):
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
#for w in it.product(x_range,repeat=2):
#    D = D + tau**(w[0]*w[1]) * np.dot(power(X, w[0]), power(Z, w[1]))
#print(np.real(D)/DIM)
#print(np.imag(D)/DIM)

#W_out = 0
#for w in it.product(x_range,repeat=2):
#    w = np.array(w).flatten()
#    W_out = W_out + h(Cov,S,w)*np.trace(np.dot(rho,D1q(-w)))*FT1q(ll,w)
#print(W_out/DIM/DIM)
#
#Dsum = np.zeros((3,3))
#for w in it.product(x_range,repeat=2):
#    Dsum = Dsum + D1q(w)
#Dsum = Dsum/DIM
#print(np.real(Dsum))
#print(np.imag(Dsum))
#
#for w in it.product(x_range,repeat=2):
#    print(np.real(np.trace(np.dot(Dsum,D1q(w)))))
#
#for w in it.product(x_range,repeat=2):
#    w = np.array(w)
#    print(np.real(np.dot(D1q([w[1],-w[0]]),np.dot(Dsum,D1q([-w[1],w[0]])))))
#    print(np.real(np.dot(D1q(w),Dsum)))
#
#for w in it.product(x_range,repeat=2):
#    w = np.array(w)
#    A = np.dot(D1q(w),np.dot(Dsum,D1q(-w)))
#    B = D1q(w)
#    print(np.real(A))
#    print(np.real(B))
#    print(np.imag(A))
#    print(np.imag(B))
#    print(np.abs(A-B))
#
