import autograd.numpy as np
import itertools as it

from state_functions import(DIM, tau, power, evolve)
from circuit_components import(makeGate, makeState)

X = makeGate('X')
Z = makeGate('Z')
CSUM = makeGate('C+')

def x2Gamma(x):
    ''' Returns covariance matrix Gamma given array x of independent parameters
        with len(x) = 8 (Gamma is Hermitian with unit trace).
        x takes values in [-1,1].
        Wigner distribution corresponds to x = [1,0,0,0,0,0,1,0].
        Output - (3x3) complex ndarray
    '''
    return np.array([[ x[0],             x[1] + 1.j*x[2],  x[3] + 1.j*x[4] ],
                     [ x[1] - 1.j*x[2],  x[5],             x[6]+1.j*x[7]   ],
                     [ x[3] - 1.j*x[4],  x[6] - 1.j*x[7],  1-x[0]-x[5]     ]
                    ], dtype="complex_")

def get_trace_D(Gamma):
    ''' Returns traces tr[D_{p,q} Gamma] at all phase points x.
        Output - (DIM,DIM) complex ndarray
    '''
    return np.einsum('ijkl,lk->ij', D1q_list, Gamma)

def get_F1q0(Gamma):
    ''' Returns new displacement operator at the origin,
        F_0 = 1/DIM \sum_{p,q} tr[D_{-p,-q} Gamma] D_{p,q}.
        Output - (DIM,DIM) complex ndarray
    '''
    traces = 1/DIM/get_trace_D(Gamma).conj()
    return np.einsum('ij,ijkl->kl', traces, D1q_list)

def get_F1q_list(Gamma):
    ''' Returns new displacement operators at all phase space points x,
        F_{p,q} = D_{p,q} F_0 D_{-p,-q}.
        Output - (DIM,DIM,DIM,DIM) complex ndarray
    '''
    F1q0 = get_F1q0(Gamma)
    return np.einsum('ijkl,lm,ijnm->ijkn', D1q_list, F1q0, D1q_list.conj())

def get_G1q_list(Gamma):
    ''' Returns displaced Gamma matrix,
        G_{p,q} = D_{p,q} Gamma D_{-p,-q}.
        Output - (DIM,DIM,DIM,DIM) complex ndarray
    '''
    return np.einsum('ijkl,lm,ijnm->ijkn', D1q_list, Gamma, D1q_list.conj())

def W_state_1q(rho, Gamma):
    ''' Returns Gamma-distribution of state rho,
        W_rho(p,q) = 1/DIM tr[F_{p,q} rho].
        Output - (DIM, DIM) complex ndarray
    '''
    F1q = 1/DIM*get_F1q_list(Gamma)
    return np.einsum('ijkl,lk->ij', F1q, rho).real

def neg_state_1q(rho, Gamma):
    ''' Calculates \sum_{p,q} |W_rho(p,q)|.
        Output - float
    '''
    return np.abs(W_state_1q(rho, Gamma)).sum()

def W_meas_1q(E, Gamma):
    ''' Returns Gamma-distribution of measurement effect E,
        W_E(p,q) = 1/DIM tr[G_{p,q} E].
        Output - (DIM, DIM) complex ndarray
    '''
    G1q = get_G1q_list(Gamma)
    return np.einsum('ijkl,lk->ij', G1q, E).real

def neg_meas_1q(E, Gamma):
    ''' Calculates \max_{p,q} |W_E(p,q)|.
        Output - float
    '''
    return np.max(np.abs(W_meas_1q(E, Gamma)))

def W_gate_1q(U1q, Gamma_in, Gamma_out):
    ''' Returns Gamma-distribution of 1-qudit gate U,
        W_U(p_out,q_out|p_in,q_in) = 1/DIM tr[U G_{p_in,q_in} U^\dagger
                                              F_{p_out,q_out}].
        Output - (DIM, DIM, DIM, DIM) complex ndarray
    '''
    G1q_in = get_G1q_list(Gamma_in)
    F1q_out = 1/DIM*get_F1q_list(Gamma_out)
    rho_ev = np.einsum('lk,ijkn,mn->ijlm', U1q, G1q_in, U1q.conj())
    return np.einsum('ijkl,mnlk->ijmn', rho_ev, F1q_out).real

def neg_gate_1q(U1q, Gamma_in, Gamma_out):
    ''' Calculates \max_{p_in,q_in} 1/DIM *
                   \sum{p_out,q_out}|W_U(p_out,q_out|p_in,q_in)|.
        Output - float
    '''
    return np.abs(W_gate_1q(U1q, Gamma_in, Gamma_out)).sum(axis=(2,3)).max()

def W_gate_2q(U2q, Gamma_in1, Gamma_in2, Gamma_out1, Gamma_out2):
    ''' Returns Gamma-distribution of 2-qudit gate U2q,
        W_U(p1_out,q1_out,p2_out,q2_out|p1_in,q1_in,p2_in,q2_in) =
        1/DIM^2 tr[U (G_{p1_in,q1_in} \otimes G_{p2_in,q2_in}) U^\dagger
                   (F_{p1_out,q1_out} \otimes F_{p2_out,q2_out})].
        Output - (DIM, DIM, DIM, DIM, DIM, DIM, DIM, DIM) complex ndarray
    '''
    G_in = np.einsum('ijkl,mnrs->ijmnkrls',
            get_G1q_list(Gamma_in1), get_G1q_list(Gamma_in2)
           ).reshape((DIM,DIM,DIM,DIM,DIM*DIM,DIM*DIM))
    F_out = 1/DIM/DIM*np.einsum('ijkl,mnrs->ijmnkrls',
                       get_F1q_list(Gamma_out1), get_F1q_list(Gamma_out2)
                      ).reshape((DIM,DIM,DIM,DIM,DIM*DIM,DIM*DIM))
    U_ev = np.einsum('lk,ijsrkn,mn->ijsrlm', U2q, G_in, U2q.conj())
    return np.einsum('ijsrkl,mnablk->ijsrmnab', U_ev, F_out).real

def neg_gate_2q(U2q, Gamma_in1, Gamma_in2, Gamma_out1, Gamma_out2):
    print('neg_gate_2q')
    neg_list = []
    G1_in = get_G1q_list(Gamma_in1)
    G2_in = get_G1q_list(Gamma_in2)
    F1_out = get_F1q_list(Gamma_out1)
    F2_out = get_F1q_list(Gamma_out2)
    for ll_in in it.product(range(DIM), repeat=4):
        p1_in,q1_in,p2_in,q2_in = ll_in[0],ll_in[1],ll_in[2],ll_in[3]
        G_in = np.kron(G1_in[p1_in,q1_in],G2_in[p2_in,q2_in])
        rho_ev = evolve(G_in, U2q)
        neg = 0
        for ll_out in it.product(range(DIM),repeat=4):
            p1_out, q1_out = ll_out[0], ll_out[1]
            p2_out, q2_out = ll_out[2], ll_out[3]
            F_out = np.kron(F1_out[p1_out,q1_out],F2_out[p2_out,q2_out])
            neg = neg + np.abs(np.trace(np.dot(rho_ev,F_out)))
        neg_list.append(neg)
    return np.max(neg_list)/DIM/DIM

def neg_gate_CSUM(GammaC_in, GammaT_in, GammaC_out, GammaT_out):
    print('neg_gate_CSUM')
    G0_in = np.kron(GammaC_in, GammaT_in)
    FC_out = get_F1q_list(GammaC_out)
    FT_out = get_F1q_list(GammaT_out)
    neg = 0
    for ll in it.product(range(DIM), repeat=4):
        p1,q1,p2,q2 = ll[0],ll[1],ll[2],ll[3]
        rho_ev = evolve(G0_in, CSUM)
        F_out = np.kron(FC_out[p1,q1], FT_out[p2,q2])
        neg += np.abs(np.trace(np.dot(rho_ev, F_out)))
    return neg/DIM/DIM

def neg_gate_Cliff_2q(U2q, GammaC_in, GammaT_in, GammaC_out, GammaT_out):
    print('neg_gate_Cliff_2q')
    G0_in = np.kron(GammaC_in, GammaT_in)
    FC_out = get_F1q_list(GammaC_out)
    FT_out = get_F1q_list(GammaT_out)
    neg = 0
    for ll in it.product(range(DIM), repeat=4):
        p1,q1,p2,q2 = ll[0],ll[1],ll[2],ll[3]
        rho_ev = evolve(G0_in, U2q)
        F_out = np.kron(FC_out[p1,q1], FT_out[p2,q2])
        neg += np.abs(np.trace(np.dot(rho_ev,F_out)))
    return neg/DIM/DIM

########################### DISPLACEMENT OPERATORS ############################
def D1q(w):
    ''' Returns 1-qudit displacement operator at phase space location w.
        w - tuple
    '''
    p = w[0]%DIM
    q = w[1]%DIM
    return tau**(p*q) * np.dot(power(X, p), power(Z, q))

def allD1qs():
    ''' Returns displacement operators D_x at all DIM*DIM phase space points x.
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

'''Sample Code'''
Gamma_w = x2Gamma([1,0,0,0,0,0,1,0])
Gamma_in1 = x2Gamma(2*np.random.rand(8)-1)
Gamma_in2 = x2Gamma(2*np.random.rand(8)-1)
Gamma_out1 = x2Gamma(2*np.random.rand(8)-1)
Gamma_out2 = x2Gamma(2*np.random.rand(8)-1)

U2q = makeGate('C+')
w = W_gate_2q(U2q, Gamma_in1, Gamma_in2, Gamma_out1, Gamma_out2)

current = w
def test(U2q, Gamma_in1, Gamma_in2, Gamma_out1, Gamma_out2):
    G_in = np.einsum('ijkl,mnrs->ijmnkrls',
            get_G1q_list(Gamma_in1), get_G1q_list(Gamma_in2)
           ).reshape((DIM,DIM,DIM,DIM,DIM*DIM,DIM*DIM))
    F_out = 1/DIM/DIM*np.einsum('ijkl,mnrs->ijmnkrls',
                       get_F1q_list(Gamma_out1), get_F1q_list(Gamma_out2)
                      ).reshape((DIM,DIM,DIM,DIM,DIM*DIM,DIM*DIM))
    U_ev = np.einsum('lk,ijsrkn,mn->ijsrlm', U2q, G_in, U2q.conj())
    return np.einsum('ijsrkl,mnablk->ijsrmnab', U_ev, F_out).real
print(w)
print('----------------------------------------------------------------')
print(test(U2q, Gamma_in1, Gamma_in2, Gamma_out1, Gamma_out2))
print(np.all(np.isclose(current,
              test(U2q, Gamma_in1, Gamma_in2, Gamma_out1, Gamma_out2))))












