import autograd.numpy as np
import itertools as it

from state_functions import(DIM, tau, power)
from circuit_components import(makeGate, makeState)

def x2Gamma(x):
    ''' Returns covariance matrix Gamma given array x of independent parameters
        with len(x) = 8 (Gamma is Hermitian with unit trace).
        x takes values in [-1,1].
        Wigner distribution corresponds to x = [1,0,0,0,0,0,1,0].
        Output - (3x3) complex ndarray
    '''
    return np.array([[ x[0],             x[1] + 1.j*x[2],  x[3] + 1.j*x[4] ],
                     [ x[1] - 1.j*x[2],  x[5],             x[6] + 1.j*x[7] ],
                     [ x[3] - 1.j*x[4],  x[6] - 1.j*x[7],  1 - x[0] - x[5] ]
                    ], dtype="complex_")

def get_trace_D(Gamma):
    ''' Returns traces tr[D_{p,q} Gamma] at all phase points x.
        Output - (DIM,DIM) complex ndarray
    '''
    return np.einsum('ijkl,lk->ij', D1q_list, Gamma)

def get_F1q0(Gamma):
    ''' Returns new displacement operator at the origin,
        F_0 = 1/DIM \sum_{p,q} 1/tr[D_{p,q} Gamma] D_{p,q}. !!! D_{-p,-q}
        Output - (DIM,DIM) complex ndarray
    '''
    traces = 1/get_trace_D(Gamma)
    return 1/DIM * np.einsum('ij,ijkl->kl', traces, D1q_list)

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
    F1q = get_F1q_list(Gamma)
    return 1/DIM * np.real(np.einsum('ijkl,lk->ij', F1q, rho))

def neg_state_1q(rho, Gamma):
    ''' Calculates \sum_{p,q} |W_rho(p,q)|.
        Output - float
    '''
    return np.abs(W_state_1q(rho, Gamma)).sum()

def W_gate_1q(U1q, Gamma_in, Gamma_out):
    ''' Returns Gamma-distribution of 1-qudit gate U,
        W_U(p_out,q_out|p_in,q_in) = 1/DIM tr[U G_{p_in,q_in} U^\dagger
                                              F_{p_out,q_out}].
        Output - (DIM, DIM, DIM, DIM) complex ndarray
    '''
    G1q_in = get_G1q_list(Gamma_in)
    F1q_out = get_F1q_list(Gamma_out)
    rho_ev = np.einsum('lk,ijkn,mn->ijlm', U1q, G1q_in, U1q.conj())
    return 1/DIM * np.real(np.einsum('ijkl,mnlk->ijmn', rho_ev, F1q_out))

def neg_gate_1q(U1q, Gamma_in, Gamma_out):
    ''' Calculates \sum{p_out,q_out}|W_U(p_out,q_out|p_in,q_in)|.
        Output - float
    '''
    return np.abs(W_gate_1q(U1q, Gamma_in, Gamma_out)).sum(axis=(2,3))

def neg_gate_1q_max(U1q, Gamma_in, Gamma_out):
    ''' Calculates \max_{p_in,q_in}
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
    F_out = np.einsum('ijkl,mnrs->ijmnkrls',
                       get_F1q_list(Gamma_out1), get_F1q_list(Gamma_out2)
                      ).reshape((DIM,DIM,DIM,DIM,DIM*DIM,DIM*DIM))
    U_ev = np.einsum('lk,ijsrkn,mn->ijsrlm', U2q, G_in, U2q.conj())
    return 1/DIM/DIM * np.real(np.einsum('ijsrkl,mnablk->ijsrmnab',
                                         U_ev, F_out))

def neg_gate_2q(U2q, Gamma_in1, Gamma_in2, Gamma_out1, Gamma_out2):
    ''' Calculates \sum{p1_out,q1_out, p2_out,q2_out}
                   |W_U(p1_out,q1_out,p2_out,q2_out|p1_in,q1_in,p2_in,q2_in)|.
        Output - float
    '''
    wigner_dist = W_gate_2q(U2q, Gamma_in1, Gamma_in2, Gamma_out1, Gamma_out2)
    return np.abs(wigner_dist).sum(axis=(4,5,6,7))

def neg_gate_2q_max(U2q, Gamma_in1, Gamma_in2, Gamma_out1, Gamma_out2):
    ''' Calculates \max_{p1_in,q1_in,p2_in,q2_in}
                   \sum{p1_out,q1_out, p2_out,q2_out}
                   |W_U(p1_out,q1_out,p2_out,q2_out|p1_in,q1_in,p2_in,q2_in)|.
        Output - float
    '''
    wigner_dist = W_gate_2q(U2q, Gamma_in1, Gamma_in2, Gamma_out1, Gamma_out2)
    return np.abs(wigner_dist).sum(axis=(4,5,6,7)).max()

def W_meas_1q(E, Gamma):
    ''' Returns Gamma-distribution of measurement effect E,
        W_E(p,q) = 1/DIM tr[G_{p,q} E].
        Output - (DIM, DIM) complex ndarray
    '''
    G1q = get_G1q_list(Gamma)
    return np.real(np.einsum('ijkl,lk->ij', G1q, E))

def neg_meas_1q(E, Gamma):
    ''' Calculates \max_{p,q} |W_E(p,q)|.
        Output - float
    '''
    return np.max(np.abs(W_meas_1q(E, Gamma)))

########################### DISPLACEMENT OPERATORS ############################
X = makeGate('X')
Z = makeGate('Z')
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
    print('Done calculating D1qs.')
    return D1q_list
D1q_list = allD1qs()

'''Test Code'''
Gamma_w    = x2Gamma([1,0,0,0,0,0,1,0])
Gamma_t    = x2Gamma([0.9,0.1,0,0,0,0,1,0])
Gamma      = x2Gamma(2*np.random.rand(8)-1)
Gamma_in1  = x2Gamma(2*np.random.rand(8)-1)
Gamma_in2  = x2Gamma(2*np.random.rand(8)-1)
Gamma_out1 = x2Gamma(2*np.random.rand(8)-1)
Gamma_out2 = x2Gamma(2*np.random.rand(8)-1)

# # Test 1q stochastic transformations
# s0  = makeState('0')
# s1  = makeState('+')
# U1q = makeGate('H')
# w0  = W_state_1q(s0, Gamma_in1)
# w1  = W_state_1q(s1, Gamma_out1)
# wh  = W_gate_1q(U1q, Gamma_in1, Gamma_out1)

# sf = np.dot(np.dot(U1q, s0), U1q.conj().T)
# is_evolved = np.allclose(sf, s1)
# wf = np.einsum('ijkl,ij->kl', wh, w0)
# is_stoch = np.allclose(wf, w1)
# print('Stoch-1q')
# # print('U s0 U* = ', sf)
# # print('s1 = ', s1)
# # print('<wh,w0> = ', wf)
# # print('w1 = ', w1)
# print('is_evolved: ', is_evolved)
# print('is_stoch: ', is_stoch)

# # Test 1q T state/gate transformations
# s0  = makeState('T')
# U1q = makeGate('T')
# s1  = np.dot(np.dot(U1q, s0), U1q.conj().T)
# w0  = W_state_1q(s0, Gamma_in1)
# w1  = W_state_1q(s1, Gamma_out1)
# wh  = W_gate_1q(U1q, Gamma_in1, Gamma_out1)

# wf = np.einsum('ijkl,ij->kl', wh, w0)
# is_stoch = np.allclose(wf, w1)
# print('Stoch-1q')
# print('U s0 U* = ', s1)
# print('<wh,w0> = ', wf)
# print('w1 = ', w1)
# print('is_stoch: ', is_stoch)

# Test 2q stochastic transformations
s0a, s0b = makeState('2'), makeState('1')
s1a, s1b = makeState('2'), makeState('0')
# s1a = np.dot(np.dot(makeGate('T'), s0a), makeGate('T').conj().T)
# s1b = np.dot(np.dot(makeGate('H'), s0b), makeGate('H').conj().T)
# U2q      = makeGate('TH')
U2q      = makeGate('C+')
s0, s1   = np.kron(s0a, s0b), np.kron(s1a, s1b)
w0= np.einsum('ij,kl->ijkl', W_state_1q(s0a, Gamma_in1),
                              W_state_1q(s0b, Gamma_in2))
w1= np.einsum('ij,kl->ijkl', W_state_1q(s1a, Gamma_out1),
                              W_state_1q(s1b, Gamma_out2))
wc= W_gate_2q(U2q, Gamma_in1, Gamma_in2, Gamma_out1, Gamma_out2)

sf = np.dot(np.dot(U2q, s0), U2q.conj().T)
is_evolved = np.allclose(sf, s1)
wf = np.einsum('ijklmnsr,ijkl->mnsr', wc, w0)
is_stoch = np.allclose(wf, w1)
print('Stoch-2q')
# print('U s0 U* = ', sf)
# print('s1 = ', s1)
# print('<wc,w0> = ', wf)
# print('w1 = ', w1)
print('is_evolved: ', is_evolved)
print('is_stoch: ', is_stoch)












