import autograd.numpy as np
import itertools as it

DIM = 2
x0 = [1.,1/2,1/2]
tau   = np.exp(1.j*np.pi/DIM)

def power(a, p):
    ''' Calculates the matrix power a**p for given 2d-array a and non-negative
        integer p.
    '''
    if p==0:
        return np.eye(a.shape[0])
    elif p==1:
        return a
    else:
        return np.dot(a, power(a, (p-1)))

def F(x):
    return get_F1q_list(x2Gamma(x))/DIM

def G(x):
    return get_G1q_list(x2Gamma(x))

def x2Gamma(x):
    ''' Returns covariance matrix Gamma given array x of independent parameters
        with len(x) = 3 (Gamma is Hermitian with unit trace).
        x takes values in [-1,1].
        Wigner distribution corresponds to x = [1,1/2,1/2]
        Output - (2x2) complex ndarray
    '''
    return np.array([[ x[0],             x[1] - 1.j*x[2]],
                     [ x[1] + 1.j*x[2],  1-x[0],         ]], dtype="complex_")

def get_trace_D(Gamma):
    ''' Returns traces tr[D_{p,q} Gamma] at all phase points x.
        Output - (DIM,DIM) complex ndarray
    '''
    return np.einsum('ijkl,lk->ij', D1q_list, Gamma)

def get_F1q0(Gamma):
    ''' Returns new displacement operator at the origin,
        F_0 = 1/DIM \sum_{p,q} 1/tr[D_{p,q} Gamma] D_{p,q}.
        Output - (DIM,DIM) complex ndarray
    '''
    traces = 1./get_trace_D(Gamma)
    return 1./DIM * np.einsum('ij,ijkl->kl', traces, D1q_list)


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

X = np.array([[0,1],[1,0]],dtype='complex_')
Z = np.array([[1,0],[0,-1]],dtype='complex_')
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
    # print('Done calculating D1qs.')
    return D1q_list
D1q_list = allD1qs()

