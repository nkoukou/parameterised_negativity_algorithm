# import numpy as np
import autograd.numpy as np
from autograd import(grad)
import itertools as it
from scipy.optimize import(minimize, basinhopping)
from QUBIT_state_functions import(psi2rho)

ksi   = np.exp(np.pi*1.j/4)
alpha = np.pi/8
beta  = np.arccos(1/np.sqrt(3))/2

Hs = psi2rho(np.array([np.cos(alpha), np.sin(alpha)]))
Ts = psi2rho(np.array([np.cos(beta), ksi*np.sin(beta)]))
Z0 = psi2rho([1,0])
Z1 = psi2rho([0,1])
X0 = psi2rho(1/np.sqrt(2)*np.array([1,1]))
X1 = psi2rho(1/np.sqrt(2)*np.array([1,-1]))

I = np.eye(2)
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1.j],[1.j,0]])
Z = np.array([[1,0],[0,-1]])
P = np.stack((I,X,Y,Z))

H = 1/np.sqrt(2)*np.array([[1, 1],[1, -1]])
K = np.diag([1, 1j])
T = np.diag([1, ksi])
CX= np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])

def R(x,y,z):
    ca = np.cos(z)
    sa = np.sin(z)
    cb = np.cos(y)
    sb = np.sin(y)
    cc = np.cos(x)
    sc = np.sin(x)

    rot = np.array(
    [[1,     0,              0,              0],
     [0, ca*cb, ca*sb*sc-sa*cc, ca*sb*cc+sa*sc],
     [0, sa*cb, sa*sb*sc+ca*cc, sa*sb*cc-ca*sc],
     [0,   -sb, cb*sc,          cb*cc         ]
    ]
    )
    return rot

def A_s1q(rho):
    A = 1./2 * np.einsum('ij,kji->k', rho, P).real
    return A

def A_g1q(U):
    P1 = np.einsum('kl,ilm->ikm',U, P)
    P2 = np.einsum('nm,jnk->jmk',U.conj(), P)
    A = 1./2 * np.einsum('ikm,jmk->ji', P1, P2).real
    return A

# def A_g2q(U):
#     U = np.transpose(U.reshape((2,2,2,2)), axes=(0,2,1,3))
#     PP = np.einsum('xyz,XYZ',P, P)
#     P1 = np.einsum('ijIJ,ajcAJC->iIacAC',U, PP)
#     P2 = np.einsum('JCjc,bjiBJI->cCbiBI',U.conj(), PP)
#     A = 1./4 * np.einsum('iIacAC,cCbiBI->baBA', P1, P2).real
#     A = np.transpose(A, axes=(0,2,1,3)).reshape((16,16))
#     return A

def A_g2q(U):
    T_out = np.zeros((16,16))
    for w in it.product(range(4),repeat=4):
        [w1_in, w2_in, w1_out, w2_out] = w
        Pauli_in = np.kron(P[w1_in],P[w2_in])
        Pauli_out = np.kron(P[w1_out],P[w2_out])
        T_out[4*w1_out+w2_out][4*w1_in+w2_in] = np.real(1./4.*np.trace(
            np.dot(np.dot(np.dot(U,Pauli_in),U.conj().T),Pauli_out)))
    return T_out

def A_m1q(meas):
    A = 1./2 * np.einsum('ij,kji->k', meas, P).real
    return A

def RA_s1q(rho, theta):
    Rt = R(*theta)
    A = np.einsum('ij,j->i', Rt, A_s1q(rho))
    # A = np.einsum('i,ki->k', A, Rt)
    return A

def RAR_g1q(U, theta):
    R_inp = R(*theta[:3])
    R_out = R(*theta[3:])
    A = np.einsum('ij,jk->ik', R_out, A_g1q(U))
    A = np.einsum('ij,kj->ik', A, R_inp)
    return A

def RAR_g2q(U, theta):
    R_inp = np.kron(R(*theta[:3]), R(*theta[3:6]))
    R_out = np.kron(R(*theta[6:9]), R(*theta[9:]))
    A = np.einsum('ij,jk->ik', R_inp, A_g2q(U))
    A = np.einsum('ij,kj->ik', A, R_out)
    return A

def RA_m1q(meas, theta):
    Rt = R(*theta)
    A = np.einsum('ij,j->i', Rt, A_m1q(meas))
    # A = np.einsum('i,ki->k', A, Rt)
    return A

def negativity(M):
    return np.abs(M).sum(axis=0)

def max_negativity(M):
    return negativity(M).max()

def neg_circuit(theta, circuit):
    neg = 1.
    for i, state in enumerate(circuit['state_list']):
        neg *= negativity(RA_s1q(state, theta[3*i:3*(i+1)]))
    N = 3*len(circuit['state_list'])
    for j, gate in enumerate(circuit['gate_list']):
        # idx = circuit['index_list'][j]
        theta_in = theta[N+3*(j-1):N+3*j]
        theta_out = theta[N+3*j:N+3*(j+1)]
        theta_comp = np.concatenate((theta_in, theta_out))
        neg *= max_negativity(RAR_g1q(gate, theta_comp))
    neg *= negativity(RA_m1q(meas, theta[-3:]))
    return neg

def optimise_circuit(circuit):
    num = len(circuit['state_list'])+ \
          sum([len(i) for i in circuit['index_list']])

    def cost_function(theta):
        return neg_circuit(theta, circuit)
    grad_cost_function = grad(cost_function)
    def func(x):
        return cost_function(x), grad_cost_function(x)

    # theta0 = np.array([0. for i in range(3*num)])
    # theta0 = np.array([0.,0.,np.arccos(1/np.sqrt(2)),0.,0.,0.])
    theta0 = np.array([0.,0.,0.,0.,0.,np.arccos(1/np.sqrt(2))])

    res = basinhopping(func, theta0,
           minimizer_kwargs={"method":"L-BFGS-B","jac":True},niter=10)
    # res = minimize(cost_function, theta0, method='Powell')
    # if not res.success: raise Exception('Optimisation failed')
    return res #res.fun, res.x





from QUBIT_circuit_components import(makeState, makeGate)

state = makeState('+')
gates = [makeGate('T')]
meas = makeState('0')

circuit = {'state_list': [state],
           'gate_list': gates,
           'index_list': [[0] for i in range(len(gates))],
           'meas_list': [meas]
           }

num = len(circuit['state_list'])+ \
  sum([len(i) for i in circuit['index_list']])
theta0 = np.array([0. for i in range(3*num)])
neg = neg_circuit(theta0, circuit)

res = optimise_circuit(circuit)
print(res)

A_state = RA_s1q(state, res.x[:3])
A_gates = [RAR_g1q(gate, np.concatenate((res.x[:3],res.x[3:6])))
            for gate in gates]
A_meas = RA_m1q(meas, res.x[-3:])

neg_state = negativity(A_state)
neg_gate = [negativity(A_gate) for A_gate in A_gates]
neg_meas = negativity(A_meas)

output = state
for gate in gates:
    output = np.dot(gate, np.dot(output, gate.T.conj()))
output = np.trace(np.dot(meas, output))





'''
def rotate_s1q(inp, out):
    def cost_function(theta, inp, out):
        A = RA_s1q(inp, theta)
        return np.sum(np.abs(A - A_s1q(out)))
    theta0 = (0,0,0)
    res = minimize(cost_function, theta0, args=(inp, out,),
                   method='L-BFGS-B')
    return res

def rotate_g1q(inp, out):
    def cost_function(theta, inp, out):
        A = RAR_g1q(inp, theta)
        return np.sum(np.abs(A - A_g1q(out)))
    theta0 = (0,0,0)*2
    res = minimize(cost_function, theta0, args=(inp, out,),
                   method='L-BFGS-B')
    return res

def rotate_g2q(inp, out):
    def cost_function(theta, inp, out):
        A = RAR_g1q(inp, theta)
        return np.sum(np.abs(A - A_g2q(out)))
    theta0 = (0,0,0)*4
    res = minimize(cost_function, theta0, args=(inp, out,),
                   method='L-BFGS-B')
    return res
'''