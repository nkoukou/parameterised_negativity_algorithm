import time
# from multiprocessing import Pool
# from pathos import multiprocessing
# import os
# os.environ['NUMBA_DISABLE_INTEL_SVML']  = '1'

import numpy as np
import numpy.random as nr
from scipy.linalg import(expm)

from qubit_circuit_generator import(show_connectivity)
from compression import(compress_circuit)
from frame_opt import(init_x_list, get_negativity_circuit, sequential_para_opt)
from phase_space import(PhaseSpace)
from prob_sample import(prepare_sampler, sample_fast)
from qubit_frame_Wigner import(F, G, DIM, x0)

ps_Wigner = PhaseSpace(F, G, x0, DIM)

plus = 0.5*np.array([[1.,1],[1.,1]])
Z    = np.array([[1,0],[0,-1]])
ZZ   = np.kron(Z,Z)
X    = np.array([[0,1],[1,0]])

def U_prob(gamma):
    return expm(-1.j*2*gamma*ZZ)
def U_mix(beta):
    return expm(-1.j*beta*X)

class Graph():
    def __init__(self, N, edges):
        self.N = N
        self.nodes = np.arange(N)
        self.edges = [list(edge) for edge in edges]

def qaoa_maxcut(G, beta_list, gamma_list):
    state_list = G.N*[plus]

    gate_list = []
    index_list = []
    for i, beta in enumerate(beta_list):
        gamma = gamma_list[i]
        UP = U_prob(gamma)
        for edge in G.edges:
            gate_list.append(UP)
            index_list.append(edge)
        UM = U_mix(beta)
        for node in range(G.N):
            gate_list.append(UM)
            index_list.append([node])

    meas_list = G.N*[Z]

    circuit = {'state_list': state_list, 'gate_list': gate_list,
               'index_list': index_list, 'meas_list': meas_list}
    return circuit



sample_size = int(1e5)
x0 = ps_Wigner.x0
W = ps_Wigner.W

edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
G = Graph(4, edges)

### Get QAOA samples

# betas  = np.linspace(0., 0.5, 4)[1:2] * np.pi/4
# gammas = np.linspace(0., 1., 6)[1:2] * np.pi/4

# neg_circ = np.zeros((betas.size, gammas.size))
# p_est = np.zeros((betas.size, gammas.size))
# neg_circ2 = np.zeros((betas.size, gammas.size))
# p_est2 = np.zeros((betas.size, gammas.size))
# for b in range(betas.size):
#     for g in range(gammas.size):
#         print("\n"+"(beta, gamma) = (%.2f, %.2f)"%(betas[b], gammas[g])+"\n")

#         circuit = qaoa_maxcut(G, [betas[b]], [gammas[g]])
#         circuit = compress_circuit(circuit, n=2)
#         x_circ  = init_x_list(circuit, x0)
#         x_out, neg_list = sequential_para_opt(W, circuit, x_circ,
#                                               l=1, niter=1)
#         meas_list, index_list, qd_list_states, qd_list_gates, qd_list_meas,\
#         pd_list_states, pd_list_gates, pd_list_meas, sign_list_states,\
#         sign_list_gates, sign_list_meas, neg_list_states, neg_list_gates,\
#         neg_list_meas = prepare_sampler(circuit=circuit,
#                                         par_list=x_out, ps=ps_Wigner)

#         print("\nSTEP 0\n")
#         test = sample_fast(sample_size, meas_list, index_list, qd_list_states,
#                 qd_list_gates, qd_list_meas, pd_list_states, pd_list_gates,
#                 pd_list_meas, sign_list_states, sign_list_gates,
#                 sign_list_meas, neg_list_states, neg_list_gates,
#                 neg_list_meas)

#         print(test)

#         # p_est[b, g] = qaoa_sampler.p_estimate

#         neg = np.prod(neg_list_states)
#         for neg_gate in neg_list_gates:
#             neg *= neg_gate.max()
#         neg *= np.prod(neg_list_meas)
#         neg_circ[b,g] = np.log2(neg)

#         np.save("neg_circ.npy", neg_circ)
#         np.save("p_est.npy", p_est)

# p_exact = np.array([[0., 0., 0., 0., 0., 0.],
#                     [0., 0.293926, 0.475554, 0.475553, 0.293923, 0.],
#                     [0., 0.50909, 0.823682, 0.823682, 0.50909, 0.],
#                     [0., 0.587845, 0.951107, 0.951107, 0.587845, 0.]])

# p_est = np.load("p_est.npy")

# neg = np.load("neg_circ.npy")

# import matplotlib.pylab as plt
# plt.close()

# plt.figure()
# plt.plot(neg.flatten(), (p_est - p_exact).flatten(), 'o')
# plt.xlabel("Negativity")
# plt.ylabel(r'$\hat{p} - p$')






### TEST SAMPLING ###
from qubit_circuit_components import(makeState, makeGate)

phi = 0.6 * np.pi
circuit = {'state_list': [makeState('0') for i in range(3)],
                'gate_list': [U_mix(phi), makeGate('C+'), X, makeGate('H')],
                'index_list': [[0], [0,1], [1], [2]],
                'meas_list': [makeState('0'),makeState('1'),makeState('+')]}

circuit = compress_circuit(circuit, n=2)
x_circuit = init_x_list(circuit, x0)
x_out, neg_list_seq = sequential_para_opt(W, circuit,
                                            x_circuit, l=1, niter=1)

meas_list, index_list, qd_list_states, qd_list_gates, qd_list_meas,\
pd_list_states, pd_list_gates, pd_list_meas, sign_list_states,\
sign_list_gates, sign_list_meas, neg_list_states, neg_list_gates,\
neg_list_meas = prepare_sampler(circuit=circuit, par_list=x_out,ps=ps_Wigner)

test = sample_fast(sample_size, meas_list, index_list, qd_list_states,
                qd_list_gates, qd_list_meas, pd_list_states, pd_list_gates,
                pd_list_meas, sign_list_states, sign_list_gates,
                sign_list_meas, neg_list_states, neg_list_gates,
                neg_list_meas)

print(test)
print(np.cos(phi)**2)









### TEST PARALLELISM

# from joblib import Parallel, delayed
# import multiprocessing
# import random

# def get_tally(n):
#     tally=0
#     for i in range(n):
#         x = random.random()
#         y = random.random()
#         if x ** 2 + y ** 2 < 1:
#             tally = tally + 1
#     return(tally)

# def calculate_pi(n):
#     tally = get_tally(n)
#     return(4 * tally / n)

# def calculate_pi_parallel(n):
#     n_core = multiprocessing.cpu_count()
#     tallies = Parallel(n_jobs=n_core)(delayed(get_tally)(n//n_core)
#                                       for i in range(n_core))
#     return(4 * sum(tallies) / n)

# N = int(1e8)

# print("STEP 0")
# t0 = time.time()
# res1 = calculate_pi(N)
# t1 = time.time() - t0
# print("STEP 1")
# t0 = time.time()
# res2 = calculate_pi_parallel(N)
# t2 = time.time() - t0
# print("STEP 2")





















