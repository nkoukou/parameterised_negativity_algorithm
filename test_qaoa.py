import time
from multiprocessing import Pool
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
from prob_sample import(prepare_sampler, Sampler)
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

    meas_list = G.N*[Z] # 0.5*ZZ

    circuit = {'state_list': state_list, 'gate_list': gate_list,
               'index_list': index_list, 'meas_list': meas_list}
    return circuit

sample_size = int(1e8)
x0 = ps_Wigner.x0
W = ps_Wigner.W

edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
G = Graph(4, edges)

betas  = np.linspace(0., 0.5, 4) * np.pi/4 # np.array([1.9793337]) #
gammas = np.linspace(0., 1., 6) * np.pi/4 # np.array([1.16663483]) #

neg_circ = np.zeros((betas.size, gammas.size))
p_est = np.zeros((betas.size, gammas.size))
for b in range(betas.size):
    for g in range(gammas.size):
        print("\n"+"(beta, gamma) = (%.2f, %.2f)"%(betas[b], gammas[g])+"\n")

        circuit = qaoa_maxcut(G, [betas[b]], [gammas[g]])
        circuit = compress_circuit(circuit, n=2)
        x_circ  = init_x_list(circuit, x0)
        x_out, neg_list = sequential_para_opt(W, circuit, x_circ,
                                              l=1, niter=1)
        meas_list, index_list, qd_list_states, qd_list_gates, qd_list_meas,\
        pd_list_states, pd_list_gates, pd_list_meas, sign_list_states,\
        sign_list_gates, sign_list_meas, neg_list_states, neg_list_gates,\
        neg_list_meas = prepare_sampler(circuit=circuit,
                                        par_list=x_out, ps=ps_Wigner)

        qaoa_sampler = Sampler(sample_size, meas_list, index_list,
          qd_list_states, qd_list_gates, qd_list_meas, pd_list_states,
          pd_list_gates, pd_list_meas, sign_list_states, sign_list_gates,
          sign_list_meas, neg_list_states, neg_list_gates, neg_list_meas)
        qaoa_sampler.sample()

        p_est[b, g] = qaoa_sampler.p_estimate

        neg = np.prod(neg_list_states)
        for neg_gate in neg_list_gates:
            neg *= neg_gate.max()
        neg *= np.prod(neg_list_meas)
        neg_circ[b,g] = np.log2(neg)

        np.save("neg_circ.npy", neg_circ)
        np.save("p_est.npy", p_est)

p_exact = np.array([[0., 0., 0., 0., 0., 0.],
                    [0., 0.293926, 0.475554, 0.475553, 0.293923, 0.],
                    [0., 0.50909, 0.823682, 0.823682, 0.50909, 0.],
                    [0., 0.587845, 0.951107, 0.951107, 0.587845, 0.]])

p_est = np.load("p_est.npy")

neg = np.load("neg_circ.npy")

import matplotlib.pylab as plt
plt.close()

plt.figure()
plt.plot(neg.flatten(), (p_est - p_exact).flatten(), 'o')
plt.xlabel("Negativity")
plt.ylabel(r'$\hat{p} - p$')




### TEST SAMPLING ###
# from qubit_circuit_components import(makeState, makeGate)

# phi = 0.7 * np.pi
# circuit = {'state_list': [makeState('0') for i in range(3)],
#                 'gate_list': [U_mix(phi), makeGate('C+'), X, makeGate('H')],
#                 'index_list': [[0], [0,1], [1], [2]],
#                 'meas_list': [makeState('0'),makeState('1'),makeState('+')]}

# circuit = compress_circuit(circuit, n=2)
# x_circuit = init_x_list(circuit, x0)
# x_out, neg_list_seq = sequential_para_opt(W, circuit,
#                                             x_circuit, l=1, niter=1)

# meas_list, index_list, qd_list_states, qd_list_gates, qd_list_meas,\
# pd_list_states, pd_list_gates, pd_list_meas, sign_list_states,\
# sign_list_gates, sign_list_meas, neg_list_states, neg_list_gates,\
# neg_list_meas = prepare_sampler(circuit=circuit, par_list=x_out,ps=ps_Wigner)

# sample_size = int(1e8)
# test = Sampler(sample_size, meas_list, index_list, qd_list_states,
#                 qd_list_gates, qd_list_meas, pd_list_states, pd_list_gates,
#                 pd_list_meas, sign_list_states, sign_list_gates,
#                 sign_list_meas, neg_list_states, neg_list_gates,
#                 neg_list_meas)

# print(np.cos(phi)**2)
# test.sample()
# print(test.p_estimate)


### TESTING

# output = get_qd_output(circuit, x_out, ps_Wigner)

# p_est = sample_circuit(circuit, output, sample_size=int(1e5))
# print(np.cos(phi)**2)

# prob = np.array([0.6, 0.2, 0.1, 0.1])
# N = int(1e5)
# ps_point1 = np.zeros(N)
# ps_point2 = np.zeros(N)
# for i in range(N):
#     if i%(N//10)==0: print(i/N*100, "%")
#     ps_point1[i] = np.arange(len(prob))[np.searchsorted(
#               np.cumsum(prob), np.random.random(), side="right")]
#     ps_point2[i] = nr.choice(np.arange(len(prob)), p=prob)

# import matplotlib.pylab as plt

# plt.hist(ps_point1)
# plt.hist(ps_point2, rwidth=0.5)


# i=1
# test_pd = output["pd_list_gates"][i]
# test_sg = output["sign_list_gates"][i]
# test_ng = output["neg_list_gates"][i]

# current_ps_point = [1,0,1,1]
# idx = [0,1]
# pq_in1 = [[current_ps_point[idx[i]]//2, current_ps_point[idx[i]]%2]
#           for i in range(len(idx))]
# pq_in1 = tuple([item for sublist in pq_in1 for item in sublist])
# prob1 = test_pd[pq_in1].flatten()
# sign1 = test_sg[pq_in1].flatten()
# neg1 = test_ng[pq_in1]

# arr_dim = np.log2(len(test_pd.flatten()))/2
# pq_in2 = 0
# for i in range(len(idx)):
#     pq_in2 += current_ps_point[idx[i]]//2 * 2**(2*(arr_dim-i)-1)
#     pq_in2 += current_ps_point[idx[i]]%2 * 2**(2*(arr_dim-i)-2)
# pq_in2 = int(pq_in2)
# prob2 = test_pd.flatten()[pq_in2:pq_in2+int(2**arr_dim)]
# sign2 = test_sg.flatten()[pq_in2:pq_in2+int(2**arr_dim)]
# neg2 = test_ng.flatten()[int(pq_in2//(2**arr_dim))]

# print(test_pd.flatten().shape)
# print(test_pd.shape, test_sg.shape)

# print("TEST:", np.allclose(prob1, prob2))
# print("TEST:", np.allclose(sign1, sign2))
# print("TEST:", np.allclose(neg1, neg2))

# ps_point = nr.choice(np.arange(len(prob1)), p=prob1)
# str_repr = np.base_repr(ps_point, 4).zfill(len(idx))
# print(ps_point, str_repr)
# print(ps_point//4**(arr_dim-1-0))
# print(ps_point//4**(arr_dim-1-1))
# for i in range(len(idx)):
#     p1 = int(ps_point//4**(np.log2(len(prob2))/2 -1-i))%4
#     p2 = int(str_repr[i])
#     print(p1,p2)




# pool = multiprocessing.processing_pool()
# def f(n):
#     print(n/sample_size*100, "%")
# p = Pool(8)
# p.map(f, range(sample_size))




















