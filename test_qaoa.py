import numpy as np
from scipy.linalg import(expm)

from qubit_circuit_generator import(show_connectivity)
from compression import(compress_circuit)
from frame_opt import(init_x_list, get_negativity_circuit, sequential_para_opt)
from phase_space import(PhaseSpace)
from prob_sample import(get_qd_output, sample_circuit)
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
        output = get_qd_output(circuit, x_out, ps_Wigner)
        exp_qaoa = sample_circuit(circuit, output, sample_size = int(5e6))

        temp = np.prod(np.array(output["neg_list_states"]))
        for neg_gate in output["neg_list_gates"]:
            temp *= neg_gate.max()
        temp *= np.prod(np.array(output["neg_list_meas"]))
        neg_circ[b,g] = np.log2(temp)
        p_est[b, g] = np.average(exp_qaoa)

        np.save("neg_circ.npy", neg_circ)
        np.save("p_est.npy", p_est)




### TEST SAMPLING ###
# from qubit_circuit_components import(makeState, makeGate)

# phi = 0.8 * np.pi
# test_circuit = {'state_list': [makeState('0') for i in range(3)],
#                 'gate_list': [U_mix(phi), makeGate('C+'), X, makeGate('H')],
#                 'index_list': [[0], [0,1], [1], [2]],
#                 'meas_list': [makeState('0'), makeState('1'), makeState('+')]}

# test_circuit = compress_circuit(test_circuit, n=3)
# x_circuit1 = init_x_list(test_circuit, x0)
# x_out1, neg_list_seq1 = sequential_para_opt(W, test_circuit,
#                                             x_circuit1, l=1, niter=1)
# output = get_qd_output(test_circuit, x_out1, ps_Wigner)
# p_est = sample_circuit(test_circuit, output,
#                                           sample_size = int(5e5))
# print(np.cos(phi)**2)























