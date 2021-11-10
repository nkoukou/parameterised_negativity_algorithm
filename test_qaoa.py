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
    return expm(-1.j*gamma*ZZ)
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


edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
G = Graph(4, edges)

beta_list = [1.9793337]
gamma_list = [1.16663483]

test = qaoa_maxcut(G, beta_list, gamma_list)
x0 = ps_Wigner.x0
W = ps_Wigner.W

print("STEP 0")
circuit1 = test.copy()
show_connectivity(circuit1)
circuit1_compressed = compress_circuit(circuit1, n=2)
print("STEP 1")
x_circuit1 = init_x_list(circuit1_compressed, x0)
x_out1, neg_list_seq1 = sequential_para_opt(W, circuit1_compressed,
                                            x_circuit1, l=1, niter=1)
print("STEP 2")
output = get_qd_output(circuit1_compressed, x_out1, ps_Wigner)
p_est, expectation_qaoa = sample_circuit(circuit1_compressed, output,
                                          sample_size = int(1e5))


### TEST SAMPLING ###
# from qubit_circuit_components import(makeState, makeGate)

# phi = 0.6 * np.pi
# test_circuit = {'state_list': [makeState('0') for i in range(3)],
#                 'gate_list': [U_mix(phi), makeGate('C+'), X, makeGate('H')],
#                 'index_list': [[0], [0,1], [1], [2]],
#                 'meas_list': [makeState('0'), makeState('1'), makeState('+')]}

# test_circuit = compress_circuit(test_circuit, n=3)
# x_circuit1 = init_x_list(test_circuit, x0)
# x_out1, neg_list_seq1 = sequential_para_opt(W, test_circuit,
#                                             x_circuit1, l=1, niter=1)
# output = get_qd_output(test_circuit, x_out1, ps_Wigner)
# p_est, expectation_qaoa = sample_circuit(test_circuit, output,
#                                           sample_size = int(1e5))
# print(np.cos(phi)**2)























