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

beta_list = [0.1]
gamma_list = [0.05]

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
                                 sample_size = 100000)

# circuit2 = test.copy()
# # circuit2 = compress_circuit(circuit2, n=2)
# # show_connectivity(circuit2)
# x_circuit2 = init_x_list(circuit2, x0)
# x_out2, neg_list_seq2 = sequential_para_opt(W, circuit2, x_circuit2, l=10,
#                                             niter=3)
























