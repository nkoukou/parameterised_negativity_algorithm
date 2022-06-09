import time
import os
from copy import deepcopy
import numpy as np
import matplotlib.pylab as plt

from compression import(compress_circuit)
from frame_opt import(init_x_list, get_negativity_circuit, sequential_para_opt)
from phase_space import(PhaseSpace)
from prob_sample import(prepare_sampler, sample_fast)
from qubit_circuit_components import(makeState, makeGate)
from qubit_circuit_generator import(qiskit_simulate, show_connectivity,
                                    haar_random_connected_circuit)
from qubit_frame_Wigner import(F, G, DIM, x0)
# from qubit_frame_Pauli import(F, G, DIM, x0)


plt.rcParams['figure.dpi'] = 200
plt.style.use('classic')
plt.rc('font',   size=24)
plt.rc('axes',   labelsize=25)
plt.rc('xtick',  labelsize=21)
plt.rc('ytick',  labelsize=21)
plt.rc('legend',  fontsize=23)
plt.rc('lines',  linewidth=2 )
plt.rc('lines', markersize=5 )

ps_Wigner = PhaseSpace(F, G, x0, DIM)
x0 = ps_Wigner.x0
W = ps_Wigner.W

def sample(circuit, n, l, sample_size):
    circ      = deepcopy(circuit)
    circ_comp = compress_circuit(circuit, n)
    prob      = qiskit_simulate(circ_comp)

    x_in      = init_x_list(circ, x0)
    x_comp    = init_x_list(circ_comp, x0)
    print("---------------------")
    print("Calculating x_opt...")
    x_opt, _  = sequential_para_opt(W, circ_comp, x_comp, l, niter=1)

    label     = ['Comp: NO || Opt: NO',
                 'Comp: YES || Opt: NO',
                 'Comp: YES || Opt: YES']
    circuits  = [circ, circ_comp, circ_comp]
    x         = [x_in, x_comp, x_opt]
    samples   = [prob]
    for i in range(3):
        print("---------------------")
        print(label[i])
        meas_list, index_list, qd_list_states, qd_list_gates, qd_list_meas,\
        pd_list_states, pd_list_gates, pd_list_meas, sign_list_states,\
        sign_list_gates, sign_list_meas, neg_list_states, neg_list_gates,\
        neg_list_meas = prepare_sampler(circuit=circuits[i], par_list=x[i],
                                        ps=ps_Wigner)

        estimate  = sample_fast(sample_size, meas_list, index_list,
                    qd_list_states, qd_list_gates, qd_list_meas,
                    pd_list_states, pd_list_gates, pd_list_meas,
                    sign_list_states, sign_list_gates, sign_list_meas,
                    neg_list_states, neg_list_gates, neg_list_meas)
        samples.append(estimate)
    return np.array(samples)

def generate_data(N, n, L, l, S, circ_num):
    fname = os.path.join("data_sampling",
                         "samples_N%d_n%d_L%d_l%d.npy"%(N,n,L,l))
    trig = 0.
    data = np.zeros((circ_num, 4))
    if os.path.isfile(fname):
        data = np.load(fname)
        trig = np.where(np.load(fname)[:,-1]!=0.)[0].size

    for c in range(circ_num):
        if c < trig: continue
        print("=========================================================")
        print("(N, n, L, l) = (%d, %d, %d, %d) | %d / %d"%(N,n,L,l,c+1,
                                                           circ_num))
        circuit = haar_random_connected_circuit(N, L, n,
                                    given_state=0, given_meas=1, method='r')
        data[c] = sample(circuit, n, l, S)
        np.save(fname, data)
    return np.array(data)

def plot(samples):
    ''' samples - 2d numpy array, output of function sample
    '''
    def dev(i):
        return np.abs(samples[:,0] - samples[:,i])

    fig, ax = plt.subplots(1,1)

    labels = ['Wigner, uncompressed', 'Wigner, compressed',
              'Optimised, compressed']
    colours = [(0.3, 0.7, 0., 0.3), (0.2, 0., 0.8, 0.7), (0.8, 0., 0.2, 0.5)]

    ymax = 0.
    width = 0.1
    for i in [0,1,2]:
        data = dev(i+1)
        bins = int((data.max() - data.min())/width)
        freq, bnds, _ = ax.hist(data, bins=bins, fc=colours[i],
                          edgecolor="none", label=labels[i])
        if freq.max() > ymax: ymax = freq.max()

    ax.set_ylim(0., 1.05*ymax)
    ax.set_xlim(-0.01, 4.)
    ax.set_xlabel(r'$|p_{\rm{est}} - p_{\rm{sim}}|$')
    ax.legend(loc='upper right').set_draggable(1)

S, circ_num = int(1e7), 1000
N, n, L, l = 3, 2, 8, 1

# samples = generate_data(N, n, L, l, S, circ_num)
# print(samples.shape)

fname = os.path.join("data_sampling",
                      "samples_N%d_n%d_L%d_l%d.npy"%(N,n,L,l))
samples = np.load(fname)
print(samples


.shape)
plt.close('all')
plot(samples)


### TEST SAMPLING WORKS
# from scipy.linalg import(expm)
# N = 6
# L = 20
# n = 2

# phi = 0.8 * np.pi
# p_analytical = np.sin(phi)**2

# phasegate = np.kron(expm(-1.j*phi*np.array([[0,1],[1,0]])), makeGate('1'))
# circuit = {'state_list': [makeState('0') for i in range(2)],
#                 'gate_list': [phasegate, makeGate('C+')],
#                 'index_list': [[0,1], [0,1]],
#                 'meas_list': [makeState('1'),makeState('1')]}
# p_sim = qiskit_simulate(circuit)


# circ_comp = compress_circuit(circuit, n)
# x_comp    = init_x_list(circ_comp, x0)
# x_opt, _  = sequential_para_opt(W, circ_comp, x_comp, l, niter=1)
# meas_list, index_list, qd_list_states, qd_list_gates, qd_list_meas,\
# pd_list_states, pd_list_gates, pd_list_meas, sign_list_states,\
# sign_list_gates, sign_list_meas, neg_list_states, neg_list_gates,\
# neg_list_meas = prepare_sampler(circuit=circ_comp, par_list=x_opt,
#                                 ps=ps_Wigner)

# p_est  = sample_fast(S, meas_list, index_list,
#             qd_list_states, qd_list_gates, qd_list_meas,
#             pd_list_states, pd_list_gates, pd_list_meas,
#             sign_list_states, sign_list_gates, sign_list_meas,
#             neg_list_states, neg_list_gates, neg_list_meas)

# print(p_analytical, p_sim, p_est)






