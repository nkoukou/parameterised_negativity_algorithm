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
# from qubit_circuit_generator import(show_connectivity, qiskit_simulate,
#                                     haar_random_connected_circuit)
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
    prob      = qiskit_simulate(circuit)
    circ      = deepcopy(circuit)
    circ_comp = compress_circuit(circuit, n)

    x_in      = init_x_list(circ, x0)
    x_comp    = init_x_list(circ_comp, x0)
    x_opt, _  = sequential_para_opt(W, circ_comp, x_comp, l, niter=1)

    label     = ['Comp: NO || Opt: NO',
                 'Comp: YES || Opt: NO',
                 'Comp: YES || Opt: YES']
    circuits  = [circ, circ_comp, circ_comp]
    x         = [x_in, x_comp, x_opt]
    samples   = [prob]
    for i in range(3):
        # print(label[i])
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

def plot(samples):
    ''' samples - 2d numpy array, output of function sample
    '''
    def dev(i):
        return np.abs(samples[:,0] - samples[:,i])

    fig, ax = plt.subplots(1,1)

    labels = ['Wigner, uncompressed', 'Wigner, compressed',
              'Optimised, compressed']
    colours = [(0.1, 0.9, 0., 0.2), (0.2, 0., 0.8, 0.7), (0.8, 0., 0.2, 0.5)]

    ymax = 0.
    width = 0.1
    for i in range(3):
        j = int((i-1)%3)+1
        data = dev(j)
        bins = int((data.max() - data.min())/width)
        freq, bnds, _ = ax.hist(data, bins=bins, fc=colours[i],
                          edgecolor="none", label=labels[i])
        if freq.max() > ymax: ymax = freq.max()

    ax.set_ylim(0., 1.05*ymax)
    ax.set_xlim(-0.05, 4.)
    ax.set_xlabel(r'$|p_{\rm{est}} - p_{\rm{sim}}|$')
    ax.legend(loc='upper right').set_draggable(1)

circuit_num = 1000
sample_size = int(1e6)
N, n, L, l = 3, 2, 8, 1

fname = os.path.join("data_sampling","samples_N%d_n%d_L%d_l%d.npy"%(N,n,L,l))
samples = np.load(fname) if os.path.isfile(fname) else np.zeros(4)

# for i in range(circuit_num):
#     print(samples.shape[0])
#     circuit = haar_random_connected_circuit(N, L, n,
#                                     given_state=0, given_meas=1, method='r')
#     temp = sample(circuit, n, l, sample_size)
#     samples = np.vstack((samples, temp))
#     np.save(fname, samples)

plt.close('all')
plot(samples)










# phi = 0.7 * np.pi
# circuit = {'state_list': [makeState('0') for i in range(2)],
#                 'gate_list': [makeGate('H'), makeGate('C+')],
#                 'index_list': [[0], [0,1]],
#                 'meas_list': [makeState('1'),makeState('1')]}
# print(np.cos(phi)**2)