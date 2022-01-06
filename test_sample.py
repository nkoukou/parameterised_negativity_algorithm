import time
from copy import deepcopy
import numpy as np
import matplotlib.pylab as plt

from compression import(compress_circuit)
from frame_opt import(init_x_list, get_negativity_circuit, sequential_para_opt)
from phase_space import(PhaseSpace)
from prob_sample import(prepare_sampler, sample_fast)
from qubit_circuit_components import(makeState, makeGate)
from qubit_circuit_generator import(show_connectivity,
                                    haar_random_connected_circuit)
from qubit_frame_Wigner import(F, G, DIM, x0)
# from qubit_frame_Pauli import(F, G, DIM, x0)

ps_Wigner = PhaseSpace(F, G, x0, DIM)
x0 = ps_Wigner.x0
W = ps_Wigner.W

def sample(circuit, n, l, sample_size):
    circ = deepcopy(circuit)
    x_in = init_x_list(circ, x0)
    circ_comp = compress_circuit(circuit, n)
    x_comp = init_x_list(circ_comp, x0)
    x_opt, _ = sequential_para_opt(W, circ_comp, x_comp, l, niter=1)

    label = ['Comp: NO || Opt: NO',
             'Comp: YES || Opt: NO',
             'Comp: YES || Opt: YES']
    circuits = [circ, circ_comp, circ_comp]
    x = [x_in, x_comp, x_opt]
    samples = []
    for i in range(3):
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
    samples = np.vstack(samples)
    return samples

def plot(samples):
    ''' samples - 2d numpy array, output of function sample
    '''
    fig, ax = plt.subplots(1,1)
    xaxis = np.arange(samples.shape[1])
    ax.plot(xaxis, samples[0], label='Not compressed, Wig repr')
    ax.plot(xaxis, samples[1], label='Compressed, Wig repr')
    ax.plot(xaxis, samples[2], label='Compressed, Opt repr')

    ax.legend(loc='upper right')


circuit = haar_random_connected_circuit(N=3, L=20, n=2,
            given_state=0, given_meas=1, method='r')
n = 2
l = 1
sample_size = int(1e6)

samples = sample(circuit, n, l, sample_size)
plot(samples)











# phi = 0.7 * np.pi
# circuit = {'state_list': [makeState('0') for i in range(2)],
#                 'gate_list': [makeGate('H'), makeGate('C+')],
#                 'index_list': [[0], [0,1]],
#                 'meas_list': [makeState('1'),makeState('1')]}
# print(np.cos(phi)**2)