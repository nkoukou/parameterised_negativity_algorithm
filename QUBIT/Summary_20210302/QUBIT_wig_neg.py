import os
import time
import autograd.numpy as np
from autograd import(grad)

from scipy.optimize import(Bounds, minimize)
from scipy.optimize import(basinhopping)
from QUBIT_phase_space import(x2Gamma, neg_state_1q, neg_gate_1q_max,
                              neg_gate_2q_max, neg_meas_1q)

def wigner_neg_compressed(compressed_circuit, **kwargs):
    options = {'Wigner Negativity Options'}
    options.update(kwargs)

    x_len = int(len(compressed_circuit['state_list']) + 2*len(
                    compressed_circuit['gate_list']))
    x0w = [1,1/2,1/2]

    start_time = time.time()
    neg = 1.
    Gamma_index = 0
    Gammas = []
    for state in compressed_circuit['state_list']:
        Gamma = x2Gamma(x0w)
        neg = neg * neg_state_1q(state, Gamma)
        Gammas.append(Gamma)
        Gamma_index += 1

    for gate in compressed_circuit['gate_list']:
        GammaC_in = x2Gamma(x0w)
        GammaT_in = x2Gamma(x0w)
        GammaC_out = x2Gamma(x0w)
        GammaT_out = x2Gamma(x0w)
        neg = neg * neg_gate_2q_max(gate, GammaC_in, GammaT_in,
                                          GammaC_out, GammaT_out)

    for meas in compressed_circuit['meas_list']:
        if str(meas) == '/': continue
        Gamma = x2Gamma(x0w)
        neg = neg * neg_meas_1q(meas, Gamma)

    dt = time.time() - start_time
    print('--------------------- WIGNER NEGATIVITY ----------------------')
    print('Wigner Log Neg:  ', np.log(neg))
    print('Computation time:', dt)
    print('--------------------------------------------------------------')
    return [1,1/2,1/2]*x_len, np.log(neg)

