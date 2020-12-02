import numpy as np
from gate_seq2symplectic import gate_sequence2symplectic_form_merged, symplectic_inverse
from quasi_dist import W_state_list_1q, W_meas_list_1q
from make_state import makeState1q, makeMeas1q

import scipy.optimize as opt
import time

DIM= 3

'''Default expression of a phase space point is (x1, p1, x2, p2, ..., xN, pN)'''
x_range = list(range(DIM))
p_range = list(range(DIM))

'''State Preparation'''
def neg_state(state_string,gamma):
    qudit_num = len(state_string)
    neg_state = 1
    for qudit_index in range(qudit_num):
        S_id_1q = np.eye(2)
        Cov_1q = np.diag(gamma[2*qudit_index:2*qudit_index+2])
        rho = makeState1q(state_string[qudit_index])
        W_list = np.reshape(W_state_list_1q(rho,Cov_1q,S_id_1q),(DIM,DIM))
        neg_state = neg_state*(abs(W_list).sum())
    return neg_state

'''Circuit components'''
'''Not need for Clifford circuits'''
'''phase space point update: w -> Sw + z (deterministic)'''

'''Measurment on a single output mode'''
def neg_meas(meas_string, gamma, S):
    Cov = np.diag(gamma)
    MeasO, Meas_mode = makeMeas1q(meas_string)
    W_list = np.reshape(W_meas_list_1q(MeasO,Meas_mode, Cov,S),(DIM,DIM))
    neg_meas = np.max(abs(W_list))
    return neg_meas

'''Total negativity of the circuit'''
def neg_tot(state_string, gate_string, meas_string, gamma):
    Stot, ztot = gate_sequence2symplectic_form_merged(gate_string)
    S = symplectic_inverse(Stot)
    return neg_state(state_string,gamma) * neg_meas(meas_string, gamma, S)

'''Optimize the parameter gamma'''
def opt_neg_tot(circuit_string, opt_method = 'Powell', show_log = False):
    state_string = circuit_string[0]
    gate_sequence = circuit_string[1:-1]
    meas_string = circuit_string[-1]

    Stot, ztot = gate_sequence2symplectic_form_merged(gate_sequence)
    S = symplectic_inverse(Stot)
#    S = Stot
    qudit_num = len(state_string)
    def cost_function(x):
        return neg_state(state_string,x) * neg_meas(meas_string, x, S)
    x0 = 2.*np.random.rand(2*qudit_num)-1.
    bnds = []
    for ll in range(2*qudit_num):
        bnds.append([-10., 10.])

    start_time = time.time()
    optimize_result = opt.minimize(cost_function, x0,
#    bounds=bnds,
    method=opt_method,
    options={'disp':show_log}
    )
    optimized_x = optimize_result.x
    optimized_value = cost_function(optimized_x)

    if(show_log):
        print('------------------------------------------------------------------------')
        print('Optimization method: ', opt_method)
        print('Initial_s_list: ', x0)
        print('Optimized_s_list: ', optimized_x)
        print('Optmized Log-Negativity (state): ', neg_state(state_string,optimized_x))
        print('Optmized Log-Negativity (meas): ', neg_meas(meas_string, optimized_x, S))
        print('Optmized Log-Negativity (total): ', optimized_value, '(Computation time: ',(time.time() - start_time),'s)')
        print('------------------------------------------------------------------------')
    return optimized_x, optimized_value

def show_neg_result(circuit_string, gamma=None):
    print('----------------------------------------------------------------')
    state_string = circuit_string[0]
    qudit_num = len(state_string)
    print('State string:\t\t', state_string)
    print('Number of qutrit:\t', qudit_num)

    gate_sequence = circuit_string[1:-1]
    Stot, ztot = gate_sequence2symplectic_form_merged(gate_sequence)
    S = symplectic_inverse(Stot)
    print('Gate sequence: \t\t', gate_sequence)

    meas_string = circuit_string[-1]
    print('Measurement string:\t', meas_string)

    if gamma is None:
        gamma = np.zeros(2*qudit_num)

    print('Covariance Matrix:\t diag',gamma)
    neg_state_Wigner = neg_state(state_string,gamma)
    neg_meas_Wigner = neg_meas(meas_string, gamma, S)
    neg_tot_Wigner = neg_state_Wigner*neg_meas_Wigner

    print('State Negativity:\t', neg_state_Wigner)
    print('Measurement Negativity:\t', neg_meas_Wigner)
    print('Total Negativity:\t',neg_tot_Wigner)
    print('----------------------------------------------------------------')


def calc_hoeffding_bound(eps, delta, circuit_string):
    neg = neg_tot(circuit_string[0], circuit_string[1:-1], circuit_string[-1],
              gamma = np.zeros(2*len(circuit_string[0])))
    gamma_opt, neg_opt = opt_neg_tot(circuit_string)
    sample_size = int( 2/eps**2 * neg**2 * np.log(2/delta) )
    sample_size_opt = int( 2/eps**2 * neg_opt**2 * np.log(2/delta) )
    return sample_size, sample_size_opt, gamma_opt

'''Example code'''
# circuit_string = ['TTTTSTS','11S11H1','1CT111S','11Z1111']
# show_neg_result(circuit_string)
# neg = neg_tot(circuit_string[0], circuit_string[1:-1], circuit_string[-1],
#               gamma = np.zeros(2*len(circuit_string[0])))
# gamma_opt, neg_opt = opt_neg_tot(circuit_string,show_log = False)
# show_neg_result(circuit_string, gamma_opt)
# opt_neg = opt_neg_tot(circuit_string)
