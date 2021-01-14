import os
import time
import autograd.numpy as np
from autograd import(grad)

from scipy.optimize import(Bounds, minimize)
from scipy.optimize import(basinhopping)

from QUBIT_circuit_components import(makeState, makeGate)
from QUBIT_phase_space import(x2Gamma, neg_state_1q, neg_gate_1q_max,
                   neg_gate_2q_max, neg_meas_1q)

def optimize_neg(circuit, opt_method='B', path='test_directory'):
    ''' List circuit contains 3 elements:
        - state_string: 'T0TT0T'
        - gate_sequence: [[[state_index], U1q],
                          [[state_index_c, state_index_t], U2q],
                          ... ]
        - meas_string:  '01T111'
    '''
    state_string, gate_sequence, meas_string = circuit

    current_state_index = []
    init_state_index = []
    x_index = 0
    for state_str in state_string:
        init_state_index.append([x_index, makeState(state_str)])
        current_state_index.append(x_index)
        x_index += 1

    gate_1q_index = []
    gate_2q_index = []
    for gate_str in gate_sequence:
        if len(gate_str[0])==1:
            t_index = (gate_str[0])[0]
            gate_1q_index.append([[current_state_index[t_index], x_index],
                                   makeGate(gate_str[1])])
            current_state_index[t_index] = x_index
            x_index += 1
        elif len(gate_str[0])==2:
            c_index = (gate_str[0])[0]
            t_index = (gate_str[0])[1]
            gate_2q_index.append( [[current_state_index[c_index],
                                    current_state_index[t_index],
                                    x_index,x_index+1], makeGate(gate_str[1])])
            current_state_index[c_index] = x_index
            current_state_index[t_index] = x_index+1
            x_index += 2

#    meas_index = []
#    for meas_str in meas_string:
#        if meas_str != '1':
#            meas_index.append([x_index, makeState1q(meas_str)])
#            x_index += 1

    meas_index = []
    k = 0
    for meas_str in meas_string:
        if meas_str != '/':
            meas_index.append([current_state_index[k], makeState(meas_str)])
        k+=1
    x_len = x_index

    # print('x_len:', x_len)
    # print('init_state_index:', init_state_index)
    # print('gate_1q_index:', gate_1q_index)
    # print('gate_2q_index:', gate_2q_index)
    # print('meas_index:', meas_index)
    # print('---------------------------------------------------------------')

    def cost_function(x):
        Gamma_list = []
        for x_index in range(x_len):
            Gamma_list.append(x2Gamma(x[3*x_index:3*(x_index+1)]))
        neg = 1.
        for state_index in init_state_index:
            rho = state_index[1]
            Gamma = Gamma_list[state_index[0]]
            neg = neg*neg_state_1q(rho,Gamma)
        for gate_index in gate_1q_index:
            U1q = gate_index[1]
            Gamma_in = Gamma_list[(gate_index[0])[0]]
            Gamma_out = Gamma_list[(gate_index[0])[1]]
            neg = neg*neg_gate_1q_max(U1q,Gamma_in,Gamma_out)
        for gate_index in gate_2q_index:
            U2q = gate_index[1]
            GammaC_in = Gamma_list[(gate_index[0])[0]]
            GammaT_in = Gamma_list[(gate_index[0])[1]]
            GammaC_out = Gamma_list[(gate_index[0])[2]]
            GammaT_out = Gamma_list[(gate_index[0])[3]]
            neg = neg*neg_gate_2q_max(U2q, GammaC_in, GammaT_in,
                                        GammaC_out, GammaT_out)
        for m_index in meas_index:
            E = m_index[1]
            Gamma = Gamma_list[m_index[0]]
            neg = neg*neg_meas_1q(E,Gamma)
        return np.log(neg)

    # Wigner distribution
    x0 = []
    x0w = [1,1/2,-1/2]
    for x_index in range(x_len):
        x0 = np.append(x0, x0w)
    start_time = time.time()
    out = cost_function(x0)
    dt = time.time() - start_time
    print('---------------------------------------------------------------')
    print('Wigner Log Neg:  ', out)
    print('Computation time:', dt)

    # Optimised distribution
    x0 = 2*np.random.rand(3*x_len)-1
    optimize_result, dt = optimizer(cost_function, x0, opt_method)
    optimized_x = optimize_result.x
    optimized_value = cost_function(optimized_x)
    print('---------------------------------------------------------------')
    print('Optimized Log Neg:', optimized_value)
    print('Computation time: ', dt)

    # # Saving data
    # directory = os.path.join('data', path)
    # if not os.path.isdir(directory): os.mkdir(directory)
    # np.save(os.path.join('data', path, 'state_string.npy'), state_string)
    # np.save(os.path.join('data', path, 'gate_sequence.npy'), gate_sequence)
    # np.save(os.path.join('data', path, 'meas_string.npy'), meas_string)
    # np.save(os.path.join('data', path, 'optimized_x.npy'), optimized_x)
    # np.save(os.path.join('data', path, 'optimized_neg.npy'), optimized_value)

    return optimized_x, optimized_value


def optimizer(cost_function, x0, opt_method='B'):
    if opt_method=='B': # autograd
        grad_cost_function = grad(cost_function)
        def func(x):
            return cost_function(x), grad_cost_function(x)
        start_time = time.time()
        optimize_result = basinhopping(func, x0,
                            minimizer_kwargs={"method":"L-BFGS-B","jac":True},
                            disp=False, niter=1)
    elif opt_method=='NG': # Powell
        start_time = time.time()
        optimize_result = minimize(cost_function, x0, method='Powell')
    # elif opt_method=='G': # Without autograd
    #     start_time = time.time()
    #     optimize_result = minimize(cost_function, x0, method='L-BFGS-B',
    #                         jac=grad_cost_function, bounds=bnds,
    #                         options={'disp': show_log})
    else: raise Exception('Invalid optimisation method')
    dt = time.time()-start_time
    return optimize_result, dt












