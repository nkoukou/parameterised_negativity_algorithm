import os
import time
import autograd.numpy as np
from autograd import(grad)

from scipy.optimize import(Bounds, minimize)
from scipy.optimize import(basinhopping)

from QUBIT_circuit_components import(makeState, makeGate)
from QUBIT_phase_space import(x2Gamma, neg_state_1q, neg_gate_1q_max,
                   neg_gate_2q_max, neg_meas_1q, n_Gammas)

def optimize_neg_compressed(circuit, opt_method='B', path='test_directory'):
    ''' circuit_compressed - output of QUBIT_random_circuit_generator.
                                       compress_circuit function.
    '''
    state_string, gate_sequence, meas_string = circuit
    x_len = n_Gammas(circuit)

    def cost_function(x):
        neg = 1.
        Gamma_index = 0
        Gammas = []
        for s in state_string:
            state = makeState(s)
            Gamma = x2Gamma(x[3*Gamma_index:3*(Gamma_index+1)])
            neg = neg * neg_state_1q(state, Gamma)
            Gammas.append(Gamma)
            Gamma_index += 1
        for g in gate_sequence:
            idx, gate = g[0], g[1]
            if len(idx)==1: raise Exception('There are disentangled wires.')
            GammaC_in = Gammas[idx[0]]
            GammaT_in = Gammas[idx[1]]
            GammaC_out = x2Gamma(x[3*Gamma_index:3*(Gamma_index+1)])
            GammaT_out = x2Gamma(x[3*(Gamma_index+1):3*(Gamma_index+2)])
            neg = neg * neg_gate_2q_max(gate, GammaC_in, GammaT_in,
                                              GammaC_out, GammaT_out)
            Gammas[idx[0]] = GammaC_out
            Gammas[idx[1]] = GammaT_out
            Gamma_index += 2
        for m in range(len(meas_string)):
            if meas_string[m]=='/': continue
            E = makeState(meas_string[m])
            Gamma = Gammas[m]
            neg = neg * neg_meas_1q(E, Gamma)
        return np.log(neg)

    # Wigner distribution
    x0 = []
    x0w = [1,1/2,1/2]
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
    # np.save(os.path.join('data', path, 'QUBIT_state_string_compressed.npy'),
    #         state_string)
    # np.save(os.path.join('data', path, 'QUBIT_gate_sequence_compressed.npy'),
    #         gate_sequence)
    # np.save(os.path.join('data', path, 'QUBIT_meas_string_compressed.npy'),
    #         meas_string)
    # np.save(os.path.join('data', path, 'QUBIT_optimized_x_compressed.npy'),
    #         optimized_x)
    # np.save(os.path.join('data', path, 'QUBIT_optimized_neg_compressed.npy'),
    #         optimized_value)

    return optimized_x, optimized_value

def optimize_neg(circuit, opt_method='B', path='test_directory'):
    ''' List circuit contains 3 elements:
        - state_string: 'T0TT0T'
        - gate_sequence: [[[state_index], U1q],
                          [[state_index_c, state_index_t], U2q],
                          ... ]
        - meas_string:  '01T111'
    '''
    state_string, gate_sequence, meas_string = circuit
    x_len = n_Gammas(circuit)

    def cost_function(x):
        neg = 1.
        Gamma_index = 0
        Gammas = []
        for s in state_string:
            state = makeState(s)
            Gamma = x2Gamma(x[3*Gamma_index:3*(Gamma_index+1)])
            neg = neg*neg_state_1q(state,Gamma)
            Gammas.append(Gamma)
            Gamma_index += 1
        for g in gate_sequence:
            idx, gate = g[0], makeGate(g[1])
            if len(idx)==1:
                if g[1]=='H':
                    Gammas[idx[0]] = np.dot(np.dot(gate, Gammas[idx[0]]),
                                            np.conjugate(gate.T))
                    continue
                Gamma_in = Gammas[idx[0]]
                Gamma_out = x2Gamma(x[3*Gamma_index:3*(Gamma_index+1)])
                neg = neg*neg_gate_1q_max(gate,Gamma_in,Gamma_out)
                Gammas[idx[0]] = Gamma_out
                Gamma_index += 1
            elif len(idx)==2:
                GammaC_in = Gammas[idx[0]]
                GammaT_in = Gammas[idx[1]]
                GammaC_out = x2Gamma(x[3*Gamma_index:3*(Gamma_index+1)])
                GammaT_out = x2Gamma(x[3*(Gamma_index+1):3*(Gamma_index+2)])
                neg = neg*neg_gate_2q_max(gate, GammaC_in, GammaT_in,
                                            GammaC_out, GammaT_out)
                Gammas[idx[0]] = GammaC_out
                Gammas[idx[1]] = GammaT_out
                Gamma_index += 2
            else: raise Exception('Too many gate indices')
        for m in range(len(meas_string)):
            if meas_string[m]!='/':
                E = makeState(meas_string[m])
                Gamma = Gammas[m]
                neg = neg*neg_meas_1q(E,Gamma)
        return np.log(neg)

    # Wigner distribution
    x0 = []
    x0w = [1,1/2,1/2]
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
                            disp=False, niter=100)
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


test = [np.eye(2)]*4
# test[0] = test[0]*makeGate('H')
# test[1] = test[1]*makeGate('H')
print(test)
#print(np.dot(makeGate('H'),test[0]))









