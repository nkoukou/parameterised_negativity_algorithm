import os
import time
import itertools as it
import autograd.numpy as np
from autograd import(grad)

from state_functions import DIM
from numpy.random import default_rng
from scipy.optimize import(Bounds, minimize)
from scipy.optimize import(basinhopping)

from phase_space import(x2Gamma, neg_state_1q, neg_gate_1q_max,
                   neg_gate_2q_max, neg_meas_1q, neg_gate_3q_max)


def wigner_neg_compressed(compressed_circuit, **kwargs):
    options = {'Wigner Negativity Options'}
    options.update(kwargs)

    x_len = int(len(compressed_circuit['state_list']) + 2*len(
                    compressed_circuit['gate_list']))
    x0w = [1,0,0,0,0,0,1,0]

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
    return [1,0,0,0,0,0,1,0]*x_len, np.log(neg)

def wigner_neg_compressed_3q(compressed_circuit, **kwargs):
    options = {'Wigner Negativity Options'}
    options.update(kwargs)

    x_len = int(len(compressed_circuit['state_list']) + 8*len(
                    compressed_circuit['gate_list']))
    x0w = [1,0,0,0,0,0,1,0]

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
        Gamma_in1 = x2Gamma(x0w)
        Gamma_in2 = x2Gamma(x0w)
        Gamma_in3 = x2Gamma(x0w)
        Gamma_out1 = x2Gamma(x0w)
        Gamma_out2 = x2Gamma(x0w)
        Gamma_out3 = x2Gamma(x0w)
        neg = neg * neg_gate_3q_max(gate, Gamma_in1, Gamma_in2, Gamma_in3,
                                          Gamma_out1, Gamma_out2, Gamma_out3)

    for meas in compressed_circuit['meas_list']:
        if str(meas) == '/': continue
        Gamma = x2Gamma(x0w)
        neg = neg * neg_meas_1q(meas, Gamma)

    dt = time.time() - start_time
    print('--------------------- WIGNER NEGATIVITY ----------------------')
    print('Wigner Log Neg:  ', np.log(neg))
    print('Computation time:', dt)
    print('--------------------------------------------------------------')
    return [1,0,0,0,0,0,1,0]*x_len, np.log(neg)

def optimize_neg_compressed(compressed_circuit, **kwargs):
    ''' circuit_compressed - output of random_circuit_generator.
                                       compress_circuit function.
    '''
    options = {'opt_method': 'B', 'niter': 1}
    options.update(kwargs)

    x_len = int(len(compressed_circuit['state_list']) + 2*len(compressed_circuit['gate_list']))

    def cost_function(x):
        neg = 1.
        Gamma_index = 0
        Gammas = []
        for state in compressed_circuit['state_list']:
            Gamma = x2Gamma(x[8*Gamma_index:8*(Gamma_index+1)])
            neg = neg * neg_state_1q(state, Gamma)
            Gammas.append(Gamma)
            Gamma_index += 1

        gate_index = 0
        for gate in compressed_circuit['gate_list']:
            idx = compressed_circuit['index_list'][gate_index]
            gate_index += 1

            if len(idx)==1: raise Exception('There are disentangled wires.')
            GammaC_in = Gammas[idx[0]]
            GammaT_in = Gammas[idx[1]]
            GammaC_out = x2Gamma(x[8*Gamma_index:8*(Gamma_index+1)])
            GammaT_out = x2Gamma(x[8*(Gamma_index+1):8*(Gamma_index+2)])
            neg = neg * neg_gate_2q_max(gate, GammaC_in, GammaT_in, GammaC_out, GammaT_out)
            Gammas[idx[0]] = GammaC_out
            Gammas[idx[1]] = GammaT_out
            Gamma_index += 2

        qudit_index = 0
        for meas in compressed_circuit['meas_list']:
            if str(meas) == '/': continue
            Gamma = Gammas[qudit_index]
            qudit_index += 1
            neg = neg * neg_meas_1q(meas, Gamma)
        return np.log(neg)

    x0 = 2*np.random.rand(8*x_len)-1
    optimize_result, dt = optimizer(cost_function, x0, options['opt_method'], niter = options['niter'])
    optimized_x = optimize_result.x
    optimized_value = cost_function(optimized_x)
    print('--------------------- GLOBAL OPTIMIZATION --------------------\n', options)
    print('Optimized Log Neg:', optimized_value)
    print('Computation time: ', dt)
    print('--------------------------------------------------------------')

    return optimized_x, optimized_value

def optimize_neg_compressed_3q(compressed_circuit, **kwargs):
    options = {'opt_method': 'B', 'niter': 1}
    options.update(kwargs)

    x_len = int(len(compressed_circuit['state_list']) + 8*len(compressed_circuit['gate_list']))

    def cost_function(x):
        neg = 1.
        Gamma_index = 0
        Gammas = []
        for state in compressed_circuit['state_list']:
            Gamma = x2Gamma(x[8*Gamma_index:8*(Gamma_index+1)])
            neg = neg * neg_state_1q(state, Gamma)
            Gammas.append(Gamma)
            Gamma_index += 1

        gate_index = 0
        for gate in compressed_circuit['gate_list']:
            idx = compressed_circuit['index_list'][gate_index]
            gate_index += 1

            if len(idx)==1 or len(idx)==2 : raise Exception('There are disentangled wires.')
            Gamma_in1 = Gammas[idx[0]]
            Gamma_in2 = Gammas[idx[1]]
            Gamma_in3 = Gammas[idx[2]]
            Gamma_out1 = x2Gamma(x[8*Gamma_index:8*(Gamma_index+1)])
            Gamma_out2 = x2Gamma(x[8*(Gamma_index+1):8*(Gamma_index+2)])
            Gamma_out3 = x2Gamma(x[8*(Gamma_index+2):8*(Gamma_index+3)])
            neg = neg * neg_gate_3q_max(gate, Gamma_in1, Gamma_in2, Gamma_in3, Gamma_out1, Gamma_out2, Gamma_out3)
            Gammas[idx[0]] = Gamma_out1
            Gammas[idx[1]] = Gamma_out2
            Gammas[idx[2]] = Gamma_out3
            Gamma_index += 3

        qudit_index = 0
        for meas in compressed_circuit['meas_list']:
            if str(meas) == '/': continue
            Gamma = Gammas[qudit_index]
            qudit_index += 1
            neg = neg * neg_meas_1q(meas, Gamma)
        return np.log(neg)

    x0 = 2*np.random.rand(8*x_len)-1
    optimize_result, dt = optimizer(cost_function, x0, options['opt_method'], niter = options['niter'])
    optimized_x = optimize_result.x
    optimized_value = cost_function(optimized_x)
    print('--------------------- GLOBAL OPTIMIZATION --------------------\n', options)
    print('Optimized Log Neg:', optimized_value)
    print('Computation time: ', dt)
    print('--------------------------------------------------------------')

    return optimized_x, optimized_value

def optimizer(cost_function, x0, opt_method='B', niter=10):
    start_time = time.time()

    if opt_method=='B': # autograd
        grad_cost_function = grad(cost_function)
        def func(x):
            return cost_function(x), grad_cost_function(x)
        optimize_result = basinhopping(func, x0, minimizer_kwargs={"method":"L-BFGS-B","jac":True},niter=niter)

    elif opt_method=='NG': # Powell
        optimize_result = minimize(cost_function, x0, method='Powell')

    elif opt_method=='G': # Without autograd
        optimize_result = minimize(cost_function, x0, method='L-BFGS-B',jac=grad_cost_function)

    else:
        raise Exception('Invalid optimisation method')

    dt = time.time()-start_time

    return optimize_result, dt

####################################################################
# ------------------ Local optimisation part ----------------------#

def local_optimizer(cost_function, x0, **kwargs):
    options = {'opt_method' : 'B', 'niter' : 3}
    options.update(kwargs)

    opt_method = options['opt_method']
    if opt_method=='B': # autograd
        grad_cost_function = grad(cost_function)
        def func(x):
            return cost_function(x), grad_cost_function(x)
        optimize_result = basinhopping(func, x0, minimizer_kwargs={"method":"L-BFGS-B","jac":True}, niter = options['niter'])
    elif opt_method=='NG': # Powell
        optimize_result = minimize(cost_function, x0, method='Powell')
    elif opt_method=='G': # Without autograd
        grad_cost_function = grad(cost_function)
        optimize_result = minimize(cost_function, x0, method='L-BFGS-B',jac=grad_cost_function)
    else: raise Exception('Invalid optimisation method')
    return optimize_result

para_len = DIM*DIM-1
x0w = [1,0,0,0,0,0,1,0]
def weight(Gamma):
    return np.sqrt(np.real(np.trace(np.dot(Gamma,np.conjugate(Gamma.T)))))

def local_opt_State1q(rho, **kwargs):
    def cost_function(x):
        Gamma = x2Gamma(x)
        return weight(Gamma)*neg_state_1q(rho, Gamma)
    optimize_result = local_optimizer(cost_function, x0w, **kwargs)
    x_opt = optimize_result.x
    return x_opt

def local_opt_Gate2q(U2q,x_in1,x_in2,**kwargs):
    Gamma_in1 = x2Gamma(x_in1)
    Gamma_in2 = x2Gamma(x_in2)
    def cost_function(x):
        x_out1 = x[0:para_len]
        x_out2 = x[para_len:2*para_len]
        Gamma_out1 = x2Gamma(x_out1)
        Gamma_out2 = x2Gamma(x_out2)
        return weight(Gamma_out1)*weight(Gamma_out2)*neg_gate_2q_max(U2q, Gamma_in1,Gamma_in2,Gamma_out1,Gamma_out2)
    x0 = np.append(x_in1,x_in2)
    optimize_result = local_optimizer(cost_function, x0,**kwargs)
    x_opt = optimize_result.x
    x_opt1 = x_opt[0:para_len]
    x_opt2 = x_opt[para_len:2*para_len]
    return [x_opt1,x_opt2]

def local_opt_neg_compressed(compressed_circuit,**kwargs):
    t0 = time.time()
    options = {'opt_method':'B', 'niter': 3, 'show_detailed_log':False}
    options.update(kwargs)

    rho_list = compressed_circuit['state_list']
    gate_U2q_list = compressed_circuit['gate_list']
    gate_qudit_index_list = compressed_circuit['index_list']
    meas_list = compressed_circuit['meas_list']

    x_rho_opt_list = []
    neg_rho_opt_list = []
    neg_tot = 1.
    for rho in rho_list:
        x_opt = local_opt_State1q(rho,**options)
        x_rho_opt_list.append(x_opt)
        neg = neg_state_1q(rho, x2Gamma(x_opt))
        neg_rho_opt_list.append(neg)
        neg_tot *= neg
        if options['show_detailed_log']:
            print('neg_state:',neg,'\t weight:\t',weight(x2Gamma(x_opt)))

    x_running = x_rho_opt_list.copy()
    x_gate_out_opt_list = []
    neg_gate_opt_list = []
    for gate_index in range(len(gate_U2q_list)):
        U2q = gate_U2q_list[gate_index]
        [qudit_index1,qudit_index2] = gate_qudit_index_list[gate_index]
        x_in1 = x_running[qudit_index1]
        x_in2 = x_running[qudit_index2]
        [x_out_opt1,x_out_opt2] = local_opt_Gate2q(U2q, x_in1, x_in2,**options)
        x_gate_out_opt_list.append([x_out_opt1,x_out_opt2].copy())
        neg = neg_gate_2q_max(U2q, x2Gamma(x_in1),x2Gamma(x_in2),x2Gamma(x_out_opt1),x2Gamma(x_out_opt2))
        neg_gate_opt_list.append(neg)
        neg_tot *= neg
        if options['show_detailed_log']:
            print('neg_gate(',gate_index+1,'):',neg,'\t weight:\t',[weight(x2Gamma(x_out_opt1)),weight(x2Gamma(x_out_opt2))])
        x_running[qudit_index1] = x_out_opt1
        x_running[qudit_index2] = x_out_opt2

    x_meas_opt_list = x_running
    qudit_index = 0
    neg_meas_opt_list = []
    for meas in compressed_circuit['meas_list']:
        if str(meas) == '/': continue
        x_out = x_meas_opt_list[qudit_index]
        qudit_index += 1
        neg = neg_meas_1q(meas, x2Gamma(x_out))
        neg_meas_opt_list.append(neg)
        neg_tot *= neg
        if options['show_detailed_log']:
            print('neg_meas:',neg)

    print('--------------------- LOCAL OPTIMIZATION ---------------------\n', options)
    print('Local Opt Log Neg:', np.log(neg_tot))
    print('Computation time: ',time.time() - t0)
    print('--------------------------------------------------------------')

    x_opt_list_tot = np.append(np.array(x_rho_opt_list).flatten(),np.array(x_gate_out_opt_list).flatten())
    return x_opt_list_tot, np.log(neg_tot)

def local_opt_Gate3q(U3q,x_in1,x_in2,x_in3,**kwargs):
    Gamma_in1 = x2Gamma(x_in1)
    Gamma_in2 = x2Gamma(x_in2)
    Gamma_in3 = x2Gamma(x_in3)
    def cost_function(x):
        x_out1 = x[0:para_len]
        x_out2 = x[para_len:2*para_len]
        x_out3 = x[2*para_len:3*para_len]
        Gamma_out1 = x2Gamma(x_out1)
        Gamma_out2 = x2Gamma(x_out2)
        Gamma_out3 = x2Gamma(x_out3)
        return weight(Gamma_out1)*weight(Gamma_out2)*weight(Gamma_out3)*neg_gate_3q_max(U3q, Gamma_in1,Gamma_in2, Gamma_in3, Gamma_out1,Gamma_out2, Gamma_out3)
    x0 = np.append(np.append(x_in1,x_in2), x_in3)
    optimize_result = local_optimizer(cost_function, x0,**kwargs)
    x_opt = optimize_result.x
    x_opt1 = x_opt[0:para_len]
    x_opt2 = x_opt[para_len:2*para_len]
    x_opt3 = x_opt[2*para_len:3*para_len]
    return [x_opt1,x_opt2, x_opt3]

def local_opt_neg_compressed_3q(compressed_circuit,**kwargs):
    t0 = time.time()
    options = {'opt_method':'B', 'niter': 3, 'show_detailed_log':False}
    options.update(kwargs)

    rho_list = compressed_circuit['state_list']
    gate_U3q_list = compressed_circuit['gate_list']
    gate_qudit_index_list = compressed_circuit['index_list']
    meas_list = compressed_circuit['meas_list']

    x_rho_opt_list = []
    neg_rho_opt_list = []
    neg_tot = 1.
    for rho in rho_list:
        x_opt = local_opt_State1q(rho,**options)
        x_rho_opt_list.append(x_opt)
        neg = neg_state_1q(rho, x2Gamma(x_opt))
        neg_rho_opt_list.append(neg)
        neg_tot *= neg
        if options['show_detailed_log']:
            print('neg_state:',neg,'\t weight:\t',weight(x2Gamma(x_opt)))

    x_running = x_rho_opt_list.copy()
    x_gate_out_opt_list = []
    neg_gate_opt_list = []
    for gate_index in range(len(gate_U3q_list)):
        U3q = gate_U3q_list[gate_index]
        [qudit_index1,qudit_index2, qudit_index3] = gate_qudit_index_list[gate_index]
        x_in1 = x_running[qudit_index1]
        x_in2 = x_running[qudit_index2]
        x_in3 = x_running[qudit_index3]
        [x_out_opt1,x_out_opt2,x_out_opt3] = local_opt_Gate3q(U3q, x_in1, x_in2, x_in3, **options)
        x_gate_out_opt_list.append([x_out_opt1,x_out_opt2,x_out_opt3].copy())
        neg = neg_gate_3q_max(U3q, x2Gamma(x_in1),x2Gamma(x_in2),x2Gamma(x_in3),x2Gamma(x_out_opt1),x2Gamma(x_out_opt2),x2Gamma(x_out_opt3))
        neg_gate_opt_list.append(neg)
        neg_tot *= neg
        if options['show_detailed_log']:
            print('neg_gate(',gate_index+1,'):',neg,'\t weight:\t',[weight(x2Gamma(x_out_opt1)),weight(x2Gamma(x_out_opt2)),weight(x2Gamma(x_out_opt3))])
        x_running[qudit_index1] = x_out_opt1
        x_running[qudit_index2] = x_out_opt2
        x_running[qudit_index3] = x_out_opt3

    x_meas_opt_list = x_running
    qudit_index = 0
    neg_meas_opt_list = []
    for meas in compressed_circuit['meas_list']:
        if str(meas) == '/': continue
        x_out = x_meas_opt_list[qudit_index]
        qudit_index += 1
        neg = neg_meas_1q(meas, x2Gamma(x_out))
        neg_meas_opt_list.append(neg)
        neg_tot *= neg
        if options['show_detailed_log']:
            print('neg_meas:',neg)

    print('--------------------- LOCAL OPTIMIZATION ---------------------\n', options)
    print('Local Opt Log Neg:', np.log(neg_tot))
    print('Computation time: ',time.time() - t0)
    print('--------------------------------------------------------------')

    x_opt_list_tot = np.append(np.array(x_rho_opt_list).flatten(),np.array(x_gate_out_opt_list).flatten())
    return x_opt_list_tot, np.log(neg_tot)







