import os
import time
import autograd.numpy as np
from autograd import(grad)

from scipy.optimize import(Bounds, minimize)
from scipy.optimize import(basinhopping)
from QUBIT_phase_space import(x2Gamma, neg_state_1q, neg_gate_1q_max, neg_gate_2q_max, neg_meas_1q)

def optimize_neg_compressed(compressed_circuit, **kwargs):
    options = {'opt_method': 'B', 'niter': 100}
    options.update(kwargs)
    
    x_len = int(len(compressed_circuit['state_list']) + 2*len(compressed_circuit['gate_list']))

    def cost_function(x):
        neg = 1.
        Gamma_index = 0
        Gammas = []
        for state in compressed_circuit['state_list']:
            Gamma = x2Gamma(x[3*Gamma_index:3*(Gamma_index+1)])
            neg = neg * neg_state_1q(state, Gamma)
            Gammas.append(Gamma)
            Gamma_index += 1
        
        gate_index = 0
        for gate in compressed_circuit['gate_list']:
            idx = compressed_circuit['qudit_index_list'][gate_index]
            gate_index += 1
            
            if len(idx)==1: raise Exception('There are disentangled wires.')
            GammaC_in = Gammas[idx[0]]
            GammaT_in = Gammas[idx[1]]
            GammaC_out = x2Gamma(x[3*Gamma_index:3*(Gamma_index+1)])
            GammaT_out = x2Gamma(x[3*(Gamma_index+1):3*(Gamma_index+2)])
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

    x0 = 2*np.random.rand(3*x_len)-1
    optimize_result, dt = optimizer(cost_function, x0, options['opt_method'], niter = options['niter'])
    optimized_x = optimize_result.x
    optimized_value = cost_function(optimized_x)
    print('------------------GLOBAL OPTIMIZATION--------------------------\n',options)
    print('Optimized Log Neg:', optimized_value)
    print('Computation time: ', dt)
    print('---------------------------------------------------------------')
    
    return optimized_x, optimized_value
    
def optimizer(cost_function, x0, opt_method='B', niter = 10):
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
