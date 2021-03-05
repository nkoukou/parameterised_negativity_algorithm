import autograd.numpy as np
import itertools as it
import time

from QUBIT_state_functions import DIM
from numpy.random import default_rng
from autograd import(grad)
from scipy.optimize import(Bounds, minimize)
from scipy.optimize import(basinhopping)

from QUBIT_circuit_components import(makeState, makeGate, makeCsum)
from QUBIT_phase_space import(D1q_list,x2Gamma, neg_state_1q, neg_gate_1q_max,
                   neg_gate_2q_max, neg_meas_1q, W_state_1q, W_gate_1q, W_gate_2q, W_meas_1q)

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
x0w = [1,1/2,1/2]
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
