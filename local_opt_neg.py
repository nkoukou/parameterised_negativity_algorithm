import autograd.numpy as np
import itertools as it

from state_functions import DIM
from numpy.random import default_rng
from autograd import(grad)
from scipy.optimize import(Bounds, minimize)
from scipy.optimize import(basinhopping)

from circuit_components import(makeState, makeGate, makeCsum)
from phase_space import(D1q_list,x2Gamma, neg_state_1q, neg_gate_1q_max,
                   neg_gate_2q_max, neg_meas_1q, W_state_1q, W_gate_1q, W_gate_2q, W_meas_1q)

def optimizer(cost_function, x0, **kwargs):
    options = {'opt_method':'B', 'niter': 1}
    options.update(kwargs)
    opt_method = options['opt_method']
    if opt_method=='B': # autograd
        grad_cost_function = grad(cost_function)
        def func(x):
            return cost_function(x), grad_cost_function(x)
        optimize_result = basinhopping(func, x0, minimizer_kwargs={"method":"L-BFGS-B","jac":True}, niter=options['niter'])
    elif opt_method=='NG': # Powell
        optimize_result = minimize(cost_function, x0, method='Powell')
    elif opt_method=='G': # Without autograd
        grad_cost_function = grad(cost_function)
        optimize_result = minimize(cost_function, x0, method='L-BFGS-B',jac=grad_cost_function)
    else: raise Exception('Invalid optimisation method')
    return optimize_result
    
x_len = DIM*DIM-1
x0w = [1,0,0,0,0,0,1,0]
def weight(Gamma):
    return np.sqrt(np.real(np.trace(np.dot(Gamma,np.conjugate(Gamma.T)))))

def local_opt_State1q(rho, **kwargs):
    def cost_function(x):
        Gamma = x2Gamma(x)
        return weight(Gamma)*neg_state_1q(rho, Gamma)
    optimize_result = optimizer(cost_function, x0w, **kwargs)
    x_opt = optimize_result.x
    return x_opt

def local_opt_Gate2q(U2q,x_in1,x_in2,**kwargs):
    Gamma_in1 = x2Gamma(x_in1)
    Gamma_in2 = x2Gamma(x_in2)
    def cost_function(x):
        x_out1 = x[0:x_len]
        x_out2 = x[x_len:2*x_len]
        Gamma_out1 = x2Gamma(x_out1)
        Gamma_out2 = x2Gamma(x_out2)
        return weight(Gamma_out1)*weight(Gamma_out2)*neg_gate_2q_max(U2q, Gamma_in1,Gamma_in2,Gamma_out1,Gamma_out2)
    x0 = np.append(x_in1,x_in2)
    optimize_result = optimizer(cost_function, x0,**kwargs)
    x_opt = optimize_result.x
    x_opt1 = x_opt[0:x_len]
    x_opt2 = x_opt[x_len:2*x_len]
    return [x_opt1,x_opt2]

def get_opt_x(circuit,**kwargs):
    [rho_list,gate_U2q_list,gate_qudit_index_list,meas_list] = circuit
    x_rho_opt_list = []
    neg_rho_opt_list = []
    neg_tot = 1.
    for rho in rho_list:
        x_opt = local_opt_State1q(rho,**kwargs)
        x_rho_opt_list.append(x_opt)
        neg = neg_state_1q(rho, x2Gamma(x_opt))
        neg_rho_opt_list.append(neg)
        neg_tot *= neg
        print('neg_state:',neg,'\t weight:\t',weight(x2Gamma(x_opt)))
    
    x_running = x_rho_opt_list.copy()
    x_gate_out_opt_list = []
    neg_gate_opt_list = []
    for gate_index in range(len(gate_U2q_list)):
        U2q = gate_U2q_list[gate_index]
        [qudit_index1,qudit_index2] = gate_qudit_index_list[gate_index]
        x_in1 = x_running[qudit_index1]
        x_in2 = x_running[qudit_index2]
        [x_out_opt1,x_out_opt2] = local_opt_Gate2q(U2q, x_in1, x_in2,**kwargs)
        x_gate_out_opt_list.append([x_out_opt1,x_out_opt2].copy())
        neg = neg_gate_2q_max(U2q, x2Gamma(x_in1),x2Gamma(x_in2),x2Gamma(x_out_opt1),x2Gamma(x_out_opt2))
        neg_gate_opt_list.append(neg)
        neg_tot *= neg
        print('neg_gate(',gate_index+1,'):',neg,'\t weight:\t',[weight(x2Gamma(x_out_opt1)),weight(x2Gamma(x_out_opt2))])
        x_running[qudit_index1] = x_out_opt1
        x_running[qudit_index2] = x_out_opt2
    
    x_meas_opt_list = x_running
    neg_meas_opt_list = []
    for qudit_index in range(len(meas_list)):
        E = meas_list[qudit_index]
        x_out = x_meas_opt_list[qudit_index]
        neg = neg_meas_1q(E, x2Gamma(x_out))
        neg_meas_opt_list.append(neg)
        neg_tot *= neg
        print('neg_meas:',neg)
        
    print('neg_tot:', neg_tot)
    
    return x_rho_opt_list, x_gate_out_opt_list, x_meas_opt_list, neg_rho_opt_list, neg_gate_opt_list, neg_meas_opt_list, neg_tot

def show_Wigner_neg_x(circuit):
    [rho_list,gate_U2q_list,gate_qudit_index_list,meas_list] = circuit
    neg_tot = 1.
    for rho in rho_list:
        neg = neg_state_1q(rho, x2Gamma(x0w))
        neg_tot *= neg
        print('neg_state:',neg)
    
    for gate_index in range(len(gate_U2q_list)):
        U2q = gate_U2q_list[gate_index]
        [qudit_index1,qudit_index2] = gate_qudit_index_list[gate_index]
        neg = neg_gate_2q_max(U2q, x2Gamma(x0w),x2Gamma(x0w),x2Gamma(x0w),x2Gamma(x0w))
        neg_tot *= neg
        print('neg_gate(',gate_index+1,'):',neg)
    
    for qudit_index in range(len(meas_list)):
        E = meas_list[qudit_index]
        neg = neg_meas_1q(E, x2Gamma(x0w))
        neg_tot *= neg
        print('neg_meas:',neg)
        
    print('neg_tot:', neg_tot)
    print('---------------------------------------')
    return neg_tot

rho_string_list = ['0','1','2','+']
# rho_string_list = ['0','1','2','+','m','S','N','T']
def get_rand_rho_list(qudit_num):
    rho_list = []
    for n in range(qudit_num):
        rho_list.append(makeState(np.random.choice(rho_string_list)))
    return rho_list

gate_string_1q_list = ['1','H','S','T']
def get_rand_gate_U2q_list(circuit_length,**kwargs):
    options = {'Tgate_prob': 0.3}
    options.update(kwargs)
    
    Tgate_prob = options['Tgate_prob']
    prob_list = [(1-Tgate_prob)/3, (1-Tgate_prob)/3, (1-Tgate_prob)/3, Tgate_prob]
    gate_U2q_list = []
    for gate_index in range(circuit_length):
        U1qA = makeGate(np.random.choice(gate_string_1q_list,p=prob_list))
        U1qB = makeGate(np.random.choice(gate_string_1q_list,p=prob_list))
        U_AB_loc = np.kron(U1qA,U1qB)
        U_AB_tot = np.dot(U_AB_loc,makeCsum(np.random.choice(['C+','+C'])))
        gate_U2q_list.append(U_AB_tot)
    return gate_U2q_list

def get_rand_gate_qudit_index_list(circuit_length,qudit_num,**kwargs):
    options = {'type': 'r'}
    options.update(kwargs)

    gate_qudit_index_list = []
    if options['type']=='r':
        for gate_index in range(circuit_length):
            rng = default_rng()
            gate_qudit_index = rng.choice(qudit_num, size=2, replace=False)
            gate_qudit_index_list.append(gate_qudit_index)
            
    elif options['type']=='c':
        qudit_index = 0
        for gate_index in range(circuit_length):
            gate_qudit_index_list.append([qudit_index,qudit_index+1])
            qudit_index += 2
            if qudit_index == qudit_num-2 and qudit_num%2 == 0:
                qudit_index = 1
            elif qudit_index == qudit_num-1 and qudit_num%2 == 0:
                qudit_index = 0
            elif qudit_index == qudit_num-2 and qudit_num%2 == 1:
                qudit_index = 0
            elif qudit_index == qudit_num-1 and qudit_num%2 == 1:
                qudit_index = 1
    return gate_qudit_index_list

def get_rand_meas_list(qudit_num):
    meas_list = []
    for n in range(qudit_num):
        meas_string = np.random.choice(['0','/'])
        if meas_string=='0':
            E = makeState(np.random.choice(rho_string_list))
        elif meas_string=='/':
            E = np.eye(DIM)
        meas_list.append(E)
    return meas_list

def get_rand_circuit(qudit_num,circuit_length,**kwargs):
    rho_list = get_rand_rho_list(qudit_num)
    gate_U2q_list = get_rand_gate_U2q_list(circuit_length,**kwargs)
    gate_qudit_index_list = get_rand_gate_qudit_index_list(circuit_length,qudit_num,**kwargs)
    meas_list = get_rand_meas_list(qudit_num)
    circuit = [rho_list,gate_U2q_list,gate_qudit_index_list,meas_list]
    return circuit
