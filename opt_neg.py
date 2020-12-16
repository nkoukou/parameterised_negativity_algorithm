import os
import time
import autograd.numpy as np
from autograd import(grad)

# from scipy.optimize import(Bounds, minimize)
from scipy.optimize import(basinhopping)

from circuit_components import(makeState1q)
from phase_space import(x2Gamma, neg_state_1q, neg_gate_1q,
                   neg_gate_Cliff_2q, neg_meas_1q)

def optimize_neg(state_string, gate_sequence, meas_string, opt_method='B',
                 path='test_directory'):
    ''' state_string: 'T0TT0T'
        gate_sequence: [[[state_index],U1q],[[state_index_c,state_index_t],U2q], ... ]
        meas_string:  '000000'
    '''
    # qudit_num = len(state_string)
    # x_length = qudit_num

    current_state_index = []
    Init_state_index = []
    x_index = 0
    for state_str in state_string:
        Init_state_index.append([x_index,makeState1q(state_str)])
        current_state_index.append(x_index)
        x_index = x_index + 1

    gate_1q_index = []
    gate_2q_index = []
    for gate_str in gate_sequence:
        if len(gate_str[0])==1:
            t_index = (gate_str[0])[0]
            gate_1q_index.append([[current_state_index[t_index],x_index],gate_str[1]])
            current_state_index[t_index] = x_index
            x_index = x_index + 1
        elif len(gate_str[0])==2:
            c_index = (gate_str[0])[0]
            t_index = (gate_str[0])[1]
            gate_2q_index.append( [[current_state_index[c_index],current_state_index[t_index],x_index,x_index+1],gate_str[1]])
            current_state_index[c_index] = x_index
            current_state_index[t_index] = x_index+1
            x_index = x_index + 2

    meas_index = []
    for meas_str in meas_string:
        if meas_str != '1':
            meas_index.append([x_index,makeState1q(meas_str)])
            x_index = x_index + 1

    x_len = x_index

    def cost_function(x):
        Gamma_list = []
        for x_index in range(x_len):
            Gamma_list.append(x2Gamma(x[8*x_index:8*(x_index+1)]))
        neg = 1.
        for state_index in Init_state_index:
            rho = state_index[1]
            Gamma = Gamma_list[state_index[0]]
            neg = neg*neg_state_1q(rho,Gamma)
        for gate_index in gate_1q_index:
            U1q = gate_index[1]
            Gamma_in = Gamma_list[(gate_index[0])[0]]
            Gamma_out = Gamma_list[(gate_index[0])[1]]
            neg = neg*neg_gate_1q(U1q,Gamma_in,Gamma_out)
        for gate_index in gate_2q_index:
            U2q = gate_index[1]
            GammaC_in = Gamma_list[(gate_index[0])[0]]
            GammaT_in = Gamma_list[(gate_index[0])[1]]
            GammaC_out = Gamma_list[(gate_index[0])[2]]
            GammaT_out = Gamma_list[(gate_index[0])[3]]
            neg = neg*neg_gate_Cliff_2q(U2q,GammaC_in,GammaT_in,GammaC_out,GammaT_out)
        for m_index in meas_index:
            E = m_index[1]
            Gamma = Gamma_list[m_index[0]]
            neg = neg*neg_meas_1q(E,Gamma)
        return np.log(neg)

    print(Init_state_index)
    print(gate_1q_index)
    print(gate_2q_index)
    print(meas_index)
    print('----------------------------')

    x0 = []
    for x_index in range(x_len):
        x0 = np.append(x0,[1,0,0,0,0,0,1,0])
    t0 = time.time()
    out = cost_function(x0)
    dt = time.time() - t0
    print('Wigner Log negativity:', out, '(Computation time:',dt,')')

#    x0 = 2*np.random.rand(8*x_len)-1

    '''Optimization with autograd'''
    if opt_method=='B':
        grad_cost_function = grad(cost_function)
        def func(x):
            return cost_function(x), grad_cost_function(x)
        t0=time.time()
        optimize_result = basinhopping(func, x0, minimizer_kwargs={"method":"L-BFGS-B", "jac":True}, disp=False, niter=1)
        dt = time.time()-t0
#    '''Optimization without autograd (Powell)'''
#    elif opt_method=='NG':
#        start_time = time.time()
#        optimize_result = minimize(cost_function, x0, method='Powell')
#        end_time = time.time()
#    '''Optimization without autograd (G)'''
#    elif opt_method=='G':
#        start_time = time.time()
#        optimize_result = opt.minimize(cost_function, x0, method='L-BFGS-B', jac=grad_cost_function, bounds=bnds, options={'disp': show_log})
#        end_time = time.time()

    '''Show results'''
    optimized_x = optimize_result.x
    optimized_value = cost_function(optimized_x)
    print(optimized_x)
    print('Optimized Log negativity:\t',optimized_value)
    print('Computation time:', dt)

    directory = os.path.join('data',path)
    if not os.path.isdir(directory):    os.mkdir(directory)
    np.save(os.path.join('data',path, 'state_string.npy'), state_string)
    np.save(os.path.join('data',path, 'gate_sequence.npy'), gate_sequence)
    np.save(os.path.join('data',path, 'meas_string.npy'), meas_string)
    np.save(os.path.join('data',path, 'optimized_x.npy'), optimized_x)
    np.save(os.path.join('data',path, 'optimized_neg.npy'), optimized_value)

    return optimized_x, optimized_value