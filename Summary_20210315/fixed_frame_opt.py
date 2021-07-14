from QD_circuit import(QD_circuit)
# from circuit_generator import(random_connected_circuit)
# from frame_opt import(wigner_neg_compressed, wigner_neg_compressed_3q,
#         optimize_neg_compressed, optimize_neg_compressed_3q,
#         local_opt_neg_compressed, local_opt_neg_compressed_3q)
import autograd.numpy as np
import time
import os
import itertools as it
from autograd import(grad)
from scipy.optimize import(Bounds, minimize)
from scipy.optimize import(basinhopping)
from circuit_components import(makeState, makeGate, makeCsum)
from circuit_generator import(random_connected_circuit, show_connectivity)
from phase_space import(x2Gamma, neg_gate_1q_max, neg_gate_2q_max, neg_meas_1q,
                       neg_state_1q, neg_gate_1q_max)
from frame_opt import(wigner_neg_compressed)

def fixed_opt_state(state, set_of_frames):
#     neg_list = [neg_state_1q(state, x2Gamma(set_of_frames[i])) for i in range(len(set_of_frames))]
#     neg = min(neg_list)
#     x_index = neg_list.index(neg)
#     print("Neg_list: ", neg_list)
#     print("Min index: ", x_index)
    neg = neg_state_1q(state, x2Gamma(set_of_frames[0]))
    return neg, 0

def fixed_opt_gate2q(U2q, x_in1, x_in2, set_of_frames):
    neg_list = [neg_gate_2q_max(U2q, x2Gamma(x_in1), x2Gamma(x_in2), x2Gamma(set_of_frames[i]), 
                                x2Gamma(set_of_frames[i])) for i,j in it.product(range(len(set_of_frames)), repeat=2)]
    neg = min(neg_list)
    index_total = neg_list.index(neg)
    x_out1_index = index_total//len(set_of_frames)
    x_out2_index = index_total%len(set_of_frames)
    return neg, x_out1_index, x_out2_index
    

def fixed_opt_neg(compressed_circuit, set_of_frames):
    t0 = time.time()
    n_frames = len(set_of_frames)
    rho_list = compressed_circuit['state_list']
    gate_U2q_list = compressed_circuit['gate_list']
    gate_qudit_index_list = compressed_circuit['index_list']
    meas_list = compressed_circuit['meas_list']

    x_rho_opt_list = []
    neg_rho_opt_list = []
    neg_tot = 1.
    for rho in rho_list:
        neg, x_opt_index = fixed_opt_state(rho, set_of_frames)
        x_rho_opt_list.append(set_of_frames[x_opt_index])
        neg_rho_opt_list.append(neg)
        neg_tot *= neg

    x_running = x_rho_opt_list.copy()
    x_gate_out_opt_list = []
    neg_gate_opt_list = []
    for gate_index in range(len(gate_U2q_list)):
        U2q = gate_U2q_list[gate_index]
        [qudit_index1, qudit_index2] = gate_qudit_index_list[gate_index]
        x_in1 = x_running[qudit_index1]
        x_in2 = x_running[qudit_index2]
        neg, x_out_opt1_index, x_out_opt2_index = fixed_opt_gate2q(U2q, x_in1, x_in2, set_of_frames)
        x_gate_out_opt_list.append([set_of_frames[x_out_opt1_index],set_of_frames[x_out_opt2_index]].copy())
        neg_gate_opt_list.append(neg)
        neg_tot *= neg
        x_running[qudit_index1] = set_of_frames[x_out_opt1_index]
        x_running[qudit_index2] = set_of_frames[x_out_opt2_index]

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

    print('--------------------- FIXED OPTIMIZATION ---------------------')
    print('Fixed Opt Log Neg:', np.log(neg_tot))
    print('Computation time: ',time.time() - t0)
    print('--------------------------------------------------------------')

    x_opt_list_tot = np.append(np.array(x_rho_opt_list).flatten(),np.array(x_gate_out_opt_list).flatten())
    return x_opt_list_tot, np.log(neg_tot)

def find_good_frames(circuit, n_frames):
    x0 = [1,0,0,0,0,0,1,0]
    rho_list = circuit['state_list']
    gate_U2q_list = circuit['gate_list']
    gate_qudit_index_list = circuit['index_list']
    meas_list = circuit['meas_list']
    
    def cost_function(x):
        set_of_frames = [x0]
        n_frames = (len(x)//8) + 1
        for n in range(0, n_frames-1):
            x_n = x[8*n:8*(n+1)]
            set_of_frames.append(x_n)
        
        x_rho_opt_list = []
        neg_tot = 1.
        for rho in rho_list:
            neg = neg_state_1q(rho, x2Gamma(x0))
            x_rho_opt_list.append(x0)
            neg_tot *= neg
        
        x_running = x_rho_opt_list
        for gate_index in range(len(gate_U2q_list)):
            U2q = gate_U2q_list[gate_index]
            [qudit_index1, qudit_index2] = gate_qudit_index_list[gate_index]
            x_in1 = x_running[qudit_index1]
            x_in2 = x_running[qudit_index2]
            neg, x_out_opt1_index, x_out_opt2_index = fixed_opt_gate2q(U2q, x_in1, x_in2, set_of_frames)
            neg_tot *= neg
            x_running[qudit_index1] = set_of_frames[x_out_opt1_index]
            x_running[qudit_index2] = set_of_frames[x_out_opt2_index]
        
        x_meas_opt_list = x_running
        qudit_index = 0
        for meas in meas_list:
            if str(meas) == '/': 
                qudit_index += 1
                continue
            neg = neg_meas_1q(meas, x2Gamma(x_meas_opt_list[qudit_index]))
            neg_tot *= neg
            qudit_index += 1
            
        return neg_tot
    
    x_ini = 2*np.random.rand(8*(n_frames-1))-1
    grad_cost_function = grad(cost_function)
    def func(x):
        return cost_function(x), grad_cost_function(x)
    
    optimize_result = basinhopping(func, x_ini, minimizer_kwargs={"method":"L-BFGS-B","jac":True}, niter=5)
    optimized_x = optimize_result.x
    optimized_value = np.log(cost_function(optimized_x))
    
    print('------------- Local Opt with ', n_frames, 'Frames ------------')
    print('Optimized Log Neg:', optimized_value)
    print('Frame 0 : ', x0)
    for n in range(n_frames-1):
        print('Frame', n+1, ': ', optimized_x[n*8:(n+1)*8])
    print('--------------------------------------------------------------')

    return optimized_x, optimized_value
    

x0 = [1,0,0,0,0,0,1,0]
x1 = [-0.    ,  0.766 ,  0.6428 , 0.   ,  -0.  ,   -0.   ,   0.   ,  -0.    ]
x2 = [-0.648,  -0.0394,  0.881 ,  0.383 ,  0.675 ,  0.2606, -0.898 ,  0.8949]
set_of_2frames = [x0,x1]
set_of_3frames = [x0,x1,x2]

# path = 'random_circuit_compare_neg.txt'

# with open(path, 'w') as f:
#     f.write("n_connectivity, n_T_gates, Wigner_neg, Opt_neg,\
#             Local_opt_neg\n")
# f.close()

n_qudit = 20
# min_n_CNOT = n_qudit-1
# max_n_CNOT = n_qudit*2
n_CNOT = 80
# n_Tgate = 15

Wigner_neg_list = []
# Opt_neg_list_2q = []
Fixed_opt_neg_list_2frames = []
Fixed_opt_neg_list_3frames = []
# CompTime_list_2q = []

# Wigner_neg_list_3q = []
# CompTime_list_3q = []

for n_Tgate in range(2*n_CNOT+1):

    # for n_Tgate in range(2*n_CNOT+1):
    for i in range(3):
        circuit, Tcount = random_connected_circuit(qudit_num=n_qudit,circuit_length=n_CNOT,
                                                    Tgate_prob =(n_Tgate/(2*n_CNOT)), given_state=None,
                                                    given_measurement=10, method='c')

        Wigner_x_list, Wigner_neg = wigner_neg_compressed(circuit)
        Fixed_opt_x_list_2frames, Fixed_opt_neg_2frames = fixed_opt_neg(circuit, set_of_2frames)
        Fixed_opt_x_list_3frames, Fixed_opt_neg_3frames = fixed_opt_neg(circuit, set_of_3frames)

        Wigner_neg_list.append([Tcount, Wigner_neg])
        Fixed_opt_neg_list_2frames.append([Tcount, Fixed_opt_neg_2frames])
        Fixed_opt_neg_list_3frames.append([Tcount, Fixed_opt_neg_3frames])


Wigner_file_name = 'New_Wigner_neg_Q'+str(n_qudit)+'_CNOT'+str(n_CNOT)+'.txt'
Fixed_2frames_opt_file_name = 'New_Fixed_opt_neg_2frames_Q'+str(n_qudit)+'_CNOT'+str(n_CNOT)+'.txt'
Fixed_3frames_opt_file_name = 'New_Fixed_opt_neg_3frames_Q'+str(n_qudit)+'_CNOT'+str(n_CNOT)+'.txt'

with open(Wigner_file_name, 'a') as f:
    for n in range(len(Wigner_neg_list)):
        f.write(str(Wigner_neg_list[n][0])+" "+str(Wigner_neg_list[n][1])+"\n")
f.close()

with open(Fixed_2frames_opt_file_name, 'a') as f:
    for n in range(len(Fixed_opt_neg_list_2frames)):
        f.write(str(Fixed_opt_neg_list_2frames[n][0])+" "+str(Fixed_opt_neg_list_2frames[n][1])+"\n")
f.close()

with open(Fixed_3frames_opt_file_name, 'a') as f:
    for n in range(len(Fixed_opt_neg_list_3frames)):
        f.write(str(Fixed_opt_neg_list_3frames[n][0])+" "+str(Fixed_opt_neg_list_3frames[n][1])+"\n")
f.close()

