import numpy as np
from state_functions import DIM
from circuit_components import(makeState, makeGate, makeCsum)
from phase_space import(D1q_list,x2Gamma, neg_state_1q, neg_gate_1q_max,
                   neg_gate_2q_max, neg_meas_1q, W_state_1q, W_gate_1q, W_gate_2q, W_meas_1q)
from local_opt_neg import (get_rand_circuit, show_Wigner_neg_x, get_opt_x)

'''Sample Code1'''
rho_a = makeState('+')
rho_b = makeState('+')
rho_c = makeState('+')
rho_d = makeState('+')

U_ab = np.dot(np.kron(makeGate('T'),makeGate('Z')),makeCsum('C+'))
U_bc = np.dot(np.kron(makeGate('S'),makeGate('T')),makeCsum('C+'))
U_cd = np.dot(np.kron(makeGate('X'),np.eye(DIM)),makeCsum('C+'))
U_ad = np.dot(np.kron(makeGate('T'),makeGate('T')),makeCsum('C+'))

index1 = [0,1]
index2 = [1,2]
index3 = [2,3]
index4 = [3,0]

E_a = makeState('0')
E_b = np.eye(DIM)
E_c = np.eye(DIM)
E_d = np.eye(DIM)

rho_list = [rho_a,rho_b,rho_c,rho_d]
gate_U2q_list = [U_ab,U_bc,U_cd,U_ad]
gate_qudit_index_list = [index1,index2,index3,index4]
meas_list = [E_a,E_b,E_c,E_d]

circuit = [rho_list,gate_U2q_list,gate_qudit_index_list,meas_list]
print('----------Wigner--------------')
wig_neg_tot = show_Wigner_neg_x(circuit)

kwargs = {'opt_method':'B', 'niter': 1}
print('----------Opt_Dist--------------',kwargs)
x_rho_opt_list, x_gate_out_opt_list, x_meas_opt_list, neg_rho_opt_list, neg_gate_opt_list, neg_meas_opt_list, neg_tot = get_opt_x(circuit,**kwargs)
print('--------------------------------')
print('Wig_neg/Opt_neg',wig_neg_tot/neg_tot)
print('--------------------------------')

'''Random Code'''
kwargs = {'type': 'c', 'Tgate_prob': 0.9}
qudit_num = 10
circuit_length = 20
circuit = get_rand_circuit(qudit_num,circuit_length,**kwargs)
print('----------Wigner--------------')
wig_neg_tot = show_Wigner_neg_x(circuit)

kwargs = {'opt_method':'B', 'niter': 3}
print('----------Opt_Dist2--------------',kwargs)
x_rho_opt_list, x_gate_out_opt_list, x_meas_opt_list, neg_rho_opt_list, neg_gate_opt_list, neg_meas_opt_list, neg_tot = get_opt_x(circuit,**kwargs)
print('--------------------------------')
print('Wig_neg/Opt_neg',wig_neg_tot/neg_tot)
print('--------------------------------')
