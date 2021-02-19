import numpy as np
from state_functions import DIM
from circuit_components import(makeState, makeGate, makeCsum)
from local_opt_neg import (get_rand_circuit, show_Wigner_neg_x, get_opt_x, get_circuit_loc)
from random_circuit_generator import(random_circuit, compress_circuit,show_circuit)

'''Sample Circuit'''
print('Test Sample Circuit')
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
kwargs = {'opt_method':'B', 'niter': 1, 'show_detailed_log': True}
wig_neg_tot = show_Wigner_neg_x(circuit,**kwargs)
x_rho_opt_list, x_gate_out_opt_list, x_meas_opt_list, neg_rho_opt_list, neg_gate_opt_list, neg_meas_opt_list, neg_tot = get_opt_x(circuit,**kwargs)
print('--------------------------------')
print('Wig_neg/Opt_neg:',wig_neg_tot/neg_tot)
print('--------------------------------')

'''Compressed Circuit'''
print('\n')
print('\n')
print('Test Compressed Circuit')
circuit = ['011', [
            [[2], 'H'],
            [[0], 'S'],
            [[1], 'T'],
            [[0], 'T'],
            [[2, 0], 'C+'],
            [[1], 'H'],
            [[1], 'S'],
            [[0], 'T'],
            [[2, 0], 'C+'],
            [[0], 'S'],
            [[1], 'T'],
            [[0], 'H'],
            [[1], 'S'],
            [[1, 2], 'C+'],
            [[1], 'H']
            ], '1T/']
show_circuit(circuit)
circuit_compressed = compress_circuit(circuit)
#show_circuit(circuit_compressed)
    
circuit_loc = get_circuit_loc(circuit_compressed)
kwargs = {'opt_method':'B', 'niter': 1, 'show_detailed_log': False}
wig_neg_tot = show_Wigner_neg_x(circuit_loc,**kwargs)
x_rho_opt_list, x_gate_out_opt_list, x_meas_opt_list, neg_rho_opt_list, neg_gate_opt_list, neg_meas_opt_list, neg_tot = get_opt_x(circuit_loc,**kwargs)
print('--------------------------------')
print('Wig_neg/Opt_neg:',wig_neg_tot/neg_tot)
print('--------------------------------')


'''Random Circuit'''
print('\n')
print('\n')
print('Test Random Circuit')
kwargs = {'type': 'c', 'Tgate_prob': 0.9}
qudit_num = 10
circuit_length = 20
circuit = get_rand_circuit(qudit_num,circuit_length,**kwargs)

kwargs = {'opt_method':'B', 'niter': 3, 'show_detailed_log': False}
wig_neg_tot = show_Wigner_neg_x(circuit,**kwargs)
x_rho_opt_list, x_gate_out_opt_list, x_meas_opt_list, neg_rho_opt_list, neg_gate_opt_list, neg_meas_opt_list, neg_tot = get_opt_x(circuit,**kwargs)
print('--------------------------------')
print('Wig_neg/Opt_neg:',wig_neg_tot/neg_tot)
print('--------------------------------')
