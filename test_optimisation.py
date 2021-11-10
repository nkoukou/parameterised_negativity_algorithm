import autograd.numpy as np
import time
import os
from autograd import(grad)

from qubit_circuit_generator import haar_random_connected_circuit
from compression import compress_circuit
from frame_opt import (init_x_list, get_negativity_circuit, sequential_para_opt)
from phase_space import PhaseSpace
from qubit_frame_Pauli import (F, G, x0)

Pauli = PhaseSpace(F, G, x0, DIM=2)
W = [Pauli.W_state, Pauli.W_gate, Pauli.W_meas]

N = 4
L = 8
gate_size = 2
circuit_haar = haar_random_connected_circuit(N, L, gate_size, d=2, given_state=None, given_meas=1, method='c')

main_path = 'Data_for_optimisation/Data_N%d_L%d'%(N,L)

###### Data
print('================= Haar-random circuit negativity reduction =================')
circuit = circuit_haar.copy()
x_circuit = init_x_list(circuit, x0)
neg_init = get_negativity_circuit(W, circuit, x_circuit)
print('Initial log-negativity:', np.log(neg_init))

n=2
compressed_circuit = compress_circuit(circuit_haar, n)
# print(len(compressed_circuit['index_list']))
print('------------------------- With n =',n,'-------------------------')
l=1
circuit = compressed_circuit.copy()
x_circuit = init_x_list(circuit, x0)
t0 = time.time()
x_out,neg_list_n2_l1 = sequential_para_opt(W,circuit,x_circuit,l=l,niter=3)
print('Optimized log-negativity (n=2, l=1):',np.log(neg_list_n2_l1[-1]))
print('Computation time:', time.time()-t0)
file_name = 'neg_list_n%d_l%d.npy'%(n,l)
path = os.path.join(main_path, file_name)
with open(path, 'wb') as f:
	np.save(f, np.array(neg_list_n2_l1))

l=2
circuit = compressed_circuit.copy()
x_circuit = init_x_list(circuit, x0)
t0 = time.time()
x_out,neg_list_n2_l2 = sequential_para_opt(W,circuit,x_circuit,l=l,niter=3)
print('Optimized log-negativity (n=2, l=2):',np.log(neg_list_n2_l2[-1]))
print('Computation time:', time.time()-t0)
file_name = 'neg_list_n%d_l%d.npy'%(n,l)
path = os.path.join(main_path, file_name)
with open(path, 'wb') as f:
	np.save(f, np.array(neg_list_n2_l2))

l=3
circuit = compressed_circuit.copy()
x_circuit = init_x_list(circuit, x0)
t0 = time.time()
x_out,neg_list_n2_l3 = sequential_para_opt(W,circuit,x_circuit,l=l,niter=3)
print('Optimized log-negativity (n=2, l=3):',np.log(neg_list_n2_l3[-1]))
print('Computation time:', time.time()-t0)
file_name = 'neg_list_n%d_l%d.npy'%(n,l)
path = os.path.join(main_path, file_name)
with open(path, 'wb') as f:
	np.save(f, np.array(neg_list_n2_l3))

l=4
circuit = compressed_circuit.copy()
x_circuit = init_x_list(circuit, x0)
t0 = time.time()
x_out,neg_list_n2_l4 = sequential_para_opt(W,circuit,x_circuit,l=l,niter=3)
print('Optimized log-negativity (n=2, l=4):',np.log(neg_list_n2_l4[-1]))
print('Computation time:', time.time()-t0)
file_name = 'neg_list_n%d_l%d.npy'%(n,l)
path = os.path.join(main_path, file_name)
with open(path, 'wb') as f:
	np.save(f, np.array(neg_list_n2_l4))

l=5
circuit = compressed_circuit.copy()
x_circuit = init_x_list(circuit, x0)
t0 = time.time()
x_out,neg_list_n2_l5 = sequential_para_opt(W,circuit,x_circuit,l=l,niter=3)
print('Optimized log-negativity (n=2, l=5):',np.log(neg_list_n2_l5[-1]))
print('Computation time:', time.time()-t0)
file_name = 'neg_list_n%d_l%d.npy'%(n,l)
path = os.path.join(main_path, file_name)
with open(path, 'wb') as f:
	np.save(f, np.array(neg_list_n2_l5))

#########
n=3
compressed_circuit = compress_circuit(circuit_haar, n)
# print(len(compressed_circuit['index_list']))
print('------------------------- With n =',n,'-------------------------')
l=1
circuit = compressed_circuit.copy()
x_circuit = init_x_list(circuit, x0)
t0 = time.time()
x_out,neg_list_n3_l1 = sequential_para_opt(W,circuit,x_circuit,l=l,niter=3)
print('Optimized log-negativity (n=3, l=1):',np.log(neg_list_n3_l1[-1]))
print('Computation time:', time.time()-t0)
file_name = 'neg_list_n%d_l%d.npy'%(n,l)
path = os.path.join(main_path, file_name)
with open(path, 'wb') as f:
	np.save(f, np.array(neg_list_n3_l1))

l=2
circuit = compressed_circuit.copy()
x_circuit = init_x_list(circuit, x0)
t0 = time.time()
x_out,neg_list_n3_l2 = sequential_para_opt(W,circuit,x_circuit,l=l,niter=3)
print('Optimized log-negativity (n=3, l=2):',np.log(neg_list_n3_l2[-1]))
print('Computation time:', time.time()-t0)
file_name = 'neg_list_n%d_l%d.npy'%(n,l)
path = os.path.join(main_path, file_name)
with open(path, 'wb') as f:
	np.save(f, np.array(neg_list_n3_l2))

l=3
circuit = compressed_circuit.copy()
x_circuit = init_x_list(circuit, x0)
t0 = time.time()
x_out,neg_list_n3_l3 = sequential_para_opt(W,circuit,x_circuit,l=l,niter=3)
print('Optimized log-negativity (n=3, l=3):',np.log(neg_list_n3_l3[-1]))
print('Computation time:', time.time()-t0)
file_name = 'neg_list_n%d_l%d.npy'%(n,l)
path = os.path.join(main_path, file_name)
with open(path, 'wb') as f:
	np.save(f, np.array(neg_list_n3_l3))

l=4
circuit = compressed_circuit.copy()
x_circuit = init_x_list(circuit, x0)
t0 = time.time()
x_out,neg_list_n3_l4 = sequential_para_opt(W,circuit,x_circuit,l=l,niter=3)
print('Optimized log-negativity (n=3, l=4):',np.log(neg_list_n3_l4[-1]))
print('Computation time:', time.time()-t0)
file_name = 'neg_list_n%d_l%d.npy'%(n,l)
path = os.path.join(main_path, file_name)
with open(path, 'wb') as f:
	np.save(f, np.array(neg_list_n3_l4))

l=5
circuit = compressed_circuit.copy()
x_circuit = init_x_list(circuit, x0)
t0 = time.time()
x_out,neg_list_n3_l5 = sequential_para_opt(W,circuit,x_circuit,l=l,niter=3)
print('Optimized log-negativity (n=3, l=5):',np.log(neg_list_n3_l5[-1]))
print('Computation time:', time.time()-t0)
file_name = 'neg_list_n%d_l%d.npy'%(n,l)
path = os.path.join(main_path, file_name)
with open(path, 'wb') as f:
	np.save(f, np.array(neg_list_n3_l5))

######
n=4
compressed_circuit = compress_circuit(circuit_haar, n)
# print(len(compressed_circuit['index_list']))
print('------------------------- With n =',n,'-------------------------')
l=1
circuit = compressed_circuit.copy()
x_circuit = init_x_list(circuit, x0)
t0 = time.time()
x_out,neg_list_n4_l1 = sequential_para_opt(W,circuit,x_circuit,l=l,niter=3)
print('Optimized log-negativity (n=4, l=1):',np.log(neg_list_n4_l1[-1]))
print('Computation time:', time.time()-t0)
file_name = 'neg_list_n%d_l%d.npy'%(n,l)
path = os.path.join(main_path, file_name)
with open(path, 'wb') as f:
	np.save(f, np.array(neg_list_n4_l1))

l=2
circuit = compressed_circuit.copy()
x_circuit = init_x_list(circuit, x0)
t0 = time.time()
x_out,neg_list_n4_l2 = sequential_para_opt(W,circuit,x_circuit,l=l,niter=3)
print('Optimized log-negativity (n=4, l=2):',np.log(neg_list_n4_l2[-1]))
print('Computation time:', time.time()-t0)
file_name = 'neg_list_n%d_l%d.npy'%(n,l)
path = os.path.join(main_path, file_name)
with open(path, 'wb') as f:
	np.save(f, np.array(neg_list_n4_l2))

l=3
circuit = compressed_circuit.copy()
x_circuit = init_x_list(circuit, x0)
t0 = time.time()
x_out,neg_list_n4_l3 = sequential_para_opt(W,circuit,x_circuit,l=l,niter=3)
print('Optimized log-negativity (n=4, l=3):',np.log(neg_list_n4_l3[-1]))
print('Computation time:', time.time()-t0)
file_name = 'neg_list_n%d_l%d.npy'%(n,l)
path = os.path.join(main_path, file_name)
with open(path, 'wb') as f:
	np.save(f, np.array(neg_list_n4_l3))

l=4
circuit = compressed_circuit.copy()
x_circuit = init_x_list(circuit, x0)
t0 = time.time()
x_out,neg_list_n4_l4 = sequential_para_opt(W,circuit,x_circuit,l=l,niter=3)
print('Optimized log-negativity (n=4, l=4):',np.log(neg_list_n4_l4[-1]))
print('Computation time:', time.time()-t0)
file_name = 'neg_list_n%d_l%d.npy'%(n,l)
path = os.path.join(main_path, file_name)
with open(path, 'wb') as f:
	np.save(f, np.array(neg_list_n4_l4))

l=5
circuit = compressed_circuit.copy()
x_circuit = init_x_list(circuit, x0)
t0 = time.time()
x_out,neg_list_n4_l5 = sequential_para_opt(W,circuit,x_circuit,l=l,niter=3)
print('Optimized log-negativity (n=4, l=5):',np.log(neg_list_n4_l5[-1]))
print('Computation time:', time.time()-t0)
file_name = 'neg_list_n%d_l%d.npy'%(n,l)
path = os.path.join(main_path, file_name)
with open(path, 'wb') as f:
	np.save(f, np.array(neg_list_n4_l5))