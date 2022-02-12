import numpy as np
import os
import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

plt.rcParams['figure.dpi'] = 200
plt.style.use('classic')
plt.rc('font',   size=24)
plt.rc('axes',   labelsize=25)
plt.rc('xtick',  labelsize=21)
plt.rc('ytick',  labelsize=21)
plt.rc('legend',  fontsize=18)
plt.rc('lines',  linewidth=2 )
plt.rc('lines', markersize=10 )
plt.rc('lines', markeredgewidth=0.)

N = 6
L = 15

main_path = os.path.join('data_optimisation', 'Data_N6_L15')

file_name = 'neg_list_n2_l1.npy'
path = os.path.join(main_path, file_name)
with open(path, 'rb') as f:
	neg_list_n2_l1 = np.load(f)
file_name = 'neg_list_n2_l2.npy'
path = os.path.join(main_path, file_name)
with open(path, 'rb') as f:
	neg_list_n2_l2 = np.load(f)
file_name = 'neg_list_n2_l3.npy'
path = os.path.join(main_path, file_name)
with open(path, 'rb') as f:
	neg_list_n2_l3 = np.load(f)
file_name = 'neg_list_n2_l4.npy'
path = os.path.join(main_path, file_name)
with open(path, 'rb') as f:
	neg_list_n2_l4 = np.load(f)
file_name = 'neg_list_n2_l5.npy'
path = os.path.join(main_path, file_name)
with open(path, 'rb') as f:
	neg_list_n2_l5 = np.load(f)

file_name = 'neg_list_n3_l1.npy'
path = os.path.join(main_path, file_name)
with open(path, 'rb') as f:
	neg_list_n3_l1 = np.load(f)
file_name = 'neg_list_n3_l2.npy'
path = os.path.join(main_path, file_name)
with open(path, 'rb') as f:
	neg_list_n3_l2 = np.load(f)
file_name = 'neg_list_n3_l3.npy'
path = os.path.join(main_path, file_name)
with open(path, 'rb') as f:
	neg_list_n3_l3 = np.load(f)
file_name = 'neg_list_n3_l4.npy'
path = os.path.join(main_path, file_name)
with open(path, 'rb') as f:
	neg_list_n3_l4 = np.load(f)
file_name = 'neg_list_n3_l5.npy'
path = os.path.join(main_path, file_name)
with open(path, 'rb') as f:
	neg_list_n3_l5 = np.load(f)

file_name = 'neg_list_n4_l1.npy'
path = os.path.join(main_path, file_name)
with open(path, 'rb') as f:
	neg_list_n4_l1 = np.load(f)
file_name = 'neg_list_n4_l2.npy'
path = os.path.join(main_path, file_name)
with open(path, 'rb') as f:
	neg_list_n4_l2 = np.load(f)
file_name = 'neg_list_n4_l3.npy'
path = os.path.join(main_path, file_name)
with open(path, 'rb') as f:
	neg_list_n4_l3 = np.load(f)
file_name = 'neg_list_n4_l4.npy'
path = os.path.join(main_path, file_name)
with open(path, 'rb') as f:
	neg_list_n4_l4 = np.load(f)
file_name = 'neg_list_n4_l5.npy'
path = os.path.join(main_path, file_name)
with open(path, 'rb') as f:
	neg_list_n4_l5 = np.load(f)

# print("The initial negativity:", np.log2(neg_list_n2_l1[0]))
# print("The lowest with n=2:", np.log2(neg_list_n2_l5[-1]))
# print("The lowest with n=3:", np.log2(neg_list_n3_l5[-1]))
# print("The lowest with n=3:", np.log2(neg_list_n4_l5[-1]))

# plt.title('Haar-random circuit with N='+str(N)+' L='+str(L))

plt.close('all')

figure(figsize=(9,6))

plt.plot(np.log2(neg_list_n2_l1), label='$\ell$ = 1', marker='o', color='tab:blue')
plt.plot(np.log2(neg_list_n2_l2), label='$\ell$ = 2', marker='^', color='tab:orange')
plt.plot(np.log2(neg_list_n2_l3), label='$\ell$ = 3', marker='v', color='tab:green')
plt.plot(np.log2(neg_list_n2_l4), label='$\ell$ = 4', marker='s', color='tab:red')
plt.plot(np.log2(neg_list_n2_l5), label='$\ell$ = 5', marker='D', color='tab:purple')

plt.plot(np.log2(neg_list_n3_l1), marker='o', color='tab:blue')
plt.plot(np.log2(neg_list_n3_l2), marker='^', color='tab:orange')
plt.plot(np.log2(neg_list_n3_l3), marker='v', color='tab:green')
plt.plot(np.log2(neg_list_n3_l4), marker='s', color='tab:red')
plt.plot(np.log2(neg_list_n3_l5), marker='D', color='tab:purple')

plt.plot(np.log2(neg_list_n4_l1), marker='o', color='tab:blue')
plt.plot(np.log2(neg_list_n4_l2), marker='^', color='tab:orange')
plt.plot(np.log2(neg_list_n4_l3), marker='v', color='tab:green')
plt.plot(np.log2(neg_list_n4_l4), marker='s', color='tab:red')
plt.plot(np.log2(neg_list_n4_l5), marker='D', color='tab:purple')

plt.plot([np.log2(neg_list_n2_l1[0])]*(len(neg_list_n2_l1)+1), ':', color='tab:grey')
plt.plot([np.log2(neg_list_n3_l1[0])]*(len(neg_list_n2_l1)+1), '--', color='tab:grey')
plt.plot([np.log2(neg_list_n4_l1[0])]*(len(neg_list_n2_l1)+1), '-.', color='tab:grey')

plt.legend(loc='lower left', ncol=2)
plt.xlabel(r'Optimisation cycles $c$')
plt.ylabel(r'Circuit negativity $\log{N_C({\cal{G}}_{\rm{opt}})}$')
plt.xlim([0,37])
plt.ylim([16,32.1])
plt.show()



'''
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

main_path = 'data_optimisation/Data_N%d_L%d'%(N,L)

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
'''







