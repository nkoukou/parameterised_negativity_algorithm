import autograd.numpy as np
import itertools
from gate_seq2symplectic \
    import gate_sequence2symplectic_form_merged, symplectic_inverse
import neg_circuit as nc
#from random_circuit_generator import random_circuit_string
import matplotlib.pyplot as plt

import scipy.optimize as opt
import time
from autograd import grad
from scipy.optimize import Bounds
from scipy.optimize import basinhopping

DIM = 3

def random_circuit_string(n, L, n_Tstates, given_measurement=None,
                          p_csum=0.6):
    ''' Creates a random circuit string in the form
        ['state', 'gate1', 'gate2', ..., 'gateL', 'measurement'].
        n - integer - number of qudits
        L - integer - circuit depth (number of gate levels)
    '''
    # state_string
    
    char1q = ['0', '1', '2'] # Can also add: '+'
    prob1q = [1/len(char1q)]*len(char1q)

    state_string = ''
    for i in range(n_Tstates):
        state_string += 'T'
    for i in range(n-n_Tstates):
        state_string += nr.choice(char1q, p=prob1q)

    # gate_string
    prob = [p_csum, 1-p_csum] # Probability of 1-qudit gate or 2-qudit gate
    char1q = ['H', 'S', '1']
    prob1q = [1/len(char1q)]*len(char1q)

    gates_string = []
    for i in range(L):
        gate_string = ''
        q1 = nr.choice([1,0], p=prob)
        if q1:
            for j in range(n):
                gate_string += nr.choice(char1q, p=prob1q)
        else:
            gate = ['1']*n
            c = nr.choice(np.arange(n))
            t = c
            while t==c:
                t = nr.choice(np.arange(n))
            gate[c] = 'C'
            gate[t] = 'T'
            for g in gate:
                gate_string += g
        gates_string.append(gate_string)

    # measurement_string
    if given_measurement is None:
        char1q = ['X', 'Z', 'T']
        prob1q = [0., 0., 1.] # [1/len(char1q)]*len(char1q)
        basis = nr.choice(char1q, p=prob1q)

        measurement = ['1']*n
        basis_index = nr.choice(np.arange(n))
        measurement[basis_index] = basis

        measurement_string = ''
        for m in measurement:
            measurement_string += m
    else:
        if len(given_measurement)!=n: raise Exception('Number of qubits is n')
        measurement_string = given_measurement
    circuit_string = [state_string] + gates_string + [measurement_string]
    return circuit_string

path = 'simulation_Tstates.txt'

for i in range(100):
	for n_T in range(101):
		circuit_string = random_circuit_string(n=100, L=300, n_Tstates=n_T, given_measurement=None, p_csum=0.6)
		state_string = circuit_string[0]
		gate_sequence = circuit_string[1:-1]
		meas_string = circuit_string[-1]
		qudit_num = len(state_string)

		gamma0 = np.zeros(2*qudit_num)

		Stot, ztot = gate_sequence2symplectic_form_merged(gate_sequence)
		S = symplectic_inverse(Stot)

		def cost_function(x):
			return nc.neg_state(state_string,x) * nc.neg_meas(meas_string, x, S)
		grad_cost_function = grad(cost_function)
		def func(x):
			return cost_function(x), grad_cost_function(x)

		x0 = 2.*np.random.rand(2*qudit_num)-1.
		start_time = time.time()
		optimize_result = basinhopping(func, x0, minimizer_kwargs={"method":"L-BFGS-B", "jac":True}, disp=False, niter=0)
		end_time = time.time()

		Wigner_value = cost_function(gamma0)
		optimized_value = optimize_result.fun

		with open(path, 'a') as f:
			f.write(str(n_T)+", "+str(optimized_value)+", "+str(Wigner_value)+", "+str(end_time-start_time))
			f.write("\n")

f.close()

