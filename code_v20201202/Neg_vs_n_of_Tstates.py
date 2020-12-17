import autograd.numpy as np
import itertools
from gate_seq2symplectic \
    import gate_sequence2symplectic_form_merged, symplectic_inverse
import neg_circuit as nc
from random_circuit_generator import random_circuit_string
import matplotlib.pyplot as plt

import scipy.optimize as opt
import time
from autograd import grad
from scipy.optimize import Bounds
from scipy.optimize import basinhopping

DIM = 3

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

