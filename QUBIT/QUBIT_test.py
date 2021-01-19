import numpy as np
from QUBIT_circuit_components import(makeGate)
from QUBIT_random_circuit_generator import(random_circuit, show_circuit)
from QUBIT_opt_neg import(optimize_neg)
from QUBIT_prob_estimation import(sample,sample_iter)

# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt


circuit = ['000', [[[0], 'X'],[[1], 'H'],[[1,2], 'C+'],[[0],'X']], '000']
# print(circuit)
# print(optimize_neg(circuit))
#sample_iter(circuit)
show_circuit(circuit)
result_Wigner = sample(circuit)
print('Using Wigner: ', result_Wigner)
opt_Gammas, Gamma_dist = optimize_neg(circuit)
result_Gamma = sample(circuit, opt_Gammas)
print('Using optimised QP: ', result_Gamma)


