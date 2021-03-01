import numpy as np
from state_functions import(evolve)
from circuit_components import(makeGate, makeState)
from random_circuit_generator import(random_circuit, compress_circuit,
                                     show_circuit, solve_circuit_symbolic)
from opt_neg import(optimize_neg, optimize_neg_compressed)
from prob_estimation import(sample, compare_Wigner_optimised)
from local_opt_neg import (get_local_opt_x,show_Wigner_neg_x)

# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


circuit = random_circuit(qudit_num=6,
                         C1qGate_num=203, TGate_num=30, CSUMGate_num=13,
                         given_state='000000', given_measurement='00////',
                         symbolic=True)
#circuit = ['001', [[[0],'H'], [[2],'H'], [[2],'H'],
#                                      [[1,2],'C+'], [[2],'t'], [[0,2],'C+'],
#                                      [[2],'T'], [[1,2],'C+'], [[2],'t'],
#                                      [[0,2],'C+'], [[1],'T'], [[2],'T'],
#                                      [[0,1],'C+'], [[2],'H'], [[0],'T'],
#                                      [[1],'t'], [[0,1],'C+'], [[0],'H']],
#                              '0//']
show_circuit(circuit)
p_born = solve_circuit_symbolic(circuit)
print("Born probabiity (Exact):", p_born)

cc = compress_circuit(circuit)
show_Wigner_neg_x(cc,**{'show_detailed_log': True})
kwargs = {'show_detailed_log': True, 'opt_method':'B', 'niter': 1}
local_opt_x, local_opt_neg = get_local_opt_x(cc,**kwargs)
opt_x, opt_neg = optimize_neg_compressed(cc)

niters = 1000000
Wigner = sample(circuit, 0, niters)
p_sample_Wigner, plot_Wigner = Wigner.MC_sampling()

local_Optimised = sample(circuit, local_opt_x, niters)
p_sample_local_opt, plot_local_opt = local_Optimised.MC_sampling()

Optimised = sample(circuit, opt_x, niters)
p_sample_opt, plot_opt = Optimised.MC_sampling()

x_axis = np.arange(niters)
plt.plot(x_axis, np.ones(len(x_axis))*p_born, linestyle='solid', color='k', label='Born Prob.')
plt.plot(x_axis, plot_Wigner, linestyle='solid', color='tab:blue', label='Wigner')
plt.plot(x_axis, plot_opt, linestyle='solid', color='tab:orange', label='Optimised')
plt.plot(x_axis, plot_local_opt, linestyle='solid', color='tab:red', label='Local_Optimised')
plt.xlabel("number of iterations")
plt.ylabel("p_estimate")
plt.legend()
plt.show()
