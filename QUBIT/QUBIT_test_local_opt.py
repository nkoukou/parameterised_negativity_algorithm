import numpy as np
import matplotlib.pylab as plt
from QUBIT_circuit_components import(makeGate)
from QUBIT_random_circuit_generator import(random_circuit, show_circuit,
                                           compress_circuit)
from QUBIT_opt_neg import(optimize_neg, optimize_neg_compressed)
from QUBIT_prob_estimation import(sample,compare_Wigner_optimised)
from QUBIT_local_opt_neg import (get_local_opt_x,show_Wigner_neg_x)

# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt


#circuit = ['000', [[[0], 'X'],[[0], 'H'],[[0,1], 'C+'],[[2],'H']], '00/']

### s = 0, 1-qubit case
Bernstein_Vazirani_circuit = ['001', [[[0],'H'], [[2],'H'], [[2],'H'],
                                      [[1,2],'C+'], [[2],'t'], [[0,2],'C+'],
                                      [[2],'T'], [[1,2],'C+'], [[2],'t'],
                                      [[0,2],'C+'], [[1],'T'], [[2],'T'],
                                      [[0,1],'C+'], [[2],'H'], [[0],'T'],
                                      [[1],'t'], [[0,1],'C+'], [[0],'H']],
                              '0//']
cc = compress_circuit(Bernstein_Vazirani_circuit)

niters = 1000000

Wigner = sample(Bernstein_Vazirani_circuit, 0, niters)
p_sample_Wigner, plot_Wigner = Wigner.MC_sampling()

kwargs = {'show_detailed_log': False, 'opt_method':'B', 'niter': 3}
#show_Wigner_neg_x(cc,**kwargs)
local_opt_x, local_opt_neg = get_local_opt_x(cc,**kwargs)
local_Optimised = sample(cc, local_opt_x, niters)
p_sample_local_opt, plot_local_opt = local_Optimised.MC_sampling()

Optimised = sample(Bernstein_Vazirani_circuit, 1, niters)
p_sample_opt, plot_opt = Optimised.MC_sampling()

x_axis = np.arange(niters)
plt.plot(x_axis, plot_Wigner, linestyle='solid', color='tab:blue', label='Wigner')
plt.plot(x_axis, plot_opt, linestyle='solid', color='tab:orange', label='Optimised')
plt.plot(x_axis, plot_local_opt, linestyle='solid', color='tab:red', label='Local_Optimised')
plt.xlabel("number of iterations")
plt.ylabel("p_estimate")
plt.legend()
plt.show()



######### New 'prob_estimation.py' example ##########
### When you just want to run sampling once
#method = 0
# method type
# 0: Wigner, 
# 1: run optimisation and use the optimised parameters, 
# or x_list: when you want to run sampling with a particular parameter list.
#test_sampling = sample(Bernstein_Vazirani_circuit, method, 500000)
#p_esimate, p_estimate_list = test_sampling.MC_sampling()
# return the sampling result (p_estimate) and the full list of p_estimate of each iteration

### When you want to plot the difference between the Wigner and the optmised cases (it also saves the full list of each case):
#compare_Wigner_optimised(Bernstein_Vazirani_circuit, 500000)
#######################################################

#plt.close('all')
# show_circuit(Bernstein_Vazirani_circuit)
# circuit_compressed = compress_circuit(Bernstein_Vazirani_circuit)
# res = optimize_neg_compressed(circuit_compressed)
# compare_Wigner_para(Bernstein_Vazirani_circuit, int(1e5))
# result_Wigner = sample(Bernstein_Vazirani_circuit)
# print('Using Wigner: ', result_Wigner)
#optimize_neg(circuit)
# result_Gamma = sample(Bernstein_Vazirani_circuit, opt_Gammas)
# print('Using optimised QP: ', result_Gamma)


