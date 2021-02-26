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


#circuit = random_circuit(qudit_num=3,
#                         C1qGate_num=4, TGate_num=1, CSUMGate_num=1,
#                         given_state=None, given_measurement=1,
#                         symbolic=True)
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
             [[1, 2], 'C+']
             ], '1T/']
show_circuit(circuit)

cc = compress_circuit(circuit)
niters = 100000

Wigner = sample(cc, 0, niters)
p_sample_Wigner, plot_Wigner = Wigner.MC_sampling()

kwargs = {'show_detailed_log': False, 'opt_method':'B', 'niter': 3}
#show_Wigner_neg_x(cc,**kwargs)
local_opt_x, local_opt_neg = get_local_opt_x(cc,**kwargs)
local_Optimised = sample(cc, local_opt_x, niters)
p_sample_local_opt, plot_local_opt = local_Optimised.MC_sampling()

Optimised = sample(cc, 1, niters)
p_sample_opt, plot_opt = Optimised.MC_sampling()

x_axis = np.arange(niters)
plt.plot(x_axis, plot_Wigner, linestyle='solid', color='tab:blue', label='Wigner')
plt.plot(x_axis, plot_opt, linestyle='solid', color='tab:orange', label='Optimised')
plt.plot(x_axis, plot_local_opt, linestyle='solid', color='tab:red', label='Local_Optimised')
plt.xlabel("number of iterations")
plt.ylabel("p_estimate")
plt.legend()
plt.show()
