from QD_circuit import(QD_circuit)
from circuit_generator import(random_connected_circuit)
import autograd.numpy as np
import time
import os
import matplotlib.pylab as plt


n_qubit = 5
n_CNOT = 4
n_Tgate = 5

circuit, Tcount = random_connected_circuit(qudit_num=n_qubit,
                                           circuit_length=n_CNOT,
    Tgate_prob =(n_Tgate/(2*n_CNOT)), given_state=None,
                 given_measurement=2, method='c')


# circ = QD_circuit(circuit)
# circ.compress_circuit()

# Wigner_x_list, Wigner_neg = circ.opt_x(method='Wigner')
# Opt_x_list, Opt_neg = circ.opt_x(method='Opt', **{'niter':10})
# Local_opt_x_list, Local_opt_neg = circ.opt_x(method='Local_opt',
#                                              **{'niter':10})

# print("Wigner: ", Wigner_neg)
# print("Opt: ", Opt_neg)
# print("Local opt: ", Local_opt_neg)


circ = QD_circuit(circuit)
circ.compress_circuit()
circ.show_connectivity()
print()
circ.show_connectivity(compressed=False)

sample_size = int(1e5)
x_list = np.linspace(1, sample_size, sample_size)

circ.opt_x(method='Wigner')
circ.get_QD_list(method='Wigner')
wigner_out_list = circ.sample(method='Wigner', sample_size=sample_size)
prob_wigner = np.cumsum(wigner_out_list)/x_list

circ.opt_x(method='Opt', **{'niter':1})
circ.get_QD_list(method = 'Opt')
opt_out_list = circ.sample(method='Opt', sample_size=sample_size)
prob_opt = np.cumsum(opt_out_list)/x_list

circ.opt_x(method='Local_opt', **{'niter':1})
circ.get_QD_list(method='Local_opt')
local_opt_out_list = circ.sample(method='Local_opt',
                                  sample_size=sample_size)
prob_local_opt = np.cumsum(local_opt_out_list)/x_list


plt.close('all')
plt.plot(x_list, prob_wigner, linestyle='solid',
          color='tab:blue', label='Wigner')
plt.plot(x_list, prob_opt, linestyle='solid',
          color='tab:orange', label='Optimised')
plt.plot(x_list, prob_local_opt, linestyle='solid',
          color='tab:red', label='Local_Optimised')
plt.xlabel('sample #')
plt.ylabel('p_estimate')
plt.legend(loc='upper right')
plt.show()

