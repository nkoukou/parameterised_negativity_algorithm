################################ SAMPLE CODE ##################################
import time
import autograd.numpy as np
import matplotlib.pylab as plt
from QUBIT_QD_circuit import QD_circuit
from QUBIT_circuit_generator import (random_connected_circuit, random_circuit,
       compress2q_circuit, compress3q_circuit, string_to_circuit,
       show_connectivity, solve_qubit_circuit, random_connected_circuit_2q3q)

sample_size = int(1e6)
x_list = np.linspace(1, sample_size, sample_size)

circuit, Tcount = random_connected_circuit(3, 7, Tgate_prob=0.24, given_state=None, given_measurement=3)
# circuit, Tcount, toffoli_num = random_connected_circuit_2q3q(6, 25, Tgate_prob=1/3, prob_2q=1, given_state=None, given_measurement=4)


print('\n_______________________No-compression_______________________')
circ = QD_circuit(circuit)
circ.show_connectivity(compressed=False)
pborn1 = solve_qubit_circuit(circ.circuit)
print("Born Probs (no-compress):", pborn1)


print('\n_______________________2q-compression_______________________')
circ.compress_circuit(m=2)
circ.show_connectivity(compressed=True)
pborn2 = solve_qubit_circuit(circ.circuit_compressed)
print("Born Probs (2q-compress):", pborn2)

circ.opt_x(method='Wigner')
circ.get_QD_list(method='Wigner')
wigner_out_list = circ.sample(method='Wigner', sample_size=sample_size)
prob_wigner = np.cumsum(wigner_out_list)/x_list

circ.opt_x(method = 'Local_opt')
circ.get_QD_list(method='Local_opt')
local_opt_out_list = circ.sample(method='Local_opt',sample_size=sample_size)
prob_local_opt = np.cumsum(local_opt_out_list)/x_list

circ.opt_x(method='Opt', **{'niter':10})
circ.get_QD_list(method = 'Opt')
opt_out_list = circ.sample(method='Opt', sample_size=sample_size)
prob_opt = np.cumsum(opt_out_list)/x_list


print('\n_______________________3q-compression_______________________')
circ.compress_circuit(m=3)
circ.show_connectivity(compressed=True)
pborn3 = solve_qubit_circuit(circ.circuit_compressed)
print("Born Probs (3q-compress):", pborn3)
circ.opt_x_3q(method = 'Wigner')
compressed_out_list = circ.sample(method='Wigner', m=3, sample_size=sample_size)
prob_compressed_wigner = np.cumsum(compressed_out_list)/x_list

circ.opt_x_3q(method = 'Local_opt')
compressed_local_opt_out_list = circ.sample(method='Local_opt', m=3, sample_size=sample_size)
prob_compressed_local_opt = np.cumsum(compressed_local_opt_out_list)/x_list

circ.opt_x_3q(method = 'Opt')
compressed_opt_out_list = circ.sample(method='Opt', m=3, sample_size=sample_size)
prob_compressed_opt = np.cumsum(compressed_opt_out_list)/x_list



plt.plot(x_list, prob_wigner, linestyle='dotted',color='tab:blue', label='Wigner')
plt.plot(x_list, prob_opt, linestyle='dotted',color='tab:green', label='Optimised')
plt.plot(x_list, prob_local_opt, linestyle='dotted',color='tab:red', label='Local_Optimised')

plt.plot(x_list, prob_compressed_wigner, linestyle='solid',color='tab:purple', label='3q_compressed_Wigner')
plt.plot(x_list, prob_compressed_opt, linestyle='solid',color='y', label='3q_compressed_Optimised')
plt.plot(x_list, prob_compressed_local_opt, linestyle='solid',color='tab:orange', label='3q_compressed_Local_Optimised')


plt.hlines(y=pborn1, xmin=0, xmax= sample_size, color='k', ls='-',label='Born Rule Prob')
plt.xlabel('sample #')
plt.ylabel('p_estimate')
plt.xlim(0,sample_size)
plt.ylim(-0.2,1.2)
plt.legend(loc='upper right')
plt.show()