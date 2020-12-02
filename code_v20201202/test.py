import numpy as np
import itertools
from gate_seq2symplectic \
    import gate_sequence2symplectic_form_merged, symplectic_inverse
from quasi_dist import W_state_list_1q, W_meas_list_1q
from make_state import makeState1q, makeMeas1q
from neg_circuit import opt_neg_tot, show_neg_result
from random_sample import sample_circuit, accum_mean
from random_circuit_generator import random_circuit_string
from calc_born_prob import simulate_circuit

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


import scipy.optimize as opt
import time

DIM = 3

# '''Circuit:      [state]   [       gate_sequence       ]  [measurement]  '''
# circuit_string = ['TS10NT1','1SS11H1', '1SH1HSH', '1CT111S','1C11TSH',
#                   'C1TH11S','1T11HCS','1111CTS','TSS111C','11Z1111']
# circuit_string = ['TS0N', 'T1C1', '1TC1', '1T1C', '1SSH', 'H111', '1C1T',
#                   '111Z']
circuit_string = random_circuit_string(n=4, L=10, given_state=None,
                                       given_measurement=None, p_csum=0.6)

# circuit_string = ['1NSN','1HH1', '1TC1', 'HS1S', 'H111', 'CT11', 'T1C1', \
#                   '1HSH', 'HHSH', '1C1T', 'HSSS', '1SHS', 'CT11', 'CT11',\
#                   '11HH', 'SS1S','111T']
#
# circuit_string = ['0S', 'TC', '1H', '1X']
# circuit_string = ['0S', 'CT', 'HS', '1Z']

# circuit_string = ['N','H', 'H', '1', 'H', 'H', 'H', 'H', 'H', 'S', 'S','H',\
#                   'H', 'S', 'S', 'S', 'S', 'S', 'H', 'S', 'S', '1', 'H','H',\
#                   '1', 'S', 'H', 'S', 'S', 'H', 'H', '1', '1', 'H', 'H','S',\
#                   '1', 'S', 'S', '1', 'H', 'S', 'H', '1', 'H', '1', 'S','H',\
#                   'H', 'H', 'S','T']

print('Calculating Born probability...')
if len(circuit_string[0])>7:
    print('\n' + '!!! The number of qudits is higher than 7 - it is too hard \
          to calculate exact probability... setting p = -0.5 \n')
    pb = -0.5
else:
    pb = simulate_circuit(circuit_string)

state_string = circuit_string[0]
qudit_num = len(state_string)
gate_sequence = circuit_string[1:-1]
Stot, ztot = gate_sequence2symplectic_form_merged(gate_sequence)
print('Sympletic matrix (Stot):\n', Stot)
print('Displacment (z):\t', ztot%DIM)
S = symplectic_inverse(Stot)
print('Displacment (Stot.z):\t', np.dot(Stot,ztot)%DIM)
print('Displacment (S.z):\t', np.dot(S,ztot)%DIM)
meas_string = circuit_string[-1]
show_neg_result(circuit_string)
gamma_opt, neg_opt = opt_neg_tot(circuit_string, show_log=False)
show_neg_result(circuit_string, gamma_opt)

sample_size = 20000
gamma = np.zeros(2*qudit_num)
sample_list = sample_circuit(circuit_string,gamma,sample_size)
print('Wigner sampling average: ', np.mean(sample_list))

sample_list_opt = sample_circuit(circuit_string,gamma_opt,sample_size)
print('Optimized Quasi-Dist. Sampling:', np.mean(sample_list_opt))
print('Born probability: ', pb)

plt.close('all')
plt.figure()
x = np.array(range(sample_size))+1
plt.plot(
x, np.repeat(np.array(pb), len(x)), 'r',
x, accum_mean(sample_list), '-.k',
x, accum_mean(sample_list_opt), '-.b'
)
plt.legend(['Born probability', 'Discrete Wigner Sampling',
            'Optimized Quasi-Dist. Sampling'],
           loc='upper right')
#plt.xscale("log")
plt.xlabel('Number of samples collected')
plt.ylabel('Estimation value')
plt.show()
