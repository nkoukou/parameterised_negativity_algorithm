import numpy as np
import matplotlib.pylab as plt
from QUBIT_circuit_components import(makeGate)
from QUBIT_random_circuit_generator import(random_circuit, show_circuit)
from QUBIT_opt_neg import(optimize_neg)
from QUBIT_prob_estimation import(sample,sample_iter,compare_Wigner_para)

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
                              '1//']

#plt.close('all')
#show_circuit(Bernstein_Vazirani_circuit)
compare_Wigner_para(Bernstein_Vazirani_circuit, int(1e5))
# result_Wigner = sample(Bernstein_Vazirani_circuit)
# print('Using Wigner: ', result_Wigner)
#optimize_neg(circuit)
# result_Gamma = sample(Bernstein_Vazirani_circuit, opt_Gammas)
# print('Using optimised QP: ', result_Gamma)


