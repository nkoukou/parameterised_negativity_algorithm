from QUBIT_QD_circuit import(QD_circuit)
from QUBIT_circuit_generator import(random_connected_circuit,
                                    solve_qubit_circuit)
import autograd.numpy as np
import time
import os

def get_data(q_rng, samples, direc='test', qcts=(0,0,0,0)):
    Q, C, T, S = qcts
    for n_qubit in range(q_rng[0], q_rng[1]):
        if n_qubit < Q: continue
        for n_CNOT in [n_qubit, n_qubit+1]:
            if (n_qubit==Q and n_CNOT < C): continue
            for n_Tgate in range(2*n_CNOT+1):
                if (n_qubit==Q and n_CNOT==C and n_Tgate < T): continue
                for num in range(samples):
                    if (n_qubit==Q and n_CNOT==C and n_Tgate==T and num<S):
                        continue
                    print('\n')
                    print('================================================')
                    print('================================================')
                    print('================================================')
                    print('\n')
                    print('STEP: Q %d || CNOT %d || T %d / %d || S %d / %d'%(
                        n_qubit, n_CNOT, n_Tgate, 2*n_CNOT, num+1, samples))

                    t0 = time.time()
                    circuit, TC = random_connected_circuit(qudit_num=n_qubit,
                        circuit_length=n_CNOT, Tgate_prob=n_Tgate,
                        given_state=None, given_measurement=2, method='c')

                    circ = QD_circuit(circuit)
                    pborn1 = solve_qubit_circuit(circ.circuit)
                    circ.compress_circuit(m=2)
                    pborn2 = solve_qubit_circuit(circ.circuit_compressed)
                    print()
                    print("Probs-2q:", np.allclose(pborn1, pborn2),
                      "(%.4f, %.4f)"%(pborn1, pborn2), "\n")
                    if not np.allclose(pborn1, pborn2):
                        raise Exception('(2q-compression) Probs: NOT equal')

                    WI_x, WI_neg = circ.opt_x(method='Wigner')
                    LO_x, LO_neg = circ.opt_x(method='Local_opt',
                                              **{'niter':10})
                    CT = (time.time()-t0)

                    fname = 'Q'+str(n_qubit)+'_CNOT'+str(n_CNOT)+\
                            '_T'+str(n_Tgate)+'_s'+str(num)+'.npy'
                    WI_fname = 'WI_'+fname
                    LO_fname = 'LO_'+fname
                    CT_fname = 'CT_'+fname
                    np.save(os.path.join(direc, WI_fname), WI_neg)
                    np.save(os.path.join(direc, LO_fname), LO_neg)
                    np.save(os.path.join(direc, CT_fname), CT)
get_data(q_rng=(0,15), samples=10, direc='test')

# direc = 'test'

# n_qubit = 10
# min_n_CNOT = n_qubit-1
# max_n_CNOT = n_qubit*2

# t0 = time.time()
# for n_CNOT in range(min_n_CNOT, max_n_CNOT+1):
#     Wigner_neg_list_2, Wigner_neg_list_3 = [], []
#     Opt_neg_list_2, Opt_neg_list_3 = [], []
#     Local_opt_neg_list_2, Local_opt_neg_list_3 = [], []
#     CompTime_list_2, CompTime_list_3 = [], []

#     for n_Tgate in range(2*n_CNOT+1):
#         print('\n')
#         print('===========================================================')
#         print('===========================================================')
#         print('===========================================================')
#         print('\n')
#         print('STEP: T %d / %d || CNOT %d / %d'%(n_Tgate,2*n_CNOT+1,
#                                                  n_CNOT, max_n_CNOT))

#         t0 = time.time()
#         circuit, Tcount = random_connected_circuit(
#             qudit_num=n_qubit, circuit_length=n_CNOT,
#             Tgate_prob =(n_Tgate/(2*n_CNOT)),
#             given_state=None, given_measurement=2, method='c')

#         circ = QD_circuit(circuit)
#         pborn1 = solve_qubit_circuit(circ.circuit)

#         circ.compress_circuit(m=2)
#         pborn2 = solve_qubit_circuit(circ.circuit_compressed)
#         print()
#         print("Probs-2q:", np.allclose(pborn1, pborn2),
#           "(%.4f, %.4f)"%(pborn1, pborn2))
#         if not np.allclose(pborn1, pborn2):
#             raise Exception('(2q-compression) Probs: NOT equal')

#         Wigner_x_list, Wigner_neg = circ.opt_x(method='Wigner')
#         Opt_x_list, Opt_neg = circ.opt_x(method='Opt', **{'niter':10})
#         Local_opt_x_list, Local_opt_neg = circ.opt_x(method='Local_opt',
#                                                      **{'niter':10})

#         Wigner_neg_list_2.append([Tcount, Wigner_neg])
#         Opt_neg_list_2.append([Tcount, Opt_neg])
#         Local_opt_neg_list_2.append([Tcount, Local_opt_neg])
#         CompTime_list_2.append([Tcount, (time.time()-t0)])

#         # circ.compress_circuit(m=3)
#         # pborn3 = solve_qubit_circuit(circ.circuit_compressed)
#         # print("Probs-3q:", np.allclose(pborn1, pborn3),
#         #   "(%.4f, %.4f)"%(pborn1, pborn3))
#         # print()
#         # if not np.allclose(pborn1, pborn3):
#         #     raise Exception('(3q-compression) Probs: NOT equal')

#         # Wigner_x_list, Wigner_neg = circ.opt_x(method='Wigner', m=3)
#         # Opt_x_list, Opt_neg = circ.opt_x(method='Opt', m=3, **{'niter':10})
#         # Local_opt_x_list, Local_opt_neg = circ.opt_x(method='Local_opt',
#         #                                              m=3, **{'niter':10})

#         # Wigner_neg_list_3.append([Tcount, Wigner_neg])
#         # Opt_neg_list_3.append([Tcount, Opt_neg])
#         # Local_opt_neg_list_3.append([Tcount, Local_opt_neg])
#         # CompTime_list_3.append([Tcount, (time.time()-t0)])

#     Wigner_file_name = 'Wigner_neg_Q'+str(n_qubit)+'_CNOT'+str(n_CNOT)+\
#                        '_m2.npy'
#     Opt_file_name = 'Opt_neg_Q'+str(n_qubit)+'_CNOT'+str(n_CNOT)+'_m2.npy'
#     Local_opt_file_name = 'Local_opt_neg_Q'+str(n_qubit)+'_CNOT'+\
#                           str(n_CNOT)+'_m2.npy'
#     CTime_file_name='ComTime_Q'+str(n_qubit)+'_CNOT'+str(n_CNOT)+'_m2.npy'

#     np.save(os.path.join(direc, Wigner_file_name), Wigner_neg_list_2)
#     np.save(os.path.join(direc, Opt_file_name), Opt_neg_list_2)
#     np.save(os.path.join(direc, Local_opt_file_name), Local_opt_neg_list_2)
#     np.save(os.path.join(direc, CTime_file_name), CompTime_list_2)

#     # Wigner_file_name = 'Wigner_neg_Q'+str(n_qubit)+'_CNOT'+str(n_CNOT)+\
#     #                    '_m3.npy'
#     # Opt_file_name = 'Opt_neg_Q'+str(n_qubit)+'_CNOT'+str(n_CNOT)+'_m3.npy'
#     # Local_opt_file_name = 'Local_opt_neg_Q'+str(n_qubit)+'_CNOT'+\
#     #                       str(n_CNOT)+'_m3.npy'
#     # CTime_file_name='ComTime_Q'+str(n_qubit)+'_CNOT'+str(n_CNOT)+'_m3.npy'

#     # np.save(os.path.join(direc, Wigner_file_name), Wigner_neg_list_3)
#     # np.save(os.path.join(direc, Opt_file_name), Opt_neg_list_3)
#     # np.save(os.path.join(direc, Local_opt_file_name), Local_opt_neg_list_3)
#     # np.save(os.path.join(direc, CTime_file_name), CompTime_list_3)

# t1 = time.time()
# print(t1-t0)