from QUBIT_QD_circuit import(QD_circuit)
from QUBIT_circuit_generator import(random_connected_circuit,
                                    solve_qubit_circuit)
import autograd.numpy as np
import time
import os

direc = 'data_Q10_m23'

n_qubit = 10
min_n_CNOT = n_qubit-1
max_n_CNOT = n_qubit*2


for n_CNOT in range(min_n_CNOT, max_n_CNOT+1):
    Wigner_neg_list_2, Wigner_neg_list_3 = [], []
    Opt_neg_list_2, Opt_neg_list_3 = [], []
    Local_opt_neg_list_2, Local_opt_neg_list_3 = [], []
    CompTime_list_2, CompTime_list_3 = [], []

    for n_Tgate in range(2*n_CNOT+1):
        print('\n')
        print('===========================================================')
        print('===========================================================')
        print('===========================================================')
        print('\n')
        print('STEP: T %d / %d || CNOT %d / %d'%(n_Tgate,2*n_CNOT+1,
                                                 n_CNOT, max_n_CNOT))

        t0 = time.time()
        circuit, Tcount = random_connected_circuit(
            qudit_num=n_qubit, circuit_length=n_CNOT,
            Tgate_prob =(n_Tgate/(2*n_CNOT)),
            given_state=None, given_measurement=2, method='c')

        circ = QD_circuit(circuit)
        pborn1 = solve_qubit_circuit(circ.circuit)

        circ.compress_circuit(m=2)
        pborn2 = solve_qubit_circuit(circ.circuit_compressed)
        print()
        print("Probs-2q:", np.allclose(pborn1, pborn2),
          "(%.4f, %.4f)"%(pborn1, pborn2))
        if not np.allclose(pborn1, pborn2):
            raise Exception('(2q-compression) Probs: NOT equal')

        Wigner_x_list, Wigner_neg = circ.opt_x(method='Wigner')
        Opt_x_list, Opt_neg = circ.opt_x(method='Opt', **{'niter':10})
        Local_opt_x_list, Local_opt_neg = circ.opt_x(method='Local_opt',
                                                     **{'niter':10})

        Wigner_neg_list_2.append([Tcount, Wigner_neg])
        Opt_neg_list_2.append([Tcount, Opt_neg])
        Local_opt_neg_list_2.append([Tcount, Local_opt_neg])
        CompTime_list_2.append([Tcount, (time.time()-t0)])

        circ.compress_circuit(m=3)
        pborn3 = solve_qubit_circuit(circ.circuit_compressed)
        print("Probs-3q:", np.allclose(pborn1, pborn3),
          "(%.4f, %.4f)"%(pborn1, pborn3))
        print()
        if not np.allclose(pborn1, pborn3):
            raise Exception('(3q-compression) Probs: NOT equal')

        Wigner_x_list, Wigner_neg = circ.opt_x(method='Wigner', m=3)
        Opt_x_list, Opt_neg = circ.opt_x(method='Opt', m=3, **{'niter':10})
        Local_opt_x_list, Local_opt_neg = circ.opt_x(method='Local_opt', m=3,
                                                     **{'niter':10})

        Wigner_neg_list_3.append([Tcount, Wigner_neg])
        Opt_neg_list_3.append([Tcount, Opt_neg])
        Local_opt_neg_list_3.append([Tcount, Local_opt_neg])
        CompTime_list_3.append([Tcount, (time.time()-t0)])

    Wigner_file_name = 'Wigner_neg_Q'+str(n_qubit)+'_CNOT'+str(n_CNOT)+\
                       '_m2.npy'
    Opt_file_name = 'Opt_neg_Q'+str(n_qubit)+'_CNOT'+str(n_CNOT)+'_m2.npy'
    Local_opt_file_name = 'Local_opt_neg_Q'+str(n_qubit)+'_CNOT'+\
                          str(n_CNOT)+'_m2.npy'
    ComTime_file_name = 'ComTime_Q'+str(n_qubit)+'_CNOT'+str(n_CNOT)+'_m2.npy'

    np.save(os.path.join(direc, Wigner_file_name), Wigner_neg_list_2)
    np.save(os.path.join(direc, Opt_file_name), Opt_neg_list_2)
    np.save(os.path.join(direc, Local_opt_file_name), Local_opt_neg_list_2)
    np.save(os.path.join(direc, ComTime_file_name), CompTime_list_2)

    Wigner_file_name = 'Wigner_neg_Q'+str(n_qubit)+'_CNOT'+str(n_CNOT)+\
                       '_m3.npy'
    Opt_file_name = 'Opt_neg_Q'+str(n_qubit)+'_CNOT'+str(n_CNOT)+'_m3.npy'
    Local_opt_file_name = 'Local_opt_neg_Q'+str(n_qubit)+'_CNOT'+\
                          str(n_CNOT)+'_m3.npy'
    ComTime_file_name = 'ComTime_Q'+str(n_qubit)+'_CNOT'+str(n_CNOT)+'_m3.npy'

    np.save(os.path.join(direc, Wigner_file_name), Wigner_neg_list_3)
    np.save(os.path.join(direc, Opt_file_name), Opt_neg_list_3)
    np.save(os.path.join(direc, Local_opt_file_name), Local_opt_neg_list_3)
    np.save(os.path.join(direc, ComTime_file_name), CompTime_list_3)

