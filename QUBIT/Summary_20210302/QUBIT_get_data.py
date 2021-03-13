from QUBIT_BVcircuit import(BValg_circuit)
from QUBIT_QD_circuit import(QD_circuit)
from QUBIT_circuit_generator import(random_connected_circuit)
import autograd.numpy as np
import time
import os

# path = 'random_circuit_compare_neg.txt'

# with open(path, 'w') as f:
#     f.write("n_connectivity, n_T_gates, Wigner_neg, Opt_neg, Local_opt_neg\n")
# f.close()

n_qubit = 10
min_n_CNOT = n_qubit-1
max_n_CNOT = n_qubit*2


for n_CNOT in range(min_n_CNOT, max_n_CNOT+1):
    Wigner_neg_list = []
    Opt_neg_list = []
    Local_opt_neg_list = []
    CompTime_list = []

    for n_Tgate in range(2*n_CNOT+1):
        
        t0 = time.time()
        circuit, Tcount = random_connected_circuit(qudit_num=n_qubit, circuit_length=n_CNOT, 
            Tgate_prob=(n_Tgate/(2*n_CNOT)), given_state=None, given_measurement=2, method='r')
        
        circ = QD_circuit(circuit)
        circ.compress_circuit()

        Wigner_x_list, Wigner_neg = circ.opt_x(method='Wigner')
        Opt_x_list, Opt_neg = circ.opt_x(method='Opt', **{'niter':10})
        Local_opt_x_list, Local_opt_neg = circ.opt_x(method='Local_opt', **{'niter':10})

        Wigner_neg_list.append([Tcount, Wigner_neg])
        Opt_neg_list.append([Tcount, Opt_neg])
        Local_opt_neg_list.append([Tcount, Local_opt_neg])
        CompTime_list.append([Tcount, (time.time()-t0)])

    Wigner_file_name = 'Wigner_neg_Q'+str(n_qubit)+'_CNOT'+str(n_CNOT)+'.npy'
    Opt_file_name = 'Opt_neg_Q'+str(n_qubit)+'_CNOT'+str(n_CNOT)+'.npy'
    Local_opt_file_name = 'Local_opt_neg_Q'+str(n_qubit)+'_CNOT'+str(n_CNOT)+'.npy'
    ComTime_file_name = 'ComTime_Q'+str(n_qubit)+'_CNOT'+str(n_CNOT)+'.npy'

    np.save(os.path.join('data', Wigner_file_name), Wigner_neg_list)
    np.save(os.path.join('data', Opt_file_name), Opt_neg_list)
    np.save(os.path.join('data', Local_opt_file_name), Local_opt_neg_list)
    np.save(os.path.join('data', ComTime_file_name), CompTime_list)



### Save the results
# with open(path, 'a') as f:
#     for n in range(len(Wigner_neg_list)):
#         f.write(str(n_connectivity_list[n])+" "+str(n_Tgate_list[n])+" "+str(Wigner_neg_list[n])
#         	+" "+str(Opt_neg_list[n])+" "+str(Local_opt_neg_list[n]))
#         f.write("\n")
# f.close()

