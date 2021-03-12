from QUBIT_BVcircuit import(BValg_circuit)
from QUBIT_QD_circuit import(QD_circuit)
from QUBIT_circuit_generator import(random_connected_circuit)
import autograd.numpy as np

path = 'random_circuit_compare_neg.txt'

with open(path, 'w') as f:
    f.write("n_connectivity, n_T_gates, Wigner_neg, Opt_neg, Local_opt_neg\n")
f.close()

n_qubit = 5
min_n_CNOT = n_qubit-1
max_n_CNOT = 10

n_connectivity_list = []
n_Tgate_list = []
Wigner_neg_list = []
Opt_neg_list = []
Local_opt_neg_list = []

for n_CNOT in range(min_n_CNOT, max_n_CNOT+1):
    for n_Tgate in range(2*n_CNOT+1):
        circuit, Tcount = random_connected_circuit(qudit_num=n_qubit, circuit_length=n_CNOT, 
            Tgate_prob=(n_Tgate/(2*n_CNOT)), given_state=None, given_measurement=2, method='r')
        # print(circuit)
        circ = QD_circuit(circuit)
        circ.compress_circuit()
        # print(circ.circuit_compressed)

        Wigner_x_list, Wigner_neg = circ.opt_x(method='Wigner')
        Opt_x_list, Opt_neg = circ.opt_x(method='Opt', **{'niter':10})
        Local_opt_x_list, Local_opt_neg = circ.opt_x(method='Local_opt', **{'niter':10})

        n_connectivity_list.append(n_CNOT)
        n_Tgate_list.append(Tcount)
        Wigner_neg_list.append(Wigner_neg)
        Opt_neg_list.append(Opt_neg)
        Local_opt_neg_list.append(Local_opt_neg)


### Save the results
with open(path, 'a') as f:
    for n in range(len(Wigner_neg_list)):
        f.write(str(n_connectivity_list[n])+" "+str(n_Tgate_list[n])+" "+str(Wigner_neg_list[n])
        	+" "+str(Opt_neg_list[n])+" "+str(Local_opt_neg_list[n]))
        f.write("\n")
f.close()

