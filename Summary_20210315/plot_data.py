import numpy as np
import matplotlib.pylab as plt
import os

path = 'data_q6'
data_names = ['Wigner_neg_Q', 'Opt_neg_Q', 'Local_opt_neg_Q', 'ComTime_Q']
n_qudit = 6

def load_data(data_name, n_qudit, comp=2):
    min_n_CNOT = n_qudit-1
    max_n_CNOT = n_qudit*2

    T_list = []
    CNOT_list = []
    neg_list = []

    for n_CNOT in range(min_n_CNOT, max_n_CNOT+1):
        fname = data_name+str(n_qudit)+'_CNOT'+str(n_CNOT)+'_'+str(comp)+'q.txt'
        fname = os.path.join(path, fname)
        f = open(fname, 'r')
        lines = f.read().splitlines()
        for line in lines:
            T_list.append(float(line.split(',')[0]))
            CNOT_list.append(n_CNOT)
            neg_list.append(float(line.split(',')[1]))
        f.close()

    return T_list, CNOT_list, neg_list


Wigner_2q_T, Wigner_2q_CNOT, Wigner_2q = load_data(data_names[0], n_qudit, comp=2)
Opt_2q_T, Opt_2q_CNOT, Opt_2q = load_data(data_names[1], n_qudit, comp=2)
Local_2q_T, Local_2q_CNOT, Local_2q = load_data(data_names[2], n_qudit, comp=2)
Wigner_3q_T, Wigner_3q_CNOT, Wigner_3q = load_data(data_names[0], n_qudit, comp=3)

ComTime_2q_T, ComTime_2q_CNOT, ComTime_2q = load_data(data_names[3], n_qudit, comp=2)
ComTime_3q_T, ComTime_3q_CNOT, ComTime_3q = load_data(data_names[3], n_qudit, comp=3)


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(Wigner_2q_T, Wigner_2q_CNOT, Wigner_2q, marker='.')
ax.scatter(Opt_2q_T, Opt_2q_CNOT, Opt_2q, marker='.')
ax.scatter(Local_2q_T, Local_2q_CNOT, Local_2q, marker='.')
ax.scatter(Wigner_3q_T, Wigner_3q_CNOT, Wigner_3q, marker='.')
ax.set_xlabel('The number of T-gates')
ax.set_ylabel('The number of CNOT-gates')
ax.set_zlabel('Negativity')
ax.legend(['Wigner_2q', 'Global_opt_2q', 'Local_opt_2q', 'Wigner_3q'])
ax.set_title('6 Qutrits')

# fig1=plt.figure()
# ax1 = fig1.add_subplot(111, projection='3d')
# ax1.scatter(ComTime_2q_T, ComTime_2q_CNOT, ComTime_2q, marker='.')
# ax1.scatter(ComTime_3q_T, ComTime_3q_CNOT, ComTime_3q, marker='.')
# ax1.set_xlabel('The number of T-gates')
# ax1.set_ylabel('The number of CNOT-gates')
# ax1.set_zlabel('Computation time')
# ax1.legend(['2q Wigner+Global_opt+Local_opt', '3q Wigner'])
# ax1.set_title('Computation Time')
# plt.show()














