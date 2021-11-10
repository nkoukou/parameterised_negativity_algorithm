import numpy as np
import os
import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

N = 6
L = 15

main_path = 'Data_for_optimisation/Data_N6_L15_Wigner'

file_name = 'neg_list_n2_l1.npy'
path = os.path.join(main_path, file_name)
with open(path, 'rb') as f:
	neg_list_n2_l1 = np.load(f)
file_name = 'neg_list_n2_l2.npy'
path = os.path.join(main_path, file_name)
with open(path, 'rb') as f:
	neg_list_n2_l2 = np.load(f)
file_name = 'neg_list_n2_l3.npy'
path = os.path.join(main_path, file_name)
with open(path, 'rb') as f:
	neg_list_n2_l3 = np.load(f)
file_name = 'neg_list_n2_l4.npy'
path = os.path.join(main_path, file_name)
with open(path, 'rb') as f:
	neg_list_n2_l4 = np.load(f)
file_name = 'neg_list_n2_l5.npy'
path = os.path.join(main_path, file_name)
with open(path, 'rb') as f:
	neg_list_n2_l5 = np.load(f)

file_name = 'neg_list_n3_l1.npy'
path = os.path.join(main_path, file_name)
with open(path, 'rb') as f:
	neg_list_n3_l1 = np.load(f)
file_name = 'neg_list_n3_l2.npy'
path = os.path.join(main_path, file_name)
with open(path, 'rb') as f:
	neg_list_n3_l2 = np.load(f)
file_name = 'neg_list_n3_l3.npy'
path = os.path.join(main_path, file_name)
with open(path, 'rb') as f:
	neg_list_n3_l3 = np.load(f)
file_name = 'neg_list_n3_l4.npy'
path = os.path.join(main_path, file_name)
with open(path, 'rb') as f:
	neg_list_n3_l4 = np.load(f)
file_name = 'neg_list_n3_l5.npy'
path = os.path.join(main_path, file_name)
with open(path, 'rb') as f:
	neg_list_n3_l5 = np.load(f)

file_name = 'neg_list_n4_l1.npy'
path = os.path.join(main_path, file_name)
with open(path, 'rb') as f:
	neg_list_n4_l1 = np.load(f)
file_name = 'neg_list_n4_l2.npy'
path = os.path.join(main_path, file_name)
with open(path, 'rb') as f:
	neg_list_n4_l2 = np.load(f)
file_name = 'neg_list_n4_l3.npy'
path = os.path.join(main_path, file_name)
with open(path, 'rb') as f:
	neg_list_n4_l3 = np.load(f)
file_name = 'neg_list_n4_l4.npy'
path = os.path.join(main_path, file_name)
with open(path, 'rb') as f:
	neg_list_n4_l4 = np.load(f)
file_name = 'neg_list_n4_l5.npy'
path = os.path.join(main_path, file_name)
with open(path, 'rb') as f:
	neg_list_n4_l5 = np.load(f)

# print("The initial negativity:", np.log2(neg_list_n2_l1[0]))
# print("The lowest with n=2:", np.log2(neg_list_n2_l5[-1]))
# print("The lowest with n=3:", np.log2(neg_list_n3_l5[-1]))
# print("The lowest with n=3:", np.log2(neg_list_n4_l5[-1]))

# plt.title('Haar-random circuit with N='+str(N)+' L='+str(L))

figure(figsize=(9,6))

plt.plot(np.log2(neg_list_n2_l1), label='$\ell$ = 1', marker='o', color='tab:blue')
plt.plot(np.log2(neg_list_n2_l2), label='$\ell$ = 2', marker='^', color='tab:orange')
plt.plot(np.log2(neg_list_n2_l3), label='$\ell$ = 3', marker='v', color='tab:green')
plt.plot(np.log2(neg_list_n2_l4), label='$\ell$ = 4', marker='s', color='tab:red')
plt.plot(np.log2(neg_list_n2_l5), label='$\ell$ = 5', marker='D', color='tab:purple')

plt.plot(np.log2(neg_list_n3_l1), marker='o', color='tab:blue')
plt.plot(np.log2(neg_list_n3_l2), marker='^', color='tab:orange')
plt.plot(np.log2(neg_list_n3_l3), marker='v', color='tab:green')
plt.plot(np.log2(neg_list_n3_l4), marker='s', color='tab:red')
plt.plot(np.log2(neg_list_n3_l5), marker='D', color='tab:purple')

plt.plot(np.log2(neg_list_n4_l1), marker='o', color='tab:blue')
plt.plot(np.log2(neg_list_n4_l2), marker='^', color='tab:orange')
plt.plot(np.log2(neg_list_n4_l3), marker='v', color='tab:green')
plt.plot(np.log2(neg_list_n4_l4), marker='s', color='tab:red')
plt.plot(np.log2(neg_list_n4_l5), marker='D', color='tab:purple')

plt.plot([np.log2(neg_list_n2_l1[0])]*len(neg_list_n2_l1), ':', color='tab:grey')
plt.plot([np.log2(neg_list_n3_l1[0])]*len(neg_list_n2_l1), '--', color='tab:grey')
plt.plot([np.log2(neg_list_n4_l1[0])]*len(neg_list_n2_l1), '-.', color='tab:grey')

plt.legend(loc='lower left')
plt.xlabel('The number of optimisation cycles', size=15)
plt.ylabel('log-negativity of the circuit', size=15)
plt.xlim([0,36])
# plt.ylim([0,20])
plt.show()