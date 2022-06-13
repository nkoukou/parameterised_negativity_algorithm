import numpy as np
import os
import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

plt.rcParams['figure.dpi'] = 200
plt.style.use('classic')
plt.rc('font',   size=24)
plt.rc('axes',   labelsize=25)
plt.rc('xtick',  labelsize=21)
plt.rc('ytick',  labelsize=21)
plt.rc('legend',  fontsize=18)
plt.rc('lines',  linewidth=2 )
plt.rc('lines', markersize=10 )
plt.rc('lines', markeredgewidth=0.)

N = 6
L = 15

main_path = os.path.join('data_optimisation', 'Data_N6_L15')

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

plt.close('all')

figure(figsize=(9,6))

plt.plot(np.log2(neg_list_n2_l1), label='$\ell$ = 1', marker='o',
         color='tab:blue')
plt.plot(np.log2(neg_list_n2_l2), label='$\ell$ = 2', marker='^',
         color='tab:orange')
plt.plot(np.log2(neg_list_n2_l3), label='$\ell$ = 3', marker='v',
         color='tab:green')
plt.plot(np.log2(neg_list_n2_l4), label='$\ell$ = 4', marker='s',
         color='tab:red')
plt.plot(np.log2(neg_list_n2_l5), label='$\ell$ = 5', marker='D',
         color='tab:purple')

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

plt.plot([np.log2(neg_list_n2_l1[0])]*(len(neg_list_n2_l1)+1),
         ls=':', color='tab:grey')
plt.plot([np.log2(neg_list_n3_l1[0])]*(len(neg_list_n2_l1)+1),
         ls='--', color='tab:grey')
plt.plot([np.log2(neg_list_n4_l1[0])]*(len(neg_list_n2_l1)+1),
         ls='-.', color='tab:grey')

plt.legend(loc='lower left', ncol=2)
plt.xlabel(r'Optimisation cycles $c$')
plt.ylabel(r'Circuit negativity $\log{N_C({\cal{G}}_{\rm{opt}})}$')
plt.xlim([0,37])
plt.ylim([16,32.1])
plt.show()



