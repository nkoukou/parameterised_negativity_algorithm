import numpy as np
import os
import matplotlib.pyplot as plt

qudit_num = 5
gate_num = 10
opt_neg_list_1 = []
wig_neg_list_1 = []
for run in range(200):
    filename = 'Q'+str(qudit_num)+'_G'+str(gate_num)+'_seq'+format(run, '03d')
    opt_neg_list_1.append(np.load(os.path.join('data',filename, 'optimized_neg.npy')))
    wig_neg_list_1.append(np.load(os.path.join('data',filename, 'Wigner_neg.npy')))

qudit_num = 10
gate_num = 30
opt_neg_list_2 = []
wig_neg_list_2 = []
for run in range(25):
    filename = 'Q'+str(qudit_num)+'_G'+str(gate_num)+'_seq'+format(run, '03d')
    opt_neg_list_2.append(np.load(os.path.join('data',filename, 'optimized_neg.npy')))
    wig_neg_list_2.append(np.load(os.path.join('data',filename, 'Wigner_neg.npy')))

aa = np.linspace(0,np.max([wig_neg_list_1,wig_neg_list_2]),10)
zz = np.zeros(len(aa))
# plt.axes().set_aspect('equal')
plt.loglog(
    np.exp(wig_neg_list_1),np.exp(opt_neg_list_1),'b.',
    np.exp(wig_neg_list_2),np.exp(opt_neg_list_2),'gd',
    np.exp(aa), np.exp(aa),'r-',
    np.exp(aa)-1, np.exp(zz),'k--',
#     np.exp([4,4]),np.exp([1.5,1.5+np.log(10)]), 'k'
)
plt.xlabel('Wigner Negativity')
plt.ylabel('Optimized Negativity')
plt.legend(['N=5, Gate_Num=10 (200 samples)','N=10, Gate_Num=30 (30 samples)','Wigner (No-optimization)'])
plt.tight_layout()

plt.savefig("Compare_Neg.eps",dpi=1000,pad_inches=0.1, transparent=True)
plt.show()
