import numpy as np
import matplotlib.pylab as plt
import os

path = 'data_Q10_m23'
data_names = ['Wigner_neg_Q', 'Opt_neg_Q', 'Local_opt_neg_Q', 'ComTime_Q']
n_qubit = 10

def groupby_mean(a):
    b = a[a[:,0].argsort()]
    idx = np.flatnonzero(np.r_[True,b[:-1,0]!=b[1:,0],True])
    counts = np.diff(idx).astype(float)[:,None]
    avg = np.add.reduceat(b[:,1:],idx[:-1],axis=0)/counts
    return np.c_[b[idx[:-1],0],avg]

def hist_vs_T(n_qubit, n_CNOT):
    data2, data3 = [], []
    cw = ['r', 'g', 'b']
    labels = ['W', 'GO', 'LO']
    for i in range(3):
        fname = data_names[i]+str(n_qubit)+'_CNOT'+str(n_CNOT)+'_m2.npy'
        fname = os.path.join(path, fname)
        temp2 = np.load(fname)
        temp2 = groupby_mean(temp2)
        data2.append(temp2[:,1])
        fname = data_names[i]+str(n_qubit)+'_CNOT'+str(n_CNOT)+'_m3.npy'
        fname = os.path.join(path, fname)
        temp3 = np.load(fname)
        temp3 = groupby_mean(temp3)
        if i==0: tgates2, tgates3 = temp2[:,0], temp3[:,0]
        data3.append(temp3[:,1])
        plt.plot(tgates2, data2[i], 'o', c=cw[i], label=labels[i]+' (m=2)')
        plt.plot(tgates3, data3[i], 'x', c=cw[i], label=labels[i]+' (m=3)')
        plt.xlabel(r'#$T$ Gates', fontsize=16)
        plt.ylabel(r'$\log{{\cal N}}$', fontsize=16)
        plt.legend(title='#qubits = %d'%(n_qubit)).set_draggable(1)

def scatter_logneg(n_qubit, m=2, diff=True, max_n_CNOT=None):
    if max_n_CNOT is None: max_n_CNOT = 2*n_qubit+1
    n_cnot = np.arange(n_qubit-1, max_n_CNOT)
    n_tgat = np.arange(0, 2*n_cnot.max()+1)
    data = np.zeros((4, n_cnot.size, n_tgat.size))
    labels = [r'$\log{{\cal N}_{\rm{Wigner}}}\quad (m=%d)$'%(m),
              r'$\log{{\cal N}_{\rm{Global}}}\quad (m=%d)$'%(m),
              r'$\log{{\cal N}_{\rm{Local}}}\quad (m=%d)$'%(m),
              r'Comp Time $(s)\quad (m=%d)$'%(m)]
    if diff:
        for k in [1,2]:
            labels[k] = r'$\log{{\cal N}_{\rm{Wigner}}} - $' + labels[k]

    for k in range(4):
        for i in range(n_cnot.size):
            fname = data_names[k]+str(n_qubit)+'_CNOT'+str(n_cnot[i])+\
                    '_m'+str(m)+'.npy'
            fname = os.path.join(path, fname)
            temp = np.load(fname)
            temp = groupby_mean(temp)
            data[k, i,temp[:,0].astype(int)] = temp[:,1]

        fig = plt.figure()
        ax  = fig.add_subplot(111)

        logneg = data[k].T
        if diff and (k in [1,2]):
            logneg = data[0].T - logneg
        logneg[logneg==0.] = np.nan
        xx, yy = np.meshgrid(n_cnot, n_tgat)
        s = 10.
        OFF = 0.05*s
        sc = ax.scatter(xx, yy, c=logneg, s=s, alpha=1, marker='s',
                        cmap=plt.cm.get_cmap('YlOrRd'))

        levels = np.linspace(np.nanmin(logneg), np.nanmax(logneg), 5)
        # ax.contour(xx, yy, logneg, levels=levels,
        #             linewidths=0.5, colors='k')
        cc = fig.colorbar(sc)
        cc.set_label(labels[k], rotation=-90, labelpad=25)
        cc.set_ticks(levels)
        cc.set_ticklabels([r'$%.2f$'%(tck) for tck in levels])

        ax.set_xlabel(r'#CNOT Gates')
        ax.set_xlim(n_cnot.min()-OFF, n_cnot.max()+OFF)
        xticks = np.arange(n_cnot.min(), n_cnot.max(), 2)
        ax.set_xticks(xticks)
        ax.set_xticklabels([r'$%d$'%(tck) for tck in xticks])

        ax.set_ylabel(r'#$T$ Gates')
        ax.set_ylim(0.-OFF, n_tgat.max()+OFF)
        yticks = np.arange(0, n_tgat.max(), 4)
        ax.set_yticks(yticks)
        ax.set_yticklabels([r'$%d$'%(tck) for tck in yticks])
    return data

# plt.close('all')
hist_vs_T(n_qubit, 15)
# data = scatter_logneg(n_qubit=n_qubit, m=2, diff=False, max_n_CNOT=17)


















