import numpy as np
import matplotlib.pylab as plt
import os

path = 'data_Q10'
data_names = ['Wigner_neg_Q', 'Opt_neg_Q', 'Local_opt_neg_Q', 'ComTime_Q']
n_qubit = 10

def groupby_mean(a):
    # Sort array by groupby column
    b = a[a[:,0].argsort()]

    # Get interval indices for the sorted groupby col
    idx = np.flatnonzero(np.r_[True,b[:-1,0]!=b[1:,0],True])

    # Get counts of each group and sum rows based on the groupings &
    # hence averages
    counts = np.diff(idx)
    avg = np.add.reduceat(b[:,1:],idx[:-1],axis=0)/counts.astype(float
                                                                 )[:,None]

    # Finally concatenate for the output in desired format
    return np.c_[b[idx[:-1],0],avg]

def hist_vs_T(n_qubit, n_CNOT):
    data = []
    cw = ['r', 'g', 'b']
    labels = ['Wigner', 'Global Opt', 'Local Opt']
    for i in range(3):
        fname = data_names[i]+str(n_qubit)+'_CNOT'+str(n_CNOT)+'.npy'
        fname = os.path.join(path, fname)
        temp = np.load(fname)
        temp = groupby_mean(temp)
        if i==0: tgates = temp[:,0]
        data.append(temp[:,1])
        plt.plot(tgates, data[i], 'o', c=cw[i], label=labels[i])
        plt.xlabel(r'#$T$ Gates', fontsize=16)
        plt.ylabel(r'$\log{{\cal N}}$', fontsize=16)
        plt.legend()

def scatter_logneg(n_qubit, diff=True):
    n_cnot = np.arange(n_qubit-1, 2*n_qubit+1)
    n_tgat = np.arange(0, 2*n_cnot.max()+1)
    data = np.zeros((4, n_cnot.size, n_tgat.size))
    labels = [r'$\log{{\cal N}_{\rm{Wigner}}}$',
              r'$\log{{\cal N}_{\rm{Global}}}$',
              r'$\log{{\cal N}_{\rm{Local}}}$',
              r'Comp Time ($s$)']
    if diff:
        for k in [1,2]:
            labels[k] = r'$\log{{\cal N}_{\rm{Wigner}}} - $' + labels[k]

    for k in range(4):
        for i in range(n_cnot.size):
            fname = data_names[k]+str(n_qubit)+'_CNOT'+str(n_cnot[i])+'.npy'
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

plt.close('all')
hist_vs_T(n_qubit, 15)
# data = scatter_logneg(n_qubit, diff=True)


















