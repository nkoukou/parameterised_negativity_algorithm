import numpy as np
import matplotlib.pylab as plt
from matplotlib.pyplot import cm
import matplotlib.legend_handler
import os

np.set_printoptions(precision=4, suppress=True)
plt.style.use('classic')
plt.rc('font',   size=24)
plt.rc('axes',   labelsize=24)
plt.rc('xtick',  labelsize=20)
plt.rc('ytick',  labelsize=20)
plt.rc('legend',  fontsize=22)
plt.rc('lines',  linewidth=2 )
plt.rc('lines', markersize=7 )

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
    labels = ['WI (m=2/3)', 'GO (m=2/3)', 'LO (m=2/3)']
    fig, ax = plt.subplots(1,1)
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

        l1, = ax.plot(tgates2, data2[i], 'o', c=cw[i], label=labels[i])
        l2, = ax.plot(tgates3, data3[i], 'x', c=cw[i])
        # handles.append((l1,l2))
        ax.axhline(y=0, c='grey', alpha=0.2)
        ax.set_xlabel(r'#$T$ Gates')
        ax.set_ylabel(r'$\log{{\cal N}}$')
        xticks = np.arange(0, 2*n_CNOT+1, 2)
        ax.set_xticks(xticks)
        ax.set_xticklabels([r'$%d$'%(tck) for tck in xticks])
    # _, labels = ax.get_legend_handles_labels()
    # title = '(#Q, #CNOT) = (%d, %d)'%(n_qubit, n_CNOT)
    ax.legend(#handles = handles, labels=labels,
              #title=title
              ).set_draggable(1)

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


        fig, ax = plt.subplots(1,1)
        logneg = data[k].T
        if diff and (k in [1,2]):
            logneg = data[0].T - logneg
        logneg[logneg==0.] = np.nan
        xx, yy = np.meshgrid(n_cnot, n_tgat)
        s = 30.
        OFF = 0.05*s
        sc = ax.scatter(xx, yy, c=logneg, s=s, alpha=1,
                        marker='s', edgecolors='none',
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
# hist_vs_T(n_qubit, 9)
# data = scatter_logneg(n_qubit=n_qubit, m=2, diff=False, max_n_CNOT=19)

def group_data(q_rng=(4,11)):
    direc = 'test_pauli'
    var = ['WI_', 'LO_', 'PS_', 'PO_', 'CT_']
    samples = 10

    for n_qubit in range(q_rng[0], q_rng[1]+1):
        for n_cnot in [n_qubit, n_qubit+1]:
            Tgates = np.arange(2*n_cnot+1)
            neg = np.zeros((len(var), 3, len(Tgates)))
            for n_Tgate in Tgates:
                temp = np.zeros((neg.shape[0],samples))
                for num in range(samples):
                    fname = 'Q'+str(n_qubit)+'_CNOT'+str(n_cnot)+\
                                    '_T'+str(n_Tgate)+'_s'+str(num)+'.npy'
                    for i in range(neg.shape[0]):
                        temp[i,num] = np.load(os.path.join(direc,var[i]+fname))
                for i in range(neg.shape[0]):
                    neg[i,0,n_Tgate] = temp[i].min()
                    neg[i,1,n_Tgate] = temp[i].mean()
                    neg[i,2,n_Tgate] = temp[i].std()
            fname = 'Q'+str(n_qubit)+'_CNOT'+str(n_cnot)+'.npy'
            for i in range(neg.shape[0]):
                np.save(os.path.join(direc,'AVG_'+var[i]+fname), neg[i])


def get_data(n_qubit, n_cnot):
    if n_cnot==0: cnots = [n_qubit]
    if n_cnot==1: cnots = [n_qubit+1]
    if n_cnot==2: cnots = [n_qubit, n_qubit+1]
    direc = 'test_pauli'
    var = ['WI_', 'LO_', 'PS_', 'PO_']
    ttl = ['Wig', 'Frame LO', 'Pauli', 'Pauli LO']
    sty = [('o', '-'), ('x', '--')]

    fig, ax = plt.subplots(1,1)
    cw = cm.rainbow(np.linspace(0,1,len(var)))
    for n_cnot in cnots:
        j = n_cnot - n_qubit
        fname = 'Q'+str(n_qubit)+'_CNOT'+str(n_cnot)+'.npy'

        Tgates = np.arange(2*n_cnot+1)
        neg = []
        for i in range(len(var)):
            neg.append( np.load(os.path.join(direc,'AVG_'+var[i]+fname)) )

        for i in range(len(var)):
            ax.errorbar(Tgates, neg[i][1], yerr=neg[i][2],
                        marker=sty[j][0], ls='', c=cw[i],
                        label=ttl[i])
            ax.plot(Tgates, neg[i][0], sty[j][1], c=cw[i])
    ax.axhline(y=0., c='grey', alpha=0.2)
    ax.set_xlabel(r'#$T$ Gates')
    ax.set_xlim(Tgates.min(), Tgates.max())
    ax.set_ylabel(r'$\log{{\cal N}}$')
    ax.set_ylim(-1, neg[0][1].max()+1)
    ax.legend(title='#qubits = %d\n#cnots = %d'%(n_qubit, n_cnot),
              loc='upper left').set_draggable(1)
    return neg

plt.close('all')
n_cnot=0
q_rng = (3,8)
# group_data(q_rng)
for i in range(q_rng[0], q_rng[1]+1):
    wi, lo, ps, po = get_data(n_qubit=i, n_cnot=n_cnot)
    plt.savefig('figs_constant_neg/Pauli_Q'+str(i)+'.pdf',
                bbox_inches='tight')
















