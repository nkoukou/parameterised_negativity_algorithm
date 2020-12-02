import numpy as np
import matplotlib.pylab as plt
import matplotlib.colors as colors

from random_circuit_generator import random_circuit_string
from neg_circuit import calc_hoeffding_bound
from calc_born_prob import simulate_circuit
from random_sample import sample_circuit #, accum_mean

MDIR = 'C:/Users/nkouk/Desktop/PhD/Algo/scripts/saves/'


def data_scaling(n=4, kmax=3, circ_num=10, L=4, eps=0.05, delta=0.1):
    ''' Creates data for Fig.1 of Pashayan et al. (2015)
    '''
    mgcs = 'T' #!!!
    state_string = '0'*(n-kmax) + mgcs*kmax
    meas_string = 'Z'+'1'*(n-1) #!!!
    filename = MDIR+'pdiffs_'+state_string+'_c'+\
               str(circ_num).zfill(2)+'_L'+str(L).zfill(2)+'_e'+\
               str(int(100*eps)).zfill(3)+'_d'+\
               str(int(100*delta)).zfill(3)+'.npy'
    print('\n', filename, '\n')
    try:
        p_diffs = np.load(filename)
        return p_diffs
    except: pass

    p_diffs = np.zeros((1+kmax,1+circ_num))
    p_diffs_opt = np.zeros((1+kmax,1+circ_num))
    for k in range(1+kmax):
        state_string = '0'*(n-k) + mgcs*k
        for i in range(circ_num):
            circuit_string = random_circuit_string(n, L,
              given_state=state_string, given_measurement=meas_string,
              p_csum=0.6)
            N, N_opt, gamma_opt = calc_hoeffding_bound(eps, delta,
                                                       circuit_string)

            p_born = simulate_circuit(circuit_string)

            sample_list = sample_circuit(circuit_string, np.zeros(2*n), N)
            sample_list_opt = sample_circuit(circuit_string, gamma_opt, N_opt)

            p_diffs[k, i] = np.abs(p_born - sample_list.mean())
            p_diffs_opt[k, i] = np.abs(p_born - sample_list_opt.mean())

            print('------------')
            print('(k,i):', (k,i))
            print('# samples: ', N, N_opt)
            print('------------')
        p_diffs[k,-1] = N
        p_diffs_opt[k,-1] = N_opt

    # np.save(filename, p_diffs)
    return p_diffs, p_diffs_opt

def plot_scaling(n=4, kmax=3, circ_num=10, L=4, eps=0.05, delta=0.1):
    ''' Plots data for Fig.1 of Pashayan et al. (2015)
    '''
    err = 5 #!!!
    pdiffs, pdiffs_opt = data_scaling(n, kmax, circ_num, L, eps, delta)
    n_sample = pdiffs[:,-1]
    p_diffs = pdiffs[:,:-1]
    p_diffs[np.where(np.isclose(p_diffs, 0.))]=10**(-err)

    n_sample_opt = pdiffs_opt[:,-1]
    p_diffs_opt = pdiffs_opt[:,:-1]
    p_diffs_opt[np.where(np.isclose(p_diffs_opt, 0.))]=10**(-err)

    kaxis = np.repeat(np.arange(1+kmax)[:,None], circ_num, axis=1)

    cb = plt.cm.get_cmap('RdYlBu')
    cr = np.repeat(n_sample[:,None], circ_num, axis=1).flatten()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    vmine = int(np.log10(n_sample.min()))
    vmaxe = int(np.log10(n_sample.max())+1)
    sc = ax.scatter(kaxis.flatten(), p_diffs.flatten(), marker='o', c=cr,
                    norm=colors.LogNorm(vmin=10**vmine, vmax=10**vmaxe),
                    cmap=cb)
    cb = fig.colorbar(sc, ticks=10.**np.arange(vmine-1, vmaxe+1))
    cb.ax.set_yticklabels([r'$10^{%d}$'%(i)
                           for i in np.arange(vmine-1, vmaxe+1)])

    ax.axhline(y=eps, c='0.', ls='-')
    ax.set_yscale('log')
    ax.set_xticks(np.arange(1+kmax))
    ax.set_yticks(10.**(-np.arange(1, err+1)))
    ax.set_xticklabels([r'{0}'.format(int(i)) for i in np.arange(1+kmax)])
    ax.set_yticklabels([r'$10^{-%d}$'%(i) for i in np.arange(1, err+1)])
    ax.set_xlabel(r'\# magic states $k$')
    ax.set_ylabel(r'$\hat{p}_{\rm{B}} -\langle \hat{p}\rangle$')
    ax.set_xlim(-0.5,kmax+0.5)
    ax.set_ylim(5*10**(-(err)), 0.5) #, 0.1)


    cb = plt.cm.get_cmap('RdYlBu')
    cr = np.repeat(n_sample_opt[:,None], circ_num, axis=1).flatten()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    vmine = int(np.log10(n_sample_opt.min()))
    vmaxe = int(np.log10(n_sample_opt.max())+1)
    sc = ax.scatter(kaxis.flatten(), p_diffs_opt.flatten(), marker='s', c=cr,
                    norm=colors.LogNorm(vmin=10**vmine, vmax=10**vmaxe),
                    cmap=cb)
    cb = fig.colorbar(sc, ticks=10.**np.arange(vmine-1, vmaxe+1))
    cb.ax.set_yticklabels([r'$10^{%d}$'%(i)
                           for i in np.arange(vmine-1, vmaxe+1)])

    ax.axhline(y=eps, c='0.', ls='-')
    ax.set_yscale('log')
    ax.set_xticks(np.arange(1+kmax))
    ax.set_yticks(10.**(-np.arange(1, err+1)))
    ax.set_xticklabels([r'{0}'.format(int(i)) for i in np.arange(1+kmax)])
    ax.set_yticklabels([r'$10^{-%d}$'%(i) for i in np.arange(1, err+1)])
    ax.set_xlabel(r'\# magic states $k$')
    ax.set_ylabel(r'$\hat{p}_{\rm{B}} -\langle \hat{p}\rangle$')
    ax.set_xlim(-0.5,kmax+0.5)
    ax.set_ylim(5*10**(-(err)), 0.5) #, 0.1)

    return pdiffs, pdiffs_opt

def plot_scalings(n=4, kmax=3, circ_num=10, L=4, eps=0.05, delta=0.1):
    ''' Plots comparison of Wigner sampling and Optimised sampling in the form
        of Fig.1, Pashayan et al. (2015).
    '''
    err, markers, i = 6, ['o', 's'], 0#!!!
    kaxis = np.arange(1+kmax)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    pdiffs, pdiffs_opt = data_scaling(n, kmax, circ_num, L, eps, delta)
    for pd in [pdiffs, pdiffs_opt]:
        sample_sizes = pd[:,-1]
        print(pd, pd.shape)
        pd_mean = pd[:,:-1].mean(axis=1)
        pd_std = np.std(pd[:,:-1], axis=1)
        print(pd_mean)
        print(pd_std)
        pd_mean[np.where(np.isclose(pd_mean, 0.))]=10**(-err)

        # cb = plt.cm.get_cmap('RdYlBu')
        # ax.scatter(kaxis, pd_mean, marker=markers[i], c=sample_sizes)
        ax.errorbar(kaxis, pd_mean, yerr=pd_std, fmt='o')
        i+=1
    ax.axhline(y=eps, c='0.', ls='-')
    ax.set_xlabel(r'# magic states $k$')
    ax.set_xlim(-0.5,kmax+0.5)
    ax.set_xticks(np.arange(1+kmax))
    ax.set_xticklabels([r'${%d}$'%(int(i)) for i in np.arange(1+kmax)])
    ax.set_ylabel(r'$\hat{p}_{\rm{B}} -\langle \hat{p}\rangle$')
    ax.set_yscale('log')
    # ax.set_ylim(5*10.**(-err), 0.5) #, 0.1)
    # ax.set_yticks(10.**(-np.arange(1, err+1)))
    # ax.set_yticklabels([r'$10^{-%d}$'%(i) for i in np.arange(1, err+1)])

    return pdiffs, pdiffs_opt



pd, pdopt = plot_scalings(n=3, kmax=1, circ_num=3, L=1, eps=0.6, delta=0.5)










