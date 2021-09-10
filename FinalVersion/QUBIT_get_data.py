import autograd.numpy as np
import time
import os
import matplotlib.pyplot as plt
from QUBIT_QD_circuit import (QD_circuit)
from QUBIT_circuit_generator import (haar_random_connected_circuit,
                                     haar_2gate_circuit)
from QUBIT_phase_space import (W_state, W_gate, W_meas)
from QUBIT_Pauli_sampling import (W_state_pauli, W_gate_pauli, W_meas_pauli)

direc='test'

def hist_2gate_neg(samples, max_n_blocks):
    if os.path.isfile(os.path.join(direc,'neg_wig_S'+str(samples)+'.npy')):
        neg_wig = np.load(os.path.join(direc,'neg_wig_S'+str(samples)+'.npy'))
        neg_pau = np.load(os.path.join(direc,'neg_pau_S'+str(samples)+'.npy'))
        trigger = np.where(neg_wig[0,0]==0.)[0][0]
        print('TRIGGER = %d'%(trigger))
    else:
        neg_wig = np.zeros((2, 3, samples))
        neg_pau = np.zeros((2, 3, samples))
        trigger = 0

    x0_wig = [1.,1/2,1/2]
    qp_function_wig = [W_state, W_gate, W_meas]
    x0_pau = [0.,0.,0.]
    qp_function_pau = [W_state_pauli, W_gate_pauli, W_meas_pauli]

    t0 = time.time()
    for s in range(samples):
        if s < trigger: continue
        print('\n============================================================')
        print('s = %d/%d  ||  t = %.0f'%(s,samples, time.time()-t0))
        t0 = time.time()
        print('============================================================\n')
        for b in range(max_n_blocks):
            circ_wig = haar_2gate_circuit(n_blocks=b+1)
            circ_pau = circ_wig.copy()

            circuit = QD_circuit(circ_wig, n=3, l=2, DIM=2,
                                 par0=x0_wig, W=qp_function_wig)
            neg_wig[b,0,s] = circuit.get_neg_circuit(ref=True)
            circuit.opt_x()
            neg_wig[b,1,s] = circuit.get_neg_circuit(ref=False)
            circuit.compress_circuit()
            neg_wig[b,2,s] = circuit.get_neg_circuit(ref=True)

            circuit = QD_circuit(circ_pau, n=3, l=2, DIM=2,
                                 par0=x0_pau, W=qp_function_pau)
            neg_pau[b,0,s] = circuit.get_neg_circuit(ref=True)
            circuit.opt_x()
            neg_pau[b,1,s] = circuit.get_neg_circuit(ref=False)
            circuit.compress_circuit()
            neg_pau[b,2,s] = circuit.get_neg_circuit(ref=True)
        fname = 'neg_wig_S'+str(samples)+'.npy'
        np.save(os.path.join(direc, fname), neg_wig)
        fname = 'neg_pau_S'+str(samples)+'.npy'
        np.save(os.path.join(direc, fname), neg_pau)
    return neg_wig, neg_pau

def plot_hist(samples):
    plt.close('all')
    fnames = ['neg_wig_S'+str(samples)+'.npy', 'neg_pau_S'+str(samples)+'.npy']
    labels = ['No compress & opt', 'Opt', 'Compress']
    titles = [['1 block, Wigner frame', '2 block, Wigner frame'],
             ['1 block, Pauli frame', '2 block, Pauli frame']]
    for i in range(len(fnames)):
        data = np.load(fnames[i])[:,:,:91]
        for b in range(data.shape[0]):
            fig, ax = plt.subplots(1,1)
            for n in range(data.shape[1]):
                bins = np.linspace(data[b,n].min(), data[b,n].max(),
                                   data[b,n].size//2)
                ax.hist(data[b,n], bins, alpha=0.5, label=labels[n])
            ax.legend(loc='upper left')
            ax.set_title(titles[i][b])

def hist_block_nl(samples, L=4, n_rng=[2,3]):
    if L==4: ell_rng = [2,4]
    elif L>4: ell_rng = np.arange(1,L+1)
    else: raise Exception('L is not valid (must have L >= 4)')
    N = L+1

    fname = os.path.join(direc,'neg_L'+str(L)+'_S'+str(samples)+'.npy')
    if os.path.isfile(fname):
        data = np.load(fname)
        trigger = np.where(data[0,0]==0.)[0][0]
        print('TRIGGER = %d'%(trigger))
    else:
        data = np.zeros((3, len(n_rng), len(ell_rng), 2, samples))
        trigger = 0

    x0 = [[1.,1/2,1/2],[0.,0.,0.]]
    qp_fun = [[W_state, W_gate, W_meas],
              [W_state_pauli, W_gate_pauli, W_meas_pauli]]

    t0 = time.time()
    for s in range(samples):
        if s < trigger: continue
        print('\n============================================================')
        print('s = %d/%d  ||  t = %.0f'%(s,samples, time.time()-t0))
        t0 = time.time()
        print('============================================================\n')

        circ_wig = haar_random_connected_circuit(N=N, L=L, n=2, d=2,
                     given_state=0, given_meas=N, method='cc')
        circ_pau = circ_wig.copy()

        for i, en in enumerate(n_rng):
            for j, ell in enumerate(ell_rng):
                for k, circ in enumerate([circ_wig, circ_pau]):
                    if (k==1 and en>2): continue
                    print('n,l,k = %d,%d,%d'%(en,ell,k))

                    circuit = QD_circuit(circ, n=en, l=ell, DIM=2,
                                         par0=x0[k], W=qp_fun[k])
                    data[0,i,j,k,s] = circuit.get_neg_circuit(ref=True)
                    circuit.opt_x()
                    data[1,i,j,k,s] = circuit.get_neg_circuit(ref=False)
                    circuit.compress_circuit()
                    data[2,i,j,k,s] = circuit.get_neg_circuit(ref=True)

                    np.save(fname, data)
    return data

def plot_hist_block(samples, L):
    fname = os.path.join(direc,'neg_L'+str(L)+'_S'+str(samples)+'.npy')
    data = np.load(fname)[:,:,:,:,:16]
    circ_ttl = ['Wigner', 'Pauli']

    plt.close('all')
    for i in range(data.shape[1]):
        en = i+2
        for k in range(data.shape[3]):
            if (k==1 and en>2): continue
            fig, ax = plt.subplots(1,1)
            bins = np.linspace(data[0,i,0,k].min(), data[0,i,0,k].max(),
                                data[0,i,0,k].size//2)
            ax.hist(data[0,i,0,k], bins, alpha=0.5, label=r'$\ell=1$')
            for j in range(data.shape[2]):
                if L==4: ell = 2 if j==0 else 4
                elif L>4: ell = j+1
                bins = np.linspace(data[1,i,j,k].min(), data[1,i,0,k].max(),
                                    data[1,i,j,k].size//2)
                ax.hist(data[1,i,j,k], bins, alpha=0.5,
                        label=r'$\ell=%d$'%(ell))
            ax.legend(loc='upper left')
            ax.set_title(circ_ttl[k]+', n=%d'%(en))


# def get_data(N, L, n_rng, l_rng):
#     dim = 2
#     # x0_wig = [1.,1/2,1/2]
#     # qp_function_wig = [W_state, W_gate, W_meas]
#     x0_pau = [0.,0.,0.]
#     qp_function_pau = [W_state_pauli, W_gate_pauli, W_meas_pauli]
#     circ = haar_random_connected_circuit(N=N, L=L, n=2, d=dim,
#           given_state=None, given_meas=1, method='r')

#     fname = 'neg_N'+str(N)+'_L'+str(L)+'.npy'
#     if os.path.isfile(fname):
#         data = np.load(fname)
#         trigger = np.where(data[0,0]==0.)[0][0]
#         print('TRIGGER = %d'%(trigger))
#     else:
#         data = np.zeros((4, n_rng[1]-n_rng[0]+1, l_rng[1]-l_rng[0]+1))
#         trigger = 0
#     for l in range(l_rng[0], l_rng[1]+1):
#         if l < trigger: continue
#         for n in range(n_rng[0], n_rng[1]+1):
#             print('========================================================\n')
#             print('STEP: n %d || l %d'%(n, l))
#             print(data)

#             t0 = time.time()
#             circuit = QD_circuit(circ, n=n, l=l, DIM=dim,
#                                  par0=x0_pau, W=qp_function_pau)

#             data[0,n-n_rng[0],l-l_rng[0]] = circuit.get_neg_circuit(ref=1)
#             circuit.compress_circuit()
#             circuit.opt_x()
#             data[1,n-n_rng[0],l-l_rng[0]] = circuit.get_neg_circuit(ref=1)
#             data[2,n-n_rng[0],l-l_rng[0]] = circuit.get_neg_circuit(ref=0)

#             t1 = time.time()
#             data[3,n-n_rng[0],l-l_rng[0]] = t1-t0
#             np.save(os.path.join(direc, fname), data)
#     return data

# def scatter_logneg(data):
#     circuit = QD_circuit(circ, n_rng[0], l_rng[0], dim, x0, qp_function)

#     dn = []
#     for n in range(n_rng[0], n_rng[1]+1):
#         dl = []
#         for l in range(l_rng[0], l_rng[1]+1):
#             fname = 'N'+str(circuit.N)+'_L'+str(circuit.L)+\
#                     '_n'+str(n)+'_l'+str(l)+'.npy'
#             dl.append(np.load(os.path.join(direc, fname)))
#         dn.append(dl)
#     data = np.stack(dn)

#     fig, ax = plt.subplots(1,1)
#     sc = ax.imshow(data[:,:,2]-data[:,:,0],
#                       cmap='viridis', origin='lower', aspect='auto',
#                       extent=(l_rng[0], l_rng[1], n_rng[0], n_rng[1]))
#     cc = fig.colorbar(sc)
#     cc.set_label('neg reduction', rotation=270, labelpad=15)

#     ax.set_xlabel(r'$\ell$')
#     xticks = np.arange(l_rng[0], l_rng[1]+1, dtype=int)
#     # cc.set_ticks(xticks)
#     ax.set_ylabel(r'$n$')
#     yticks = np.arange(n_rng[0], n_rng[1]+1, dtype=int)
#     # cc.set_ticks(yticks)

#     fig, ax = plt.subplots(1,1)
#     sc = ax.imshow(data[:,:,3],
#                       cmap='viridis', origin='lower', aspect='auto',
#                       extent=(l_rng[0], l_rng[1], n_rng[0], n_rng[1]))
#     cc = fig.colorbar(sc)
#     cc.set_label('comp time', rotation=270, labelpad=15)

#     return data

# plt.close('all')
# circ = haar_random_connected_circuit(N=N, L=L, n=2, d=dim,
#           given_state=None, given_meas=1, method='r')

# n_rng=(2,3)
# l_rng=(2,6)

# get_data(circ, n_rng, l_rng, samples=1, nls=(0,0,0))

# data = scatter_logneg(circ, n_rng, l_rng)























