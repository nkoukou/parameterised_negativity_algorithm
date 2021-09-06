import autograd.numpy as np
import time
import os
import matplotlib.pyplot as plt
from QUBIT_QD_circuit import (QD_circuit)
from QUBIT_circuit_generator import (haar_random_connected_circuit)
from QUBIT_phase_space import (W_state, W_gate, W_meas)

N = 10
L = 30
dim = 2
x0 = [1,1/2,1/2]
qp_function = [W_state, W_gate, W_meas]
direc='test'

def get_data(circ, n_rng, l_rng, samples, nls=(0,0,0)):
    for n in range(n_rng[0], n_rng[1]+1):
        if n < nls[0]: continue
        for l in range(l_rng[0], l_rng[1]+1):
            if (n==nls[0] and l < nls[1]): continue
            for s in range(samples):
                if (n==nls[0] and l==nls[1] and s < nls[2]): continue
                print('\n')
                print('================================================')
                print('================================================')
                print('================================================')
                print('\n')
                print('STEP: n %d || l %d || s %d / %d'%(n, l, s+1, samples))

                t0 = time.time()
                circuit = QD_circuit(circ, n, l, dim, x0, qp_function)

                neg0 = circuit.get_neg_circuit(ref=1)
                circuit.compress_circuit()
                circuit.opt_x()
                neg1 = circuit.get_neg_circuit(ref=1)
                neg2 = circuit.get_neg_circuit(ref=0)

                t1 = time.time()
                neg = np.array([neg0, neg1, neg2, t1-t0])
                fname = 'N'+str(N)+'_L'+str(L)+'_n'+str(n)+'_l'+str(l)+'.npy'
                np.save(os.path.join(direc, fname), neg)

def scatter_logneg(circ, n_rng, l_rng):
    circuit = QD_circuit(circ, n_rng[0], l_rng[0], dim, x0, qp_function)

    dn = []
    for n in range(n_rng[0], n_rng[1]+1):
        dl = []
        for l in range(l_rng[0], l_rng[1]+1):
            fname = 'N'+str(circuit.N)+'_L'+str(circuit.L)+\
                    '_n'+str(n)+'_l'+str(l)+'.npy'
            dl.append(np.load(os.path.join(direc, fname)))
        dn.append(dl)
    data = np.stack(dn)

    fig, ax = plt.subplots(1,1)
    sc = ax.imshow(data[:,:,2]-data[:,:,0],
                      cmap='viridis', origin='lower', aspect='auto',
                      extent=(l_rng[0], l_rng[1], n_rng[0], n_rng[1]))
    cc = fig.colorbar(sc)
    cc.set_label('neg reduction', rotation=270, labelpad=15)

    ax.set_xlabel(r'$\ell$')
    xticks = np.arange(l_rng[0], l_rng[1]+1, dtype=int)
    # cc.set_ticks(xticks)
    ax.set_ylabel(r'$n$')
    yticks = np.arange(n_rng[0], n_rng[1]+1, dtype=int)
    # cc.set_ticks(yticks)

    fig, ax = plt.subplots(1,1)
    sc = ax.imshow(data[:,:,3],
                      cmap='viridis', origin='lower', aspect='auto',
                      extent=(l_rng[0], l_rng[1], n_rng[0], n_rng[1]))
    cc = fig.colorbar(sc)
    cc.set_label('comp time', rotation=270, labelpad=15)

    return data

plt.close('all')
circ = haar_random_connected_circuit(N=N, L=L, n=2, d=dim,
          given_state=None, given_meas=1, method='r')

n_rng=(2,3)
l_rng=(2,6)

# get_data(circ, n_rng, l_rng, samples=1, nls=(0,0,0))

data = scatter_logneg(circ, n_rng, l_rng)























