import autograd.numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt
# import itertools as it
# import time
import os

from qubit_circuit_components import(makeState, makeGate)
from compression import(compress_circuit)
from frame_opt import(neg_gate_max)
from phase_space import(PhaseSpace)
from qubit_frame_Pauli import(F, G, DIM, x0)
ps_Pauli = PhaseSpace(F,G,x0,DIM)
from qubit_frame_Wigner import(F, G, DIM, x0)
ps_Wigner = PhaseSpace(F,G,x0,DIM)

direc = 'data_compression'
np.set_printoptions(precision=4, suppress=True)

plt.rcParams['figure.dpi'] = 200
plt.style.use('classic')
plt.rc('font',   size=24)
plt.rc('axes',   labelsize=25)
plt.rc('xtick',  labelsize=21)
plt.rc('ytick',  labelsize=21)
plt.rc('legend',  fontsize=23)
plt.rc('lines',  linewidth=2 )
plt.rc('lines', markersize=5 )

to_beat = np.array([0.228443, 2*0.228443, 2*0.271553])

def generate_random_CliffT(N, L, T):
    state_list = N*[makeState('0')]

    Cliff_list = ['X', 'Z', 'H', 'K', 'C+', '+C']
    Cliff_prob = [1./len(Cliff_list)]*len(Cliff_list)
    gate_seq = []
    for i in range(L):
        gate = makeGate(nr.choice(Cliff_list, p=Cliff_prob))
        index = list(nr.choice(N, size=int(np.log2(len(gate))), replace=False))
        gate_seq.append((gate, index))
    for i in range(T):
        gate = makeGate('T')
        index = [nr.randint(N)]
        gate_seq.append((gate, index))
    nr.shuffle(gate_seq)
    gate_list, index_list = zip(*gate_seq)
    gate_list, index_list = list(gate_list), list(index_list)

    meas_list = N*[makeState('0')]

    circuit = {'state_list': state_list, 'gate_list': gate_list,
               'index_list': index_list, 'meas_list': meas_list}
    return circuit

def generate_data(N, n, L, T, S):
    fname = os.path.join(direc,
              "neg_compressed_pau_N%d_n%d_L%d_T%d_S%d.npy"%(N,n,L,T,S))
    trig = 0.
    data = np.zeros(S)
    if os.path.isfile(fname):
        data = np.load(fname)
        trig = np.where(np.load(fname)!=0.)[0].size

    x0 = ps_Pauli.x0
    for s in range(S):
        if s < trig: continue
        print("(N, n, L, T) = (%d, %d, %d, %d) | %d / %d PAU"%(N,n,L,T,s+1,S))
        circuit = generate_random_CliffT(N, L, T)
        cc = circuit.copy()
        cc = compress_circuit(cc, n)
        neg_cc = 1
        for gate in cc['gate_list']:
            params = int(np.log2(len(gate)))*[x0]
            neg_cc *= neg_gate_max(ps_Pauli.W_gate, gate, params, params)
        data[s] = neg_cc
        np.save(fname, data)
    return np.array(data)


def plot_data(N, n, L, Ts, S):
    if type(Ts)==int: Ts = [Ts]
    stats, stats1, stats2 = [], [], []
    fig, ax = plt.subplots(2,2)
    for i,t in enumerate(Ts):
        i, j = i//2, i%2
        data = np.load(os.path.join(direc,
                 "neg_compressed_pau_N%d_n%d_L%d_T%d_S%d.npy"%(N,n,L,t,S)))
        data = 2*np.log2(data)/t
        stat = [data.mean(), data.std(), np.where(data<to_beat[0]
                )[0].size/S, np.where(data<to_beat[1]
                )[0].size/S, np.where(data<to_beat[2])[0].size/S]
        stats.append(stat)
        freq, bnds, _ = ax[i,j].hist(data, bins=20, fc=(0, 0, 1., 0.3),
                          edgecolor="none")
        xbnds = bnds

        ax[i,j].vlines(x=stat[0], ymin=0, ymax=300, lw=4, #freq.max()
                            color='b', ls='-', label=r'$n = %d$'%(n))
        ax[i,j].axvspan(stat[0]-stat[1], stat[0]+stat[1],
                          color='gray', alpha=0.2)

        temp1 = np.load(os.path.join(direc,
                  "neg_compressed_pau_N%d_n%d_L%d_T%d_S%d.npy"%(N,n-1,L,t,S)))
        temp1 = 2*np.log2(temp1)/t
        stat1 = [temp1.mean(), temp1.std(), np.where(temp1<to_beat[0]
                )[0].size/S, np.where(temp1<to_beat[1]
                )[0].size/S, np.where(temp1<to_beat[2])[0].size/S]
        stats1.append(stat1)
        # freq, bnds, _ = ax[i,j].hist(temp1, bins=20, fc=(0, 1, 0, 0.3),
        #                   edgecolor="none")
        ax[i,j].vlines(x=stat1[0], ymin=0, ymax=300, lw=4,
                            color=(0.1, 0.8, 0.1, 1.), ls='-',
                            label=r'$n = %d$'%(n-1))
        xbnds = np.append(xbnds, stat1[0])

        temp2 = np.load(os.path.join(direc,
                  "neg_compressed_pau_N%d_n%d_L%d_T%d_S%d.npy"%(N,n-2,L,t,S)))
        temp2 = 2*np.log2(temp2)/t
        stat2 = [temp2.mean(), temp2.std(), np.where(temp2<to_beat[0]
                )[0].size/S, np.where(temp2<to_beat[1]
                )[0].size/S, np.where(temp2<to_beat[2])[0].size/S]
        stats2.append(stat2)
        # freq, bnds, _ = ax[i,j].hist(temp2, bins=20, fc=(1, 0, 0, 0.3),
        #                   edgecolor="none")
        ax[i,j].vlines(x=stat2[0], ymin=0, ymax=300, lw=4,
                            color=(0.8, 0.1, 0.1, 1.), ls='-',
                            label=r'$n = %d$'%(n-2))
        xbnds = np.append(xbnds, stat2[0])

        ax[i,j].set_xticks(np.round(np.linspace(0., bnds.max(), 5), 2))
        ax[i,j].tick_params(axis='x', which='major', pad=10)
        ax[i,j].set_xlabel(r'$\log{(N^{\rm Pauli}_T)^2}/t$')
        ax[i,j].set_xlim(0., xbnds.max()*1.05)
        ax[i,j].set_ylim(0., freq.max())
        ax[i,j].legend(title=r'$t = %d$'%(t), loc='upper left')

        for val in to_beat:
            ax[i,j].vlines(x=val, ymin=0, ymax=freq.max(),
                            color='brown', ls='--')
    stats,stats1,stats2 = np.array(stats),np.array(stats1),np.array(stats2)

    fig, ax = plt.subplots(1,1)
    ls0 = '-'
    ax.plot(Ts, stats[:,0], ls=ls0, marker='s', markersize=10,
            color=(0.1, 0.2, 0.7, 1.), label="n=5")
    ax.plot(Ts, stats1[:,0], ls=ls0, marker='o', markersize=10,
            color=(0.1, 0.8, 0.1, 1.), label="n=4")
    ax.plot(Ts, stats2[:,0], ls=ls0, marker='x', markersize=10,
            color=(0.8, 0.1, 0.1, 1.), label="n=3")
    ax.set_xticks(Ts)
    ax.set_xlabel(r'$t$')
    ax.set_xlim(min(Ts)-0.2, max(Ts)+0.2)
    ax.set_ylabel(r'Average $\log{(N^{\rm Pauli}_T)^2}/t$')
    ax.legend(loc='lower left').set_draggable(1)

    return stats, stats1, stats2

Ts = [12,13,14,15]
plt.close('all')
stats, stats1, stats2 = plot_data(N=5, n=5, L=100, Ts=Ts, S=1000)










