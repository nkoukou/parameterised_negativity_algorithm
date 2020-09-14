import numpy as np
import numpy.random as nr
import matplotlib.pylab as plt

import circuit_components as cc
from copy_phase_space import evolve

plt.rcParams.update({
      "text.usetex": True,
      "font.family": "sans-serif",
      "font.sans-serif": ["Helvetica"]})

MDIR = 'C:/Users/nkouk/Desktop/PhD/Algo/scripts/files/'

DIM = 3

class Circuit(object):
    '''
    Represents a quantum qutrit circuit as described in Pashayan et al. (2015).
    '''
    def __init__(self, sequence):
        ''' sequence - list of strings - describes each step in the circuit.

        Example: ['000SSSS', 'HSHH111', '111CT11', '1111CT1', 'ZZZZZZZ']
                 describes a circuit with:
                 - a 7-qutrit initial state |000SSSS>
                 (0: zero state in computational Z basis and S: Strange state),
                 - followed by a product of local gates HSHH111 (H: Hadamard,
                 S: Phase gate, 1: Identity),
                 - followed by two C_SUM gates
                 (111CT11 indicates that qutrit 4 is the control and qutrit 5
                  the target)
                 - followed by measurement in the Z basis
        '''
        errtest = [len(sequence[i]) for i in range(len(sequence))]
        if not all(x==errtest[0] for x in errtest):
            raise Exception('Number of qudits in sequence is inconsistent.')
        self.n = errtest[0]
        if sequence[-1].find('Z')==-1: sequence.append('Z'*self.n)
        self.d = DIM
        self.L = len(errtest)-2

        self.state_string = sequence[0]
        self.gates_string = sequence[1:-1]
        self.meas_string = sequence[-1]

        self.state0 = None
        self.state = None
        self.gates = None
        self.prob = None

        self.w = None
        self.traj = []
        self.est = []
        self.M = []
        self.Mforw = (5/3)**self.state_string.count('S') #!!!

    def reset(self):
        if self.w is None:
            return
        self.w = None
        self.traj = []
        self.est = []
        self.M = []
        self.Mforw = None


    def simulate_circuit(self, qudit_num, outcome):
        ''' Calculates outcome Born probability for given qudit and outcome
        state.
        qudit_num - int (between 0 - self.n inclusive)
        outcome - int (between 0-2 inclusive)
        '''
        self.state0 = cc.makeState(self.state_string)
        self.state = self.state0
        self.gates = []
        for i in range(self.L):
            gate = cc.makeGate(self.gates_string[i])
            self.gates.append(gate)
            self.state = evolve(self.state, gate)
        measp = cc.makeMeas(self.meas_string, qudit_num, outcome)
        prob = np.trace( np.dot(measp, self.state) )
        if not np.isclose(np.imag(prob), 0):
            raise Exception('Probabilty must be real')
        self.prob = 0 if np.isclose(prob, 0) else np.real(prob)

    def sample(self):
        ''' Performs MC sampling on current Wigner distribution.
        '''
        qdist = self.w
        self.M.append(np.abs(self.w).sum())
        prob = np.abs(qdist) / self.M[-1]

        x1 = np.arange(qdist.size)
        x1 = nr.choice(x1, p=prob)
        self.traj.append(x1)
        self.est.append(np.sign(qdist[x1]))

    def simulate_outcome(self, qudit_num, outcome):
        ''' Simulates outcome through MS sampling.
        '''
        self.reset()
        self.w = cc.calcWstate(self.state_string)
        self.sample()
        for i in range(self.L):
            # print(self.gates_string[i])
            # print(self.traj)
            # print(self.est)
            self.w = cc.calcWgate(self.gates_string[i], self.traj[-1])
            self.sample()
        self.w = cc.calcWmeas(self.meas_string, qudit_num, outcome,
                              self.traj[-1])

    def estimate_outcome(self, qudit_num, outcome, s=100):
        ''' Estimates outcome by performing s simulations.
        '''
        p_est = np.zeros(s)
        for i in range(s):
            self.simulate_outcome(qudit_num, outcome)

            self.traj = np.array(self.traj)
            self.est = np.array(self.est)
            self.M = np.array(self.M)
            self.Mforw = np.prod(self.M)

            p_est[i] = np.prod(self.est*self.M)*self.w
        return p_est.sum()/s

    def calc_hoeffding_bound(self, eps, delta):
        return int( 2/(eps*eps) * self.Mforw**2 * np.log(2/delta) )

def random_gates_string(n, L):
    ''' Creates a random circuit sequence as input for a Circuit object.
    '''
    prob = [0.5, 0.5] # Probability of 1-qudit gate or 2-qudit gate
    char1q, probs1q = ['H', 'S', '1'], [1/3, 1/3, 1/3] # Parameters for 1-qudit

    gates_string = []
    for i in range(L):
        gate_string = ''
        q1 = nr.choice([0,1], p=prob)
        if q1:
            for j in range(n):
                gate_string += nr.choice(char1q, p=probs1q)
        else:
            gate = ['1']*n
            c = nr.choice(np.arange(n))
            t = c
            while t==c:
                t = nr.choice(np.arange(n))
            gate[c] = 'C'
            gate[t] = 'T'
            for g in gate:
                gate_string += g
        gates_string.append(gate_string)
    return gates_string


def data_scaling(n=4, kmax=3, circ_num=10, L=4, eps=0.05, delta=0.1,
                 qudit_num=0, outcome=0):
    ''' Creates data for Fig.1 of Pashayan et al. (2015)
    '''
    filename = MDIR+'pdiffs_n'+str(n)+'_k'+str(kmax)+'_c'+\
               str(circ_num).zfill(2)+'_L'+str(L).zfill(2)+'_e'+\
               str(int(100*eps)).zfill(3)+'_d'+\
               str(int(100*delta)).zfill(3)+'.npy'
    print(filename)
    try:
        p_diffs = np.load(filename)
        return p_diffs
    except: pass

    p_diffs = np.zeros((1+kmax,1+circ_num))
    for k in range(1+kmax):
        state_string = '0'*(n-k) + 'S'*k
        for i in range(circ_num):
            gates_string = random_gates_string(n, L)
            c = Circuit([state_string]+gates_string)
            s = c.calc_hoeffding_bound(eps, delta)

            print('(k,i):', (k,i))
            print('s = ', s)
            print('------------')

            c.simulate_circuit(qudit_num, outcome)
            p_born = c.prob
            p_est = c.estimate_outcome(qudit_num, outcome, s)
            p_diffs[k, i] = np.abs(p_born - p_est)
        p_diffs[k,-1] = s

    np.save(filename, p_diffs)
    return p_diffs

def plot_scaling(n=4, kmax=3, circ_num=10, L=4, eps=0.05, delta=0.1,
                 qudit_num=0, outcome=0):
    ''' Plots data for Fig.1 of Pashayan et al. (2015)
    '''
    err = 5 #!!!
    p_diffs = data_scaling(n, kmax, circ_num, L, eps, delta,
                 qudit_num, outcome)
    n_sampl = p_diffs[:,-1]
    p_diffs = p_diffs[:,:-1]
    p_diffs[np.where(np.isclose(p_diffs, 0.))]=10**(-err)
    kaxis = np.repeat(np.arange(1+kmax)[:,None], circ_num, axis=1)

    cb = plt.cm.get_cmap('RdYlBu')
    cr = np.repeat(n_sampl[:,None], circ_num, axis=1).flatten()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    vmine = int(np.log10(n_sampl.min()))
    vmaxe = int(np.log10(n_sampl.max())+1)
    sc = ax.scatter(kaxis.flatten(), p_diffs.flatten(), marker='o',
                    c=cr, vmin=10**vmine, vmax=10**vmaxe, cmap=cb)
    cb = fig.colorbar(sc, ticks=10**np.arange(vmine, vmaxe+1))
    cb.ax.set_yticklabels([r'$10^{%d}$'%(i)
                           for i in np.arange(vmine, vmaxe+1)])

    ax.axhline(y=eps, c='0.', ls='-')
    ax.set_yscale('log')
    ax.set_xticks(np.arange(1+kmax))
    ax.set_yticks(10.**(-np.arange(1, err+1)))
    ax.set_xticklabels([r'{0}'.format(int(i)) for i in np.arange(1+kmax)])
    ax.set_yticklabels([r'$10^{-%d}$'%(i) for i in np.arange(1, err+1)])
    ax.set_xlabel(r'\# magic states $k$')
    ax.set_ylabel(r'$\hat{p}_{\rm{B}} -\langle \hat{p}\rangle$')
    ax.set_xlim(-0.5,3.5)
    ax.set_ylim(5*10**(-(err+1)), 0.1)

















