import numpy as np
import numpy.random as nr
import matplotlib.pylab as plt
import matplotlib.colors as colors

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
    def __init__(self, sequence, s=0.):
        ''' sequence - list of strings - describes each level of circuit
                                         components.

            Example: ['000SSSS', 'HSHH111', '111CT11', '1111CT1', 'xxxx0xx']
                     describes a circuit with:
                - a 7-qutrit initial state |000SSSS>
                  (0: zero state in computational Z basis and S: Strange
                   state);
                - followed by a product of local gates HSHH111
                  (H: Hadamard, S: Phase gate, 1: Identity);
                - followed by two C_SUM gates
                  (111CT11 indicates that qutrit 4 is the control and qutrit 5
                   is the target);
                - followed by a single-qudit projective measurement <xxxx0xx|
                  (x: indicates a qudit which is not measured, 0: the state on
                   which the qudit is projected).

            s        - list of floats - smoothing parameters of circuit
                                        components

            Example: [ [0., 1.], [ [0.5, 0.1], [0.3, 0.2] ],
                       [ [0.55, 0.15], [0.35, 0.25] ], [0.4, 0.7] ]
                     for cicruit ['0S', 'HS', 'CT' 'x1'] means that:
                - States 0 and S have s-parameters 0. and 1. respectively;
                - Gate H has s-parameters (s_in, s_out) = (0.5, 0.3);
                  Gate S has s-parameters (s_in, s_out) = (0.1, 0.2);
                  Gate CT has s-parameters (s_in, s_out) =
                  ([0.55,0.35], [0.35, 0.25]);
                - Measurement projectors have s-parameters 0.4 and 0.7.
        '''
        errtest = [len(sequence[i]) for i in range(len(sequence))]
        if not all(x==errtest[0] for x in errtest):
            raise Exception('Number of qudits in sequence is inconsistent.')
        self.n = errtest[0]
        self.d = DIM
        self.L = len(errtest)-2

        self.state_string = sequence[0]
        self.gates_string = sequence[1:-1]
        self.meas_string = sequence[-1]

        if type(s) in [float, int]:
            print('Setting s = 0')
            temp = [0.]*self.n
            s = [temp] + [[temp, temp]]*self.L + [temp]

        self.state_sparam = s[0]
        self.gates_sparam = s[1:-1]
        self.meas_sparam = s[-1]

        self.state0 = None
        self.state = None
        self.gates = None
        self.prob = None

        self.w = None
        self.traj = []
        self.est = []
        self.M = []
        self.Mforw = self.calc_Mforw()

    def reset(self):
        if self.w is None:
            return
        self.w = None
        self.traj = []
        self.est = []
        self.M = []
        self.Mforw = None

    def calc_Mforw(self):
        ''' Calculates M_forward. #!!!
        '''
        self.w = cc.calcWstate(self.state_string, self.state_sparam)
        self.sample()
        Mforw = np.abs(self.w).sum()
        self.reset()
        return Mforw

    def simulate_circuit(self):
        ''' Calculates outcome Born probability for given qudit and outcome
        state.
        '''
        self.state0 = cc.makeState(self.state_string)
        self.state = self.state0
        self.gates = []
        for i in range(self.L):
            gate = cc.makeGate(self.gates_string[i])
            self.gates.append(gate)
            self.state = evolve(self.state, gate)
        measp = cc.makeMeas(self.meas_string)
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

    def simulate_outcome(self):
        ''' Simulates outcome through MS sampling.
        '''
        self.reset()
        self.w = cc.calcWstate(self.state_string, self.state_sparam)
        self.sample()
        for i in range(self.L):
            # print(self.gates_string[i])
            # print(self.traj)
            # print(self.est)
            # print(i, self.L)
            # print(self.gates_sparam[i])
            self.w = cc.calcWgate(self.gates_string[i], self.gates_sparam[i],
                                  self.traj[-1])
            self.sample()
        self.w = cc.calcWmeas(self.meas_string, self.meas_sparam,
                              self.traj[-1])

    def estimate_outcome(self, N=100):
        ''' Estimates outcome by performing s simulations.
        '''
        p_est = np.zeros(N)
        for i in range(N):
            self.simulate_outcome()

            self.traj = np.array(self.traj)
            self.est = np.array(self.est)
            self.M = np.array(self.M)
            # self.Mforw = np.prod(self.M) #!!!

            p_est[i] = np.prod(self.est*self.M)*self.w
        return p_est.sum()/N

    def calc_hoeffding_bound(self, eps, delta):
        return int( 2/eps**2 * self.Mforw**2 * np.log(2/delta) )



def random_gates_string(n, L):
    ''' Creates a random circuit sequence as input for a Circuit object.
        n - integer - number of qudits
        L - integer - circuit depth (number of gate levels)
    '''
    prob = [1., 0.] # Probability of 1-qudit gate or 2-qudit gate
    char1q, probs1q = ['H', 'S', '1'], [1/3, 1/3, 1/3] # Parameters for 1-qudit

    gates_string = []
    for i in range(L):
        gate_string = ''
        q1 = nr.choice([1,0], p=prob)
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


def data_scaling(n=4, kmax=3, circ_num=10, L=4, eps=0.05, delta=0.1, s=0.):
    ''' Creates data for Fig.1 of Pashayan et al. (2015)
    '''
    mgcs = 'S' #!!!
    state_string = '0'*(n-kmax) + mgcs*kmax
    meas_string = '0'+'x'*(n-1)
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
    for k in range(1+kmax):
        state_string = '0'*(n-k) + mgcs*k
        for i in range(circ_num):
            gates_string = random_gates_string(n, L)
            print([state_string]+gates_string+[meas_string])
            c = Circuit([state_string]+gates_string+[meas_string], s)
            N = c.calc_hoeffding_bound(eps, delta)

            print('(k,i):', (k,i))
            print('# samples: ', N)
            print('------------')

            c.simulate_circuit()
            p_born = c.prob
            p_est = c.estimate_outcome(N)
            p_diffs[k, i] = np.abs(p_born - p_est)
        p_diffs[k,-1] = N

    np.save(filename, p_diffs)
    return p_diffs

def plot_scaling(n=4, kmax=3, circ_num=10, L=4, eps=0.05, delta=0.1, s=0.):
    ''' Plots data for Fig.1 of Pashayan et al. (2015)
    '''
    err = 5 #!!!
    pdiffs = data_scaling(n, kmax, circ_num, L, eps, delta, s)
    n_sampl = pdiffs[:,-1]
    p_diffs = pdiffs[:,:-1]
    p_diffs[np.where(np.isclose(p_diffs, 0.))]=10**(-err)
    kaxis = np.repeat(np.arange(1+kmax)[:,None], circ_num, axis=1)

    cb = plt.cm.get_cmap('RdYlBu')
    cr = np.repeat(n_sampl[:,None], circ_num, axis=1).flatten()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    vmine = int(np.log10(n_sampl.min()))
    vmaxe = int(np.log10(n_sampl.max())+1)
    sc = ax.scatter(kaxis.flatten(), p_diffs.flatten(), marker='o', c=cr,
                    norm=colors.LogNorm(vmin=10**vmine, vmax=10**vmaxe),
                    cmap=cb)
    cb = fig.colorbar(sc, ticks=10**np.arange(vmine-1, vmaxe+1))
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
    return pdiffs

s= [
    [0.1,0.0,0.0],
    [ [0.0,0.0,0.0],[+0.0,0.2,0.1] ],
    [ [0.0,0.0,0.3],[-0.2,0.0,0.2] ],
    [0.0,0.0,0.5]
    ]














