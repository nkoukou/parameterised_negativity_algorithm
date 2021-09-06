from QUBIT_circuit_generator import (haar_random_connected_circuit,
                                     show_connectivity)
from QUBIT_compression import (compress_circuit)
from QUBIT_frame_opt import (frame_opt, neg_circuit)
from QUBIT_get_prob import (get_prob_list)
from QUBIT_sample import (sample_circuit, sample_circuit_3q)
from QUBIT_phase_space import (W_state, W_gate, W_meas)

class QD_circuit(object):
    ''' Represents quasi-probability circuit.
    '''
    def __init__(self, circuit, n, l, DIM, par0, W):
        '''
        circuit = {'state_list': states, 'gate_list': gates,
                   'index_list': indices, 'meas_list': measurements}
        '''
        self.circuit = circuit
        self.N = len(self.circuit['state_list'])
        self.L = len(self.circuit['gate_list'])
        self.n = n
        self.l = l
        self.DIM = DIM
        self.par0 = par0
        self.W = W
        self.W_state = W[0]
        self.W_gate = W[1]
        self.W_meas = W[2]

        self.is_compressed = False
        self.circuit_compressed = None

        self.par_len = self.get_par_len()
        self.par_list_opt = par0*self.par_len
        self.log_neg_tot = None

    def compress_circuit(self):
        '''
        Compresses circuit, so that it only consists of n-qudit gates.
        INPUT - n : spatial parameter
        '''
        self.is_compressed = True
        self.circuit_compressed = compress_circuit(self.circuit, self.n)

    def show_connectivity(self, compressed=True):
        circuit = self.circuit_compressed if compressed else self.circuit
        self.diagram = show_connectivity(circuit)

    def get_par_len(self):
        n_state = len(self.circuit['state_list'])
        par_len = n_state + sum([len(index)
                                 for index in self.circuit['index_list']])
        return par_len

    def get_neg_circuit(self, ref=False):
        '''
        Calculate the negativity of the circuit
        INPUT - ref: whether using the reference frames.
        '''
        par_list = self.par0*self.par_len if ref else self.par_list_opt
        circuit = self.circuit_compressed if self.is_compressed \
                                          else self.circuit

        self.log_neg_tot = neg_circuit(circuit, self.W, par_list, self.par0)
        return self.log_neg_tot

    def opt_x(self, **kwargs):
        ''' Frame optimisation with temporal parameter 'l'
        '''
        options = {'niter': 3}
        options.update(kwargs)

        circuit = self.circuit_compressed if self.is_compressed \
                                          else self.circuit
        par_opt_list, log_neg_tot = frame_opt(circuit, self.l, self.par0,
                                              self.W, **kwargs)

        self.par_list_opt = par_opt_list
        self.log_neg_tot = log_neg_tot

        return par_opt_list, log_neg_tot

    def get_QD_list(self, opt=True, **kwargs):
        ''' Get quasi-prob, negativity, sign, & normalised probability.

            method - string ('refere', 'opt')
        '''

        if opt:
            par_list = self.par_list_opt
        else:
            print("Using the reference frames:")
            par_list = self.par0*self.par_len

        circuit = self.circuit_compressed if self.is_compressed \
                                          else self.circuit
        return get_prob_list(circuit, par_list, self.par0, self.W, **kwargs)

    def sample(self, opt=True, sample_size=int(1e5), **kwargs):
        ''' Sample the circuit

            method - string ('Wigner', 'Local_opt', 'Opt')

            Output: the all sampled outcome list
        '''
        QD_output = self.get_QD_list(opt, **kwargs)

        if self.n==2:
            sample_list = sample_circuit(self.circuit_compressed, QD_output,
                                         sample_size, **kwargs)
        elif self.n==3:
            sample_list = sample_circuit_3q(self.circuit_compressed,
                                            QD_output, sample_size, **kwargs)
        else:
            raise Exception("Sampling not implemented for n>3")

        return sample_list


######## EXAMPLE CODE #########
dim = 2
circ = haar_random_connected_circuit(N=10, L=40, n=2, d=dim,
          given_state=None, given_meas=1, method='r')

n=2
l=4
x0 = [1,1/2,1/2]
qp_function = [W_state, W_gate, W_meas]
circuit = QD_circuit(circ, n, l, dim, x0, qp_function)

print("\nBefore compression:\n")
circuit.show_connectivity(compressed=False)
temp = circuit.get_neg_circuit(ref=True)
print("\nAfter compression:\n")
circuit.compress_circuit()
circuit.show_connectivity(compressed=True)
print("\nPerforming frame optimisation...\n")
circuit.opt_x()
print("Optimisation done.")
print("-------------------------------------------------------------")
print("Circuit negativity N at each step:")
print("\n1. N = %.3f (initial Wigner neg)"%(temp))
print("\n2. N = %.3f (after compression with n=%d)"%(
    circuit.get_neg_circuit(ref=True), n))
print("\n3. N = %.3f (after frame optimisation with l=%d)"%(
    circuit.get_neg_circuit(ref=False), l))




