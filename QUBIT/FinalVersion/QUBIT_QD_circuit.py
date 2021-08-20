from QUBIT_frame_opt import (frame_opt, neg_circuit)
from QUBIT_circuit_generator import (random_connected_circuit, show_connectivity)
from QUBIT_get_prob import (get_prob_list)
from QUBIT_sample import (sample_circuit, sample_circuit_3q)
from QUBIT_compression import (compress_circuit)
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
        self.circuit_compressed = compress_circuit(self.circuit, n)

    def show_connectivity(self, compressed=True):
        circuit = self.circuit_compressed if compressed else self.circuit
        self.diagram = show_connectivity(circuit)

    def get_par_len(self):
        n_state = len(self.circuit['state_list'])
        par_len = n_state + sum([len(index) for index in self.circuit['index_list']])
        return par_len

    def get_neg_circuit(self, ref=False):
        '''
        Calculate the negativity of the circuit
        INPUT - ref: whether using the reference frames.
        '''
        par_list = self.par0*self.par_len if ref else self.par_list_opt

        if self.is_compressed:
            self.log_neg_tot = neg_circuit(self.circuit_compressed, self.W, par_list, self.par0)
        else:
            self.log_neg_tot = neg_circuit(self.circuit, self.W, par_list, self.par0)
        return self.log_neg_tot

    def opt_x(self, **kwargs):
        ''' Frame optimisation with temporal parameter 'l'
        '''
        options = {'niter': 3}
        options.update(kwargs)

        if self.is_compressed:
            par_opt_list, log_neg_tot = frame_opt(self.circuit_compressed, self.l, self.par0, self.W, **kwargs)
        else:
            par_opt_list, log_neg_tot = frame_opt(self.circuit, self.l, self.par0, self.W, **kwargs)

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
        
        if self.is_compressed:
            return get_prob_list(self.circuit_compressed, par_list, self.par0, self.W, **kwargs)
        else:
            return get_prob_list(self.circuit, par_list, self.par0, self.W, **kwargs)

    def sample(self, opt=True, sample_size=int(1e5), **kwargs):
        ''' Sample the circuit

            method - string ('Wigner', 'Local_opt', 'Opt')

            Output: the all sampled outcome list
        '''
        QD_output = self.get_QD_list(opt, **kwargs)

        if self.n==2:
            sample_list = sample_circuit(self.circuit_compressed, QD_output,
                                         sample_size, **kwargs)
        if self.n==3:
            sample_list = sample_circuit_3q(self.circuit_compressed,
                                            QD_output, sample_size, **kwargs)

        return sample_list


######## EXAMPLE CODE #########
n=3
l=6
x0 = [1,1/2,1/2]
dim = 2
qp_function = [W_state, W_gate, W_meas]
cir = random_connected_circuit(qudit_num=10, circuit_length=40, Tgate_prob=1/3,
                   given_state=None, given_measurement=5, method='r')

circuit = QD_circuit(cir, n, l, dim, x0, qp_function)

print("Before the compression")
circuit.show_connectivity(compressed=False)
print("Wigner negativity before compression :", circuit.get_neg_circuit(ref=True))
print("---------------------------------------------------------------------------------------------")
print("After the compression")
circuit.compress_circuit()
circuit.show_connectivity(compressed=True)
print("Wigner negativity after the compression with n =", n,":", circuit.get_neg_circuit(ref=True))
print("---------------------------------------------------------------------------------------------")
print("Perform the frame optimisation")
circuit.opt_x()
print("Total negativity after the frame optimisation with l =", l, ":", circuit.get_neg_circuit(ref=False))




