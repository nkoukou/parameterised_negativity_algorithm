from QUBIT_frame_opt import (frame_opt, neg_circuit)
from QUBIT_circuit_generator import (random_connected_circuit, show_connectivity)
# from QUBIT_circuit_generator import (random_connected_circuit, random_circuit,
#        compress2q_circuit, compress3q_circuit, string_to_circuit,
#        show_connectivity, solve_qubit_circuit, random_connected_circuit_2q3q)
from QUBIT_get_prob import (get_prob_list,get_prob_list_3q)
from QUBIT_sample import (sample_circuit,sample_circuit_3q)
# from QUBIT_circuit_components import(makeState, makeGate, makeMeas)
from QUBIT_compression import (compress_circuit)

class QD_circuit(object):
    ''' Represents quasi-probability circuit.
    '''
    def __init__(self, circuit, n=3, l=3):
        '''
        circuit = {'state_list': states, 'gate_list': gates,
                   'index_list': indices, 'meas_list': measurements}
        '''
        self.circuit = circuit
        self.n = n
        self.l = l

        self.is_compressed = False
        self.circuit_compressed = None

        self.x_list_opt = 0
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

    def get_neg_circuit(self, wigner=False):
        '''
        '''
        x_list = 0 if wigner else self.x_list_opt

        if self.is_compressed:
            self.log_neg_tot = neg_circuit(self.circuit_compressed, x_list)
        else:
            self.log_neg_tot = neg_circuit(self.circuit, x_list)
        return self.log_neg_tot

    def opt_x(self, **kwargs):
        ''' Optimise quasi-probability distribution.
        '''
        options = {'niter': 3}
        options.update(kwargs)

        if self.is_compressed:
            x_opt_list, log_neg_tot = frame_opt(self.circuit_compressed, self.n, self.l, **kwargs)
        else:
            x_opt_list, log_neg_tot = frame_opt(self.circuit, self.n, self.l, **kwargs)

        self.x_list_opt = x_opt_list
        self.log_neg_tot = log_neg_tot

        return x_opt_list, log_neg_tot

    def get_QD_list(self, method='Wigner', **kwargs):
        ''' Get quasi-prob, negativity, sign, & normalised probability.

            method - string ('Wigner', 'Local_opt', 'Opt')
        '''
        if method=='Wigner':
            x_list = self.x_list_wig
        elif method=='Opt':
            x_list = self.x_list_opt
        else:
            raise Exception('Invalid optimisation method.')

        if self.n==2:
            return get_prob_list(self.circuit_compressed, x_list, **kwargs)
        if self.n==3:
            return get_prob_list_3q(self.circuit_compressed, x_list, **kwargs)


    def sample(self, method='Wigner', sample_size=int(1e5), **kwargs):
        ''' Sample the circuit

            method - string ('Wigner', 'Local_opt', 'Opt')

            Output: the all sampled outcome list
        '''
        QD_output = self.get_QD_list(method, **kwargs)

        if self.n==2:
            sample_list = sample_circuit(self.circuit_compressed, QD_output,
                                         sample_size, **kwargs)
        if self.n==3:
            sample_list = sample_circuit_3q(self.circuit_compressed,
                                            QD_output, sample_size, **kwargs)

        return sample_list


######## EXAMPLE CODE #########
n=3
l=4
cir = random_connected_circuit(qudit_num=10, circuit_length=30, Tgate_prob=1/3,
                   given_state=None, given_measurement=5, method='r')

circuit = QD_circuit(cir, n, l)

print("Before the compression")
circuit.show_connectivity(compressed=False)
print("Wigner negativity before compression :", circuit.get_neg_circuit(wigner=True))
print("---------------------------------------------------------------------------------------------")
print("After the compression")
circuit.compress_circuit()
circuit.show_connectivity(compressed=True)
print("Wigner negativity after the compression with n =", n,":", circuit.get_neg_circuit(wigner=True))
print("---------------------------------------------------------------------------------------------")
print("Perform the frame optimisation")
circuit.opt_x()
print("Total negativity after the frame optimisation with l =", l, ":", circuit.get_neg_circuit(wigner=False))




