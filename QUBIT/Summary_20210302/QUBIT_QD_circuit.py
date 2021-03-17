from QUBIT_wig_neg import (wigner_neg_compressed)
from QUBIT_opt_neg import (optimize_neg_compressed)
from QUBIT_local_opt_neg import (local_opt_neg_compressed)
from QUBIT_circuit_generator import (random_connected_circuit, random_circuit,
       compress2q_circuit, compress3q_circuit, string_to_circuit,
       show_connectivity)
from QUBIT_get_prob import (get_prob_list)
from QUBIT_sample import (sample_circuit)
from QUBIT_BVcircuit import(BValg_circuit)

class QD_circuit(object):
    ''' Represents quasi-probability circuit.
    '''
    def __init__(self, circuit):
        '''
        circuit = {'state_list': states, 'gate_list': gates,
                   'index_list': indices, 'meas_list': measurements}
        '''
        self.circuit = circuit
        self.circuit_compressed = None

        self.x_list_wig = None
        self.x_list_opt = None
        self.x_list_local_opt = None
        self.log_neg_tot = None

    def compress_circuit(self, m=2):
        ''' Compresses circuit, so that it consists of m-qudit gates.
        '''
        if m==2:
            self.circuit_compressed = compress2q_circuit(self.circuit)
        elif m==3:
            self.circuit_compressed = compress3q_circuit(self.circuit)
        else:
            raise Exception('m must be 2 or 3')

    def show_connectivity(self, compressed=True):
        circuit = self.circuit_compressed if compressed else self.circuit
        self.diagram = show_connectivity(circuit)

    def opt_x(self, method = 'Wigner', **kwargs):
        ''' Optimise quasi-probability distribution.

            method - string ('Wigner', 'Local_opt', 'Opt')
        '''
        options = {'opt_method': 'B', 'niter': 3}
        options.update(kwargs)
        if method=='Wigner':
            x_opt_list, log_neg_tot = wigner_neg_compressed(
                self.circuit_compressed, **kwargs)
            self.x_list_wig = x_opt_list
        elif method=='Local_opt':
            x_opt_list, log_neg_tot = local_opt_neg_compressed(
                self.circuit_compressed, **kwargs)
            self.x_list_local_opt = x_opt_list
        elif method=='Opt':
            x_opt_list, log_neg_tot = optimize_neg_compressed(
                self.circuit_compressed, **kwargs)
            self.x_list_opt = x_opt_list
        else:
            raise Exception('Invalid optimisation method.')
        self.log_neg_tot = log_neg_tot
        return x_opt_list, log_neg_tot

    def get_QD_list(self, method='Wigner', **kwargs):
        ''' Get quasi-prob, negativity, sign, & normalised probability.

            method - string ('Wigner', 'Local_opt', 'Opt')
        '''
        if method=='Wigner':
            x_list = self.x_list_wig
        elif method=='Local_opt':
            x_list = self.x_list_local_opt
        elif method=='Opt':
            x_list = self.x_list_opt
        else:
            raise Exception('Invalid optimisation method.')
        return get_prob_list(self.circuit_compressed, x_list, **kwargs)

    def sample(self, method='Wigner', sample_size=int(1e5), **kwargs):
        ''' Sample the circuit

            method - string ('Wigner', 'Local_opt', 'Opt')

            Output: the all sampled outcome list
        '''
        QD_output = self.get_QD_list(method, **kwargs)
        sample_list = sample_circuit(self.circuit_compressed, QD_output,
                                     sample_size, **kwargs)
        return sample_list


################################ SAMPLE CODE ##################################
import time
import autograd.numpy as np
import matplotlib.pylab as plt

t0 = time.time()

Bernstein_Vazirani_circuit = BValg_circuit('11', 0)

# circuit, Tcount = random_connected_circuit(qudit_num=4, circuit_length=10,
#            Tgate_prob=1/3, given_state=None, given_measurement=2, method='c')

circuit = random_circuit(qudit_num=12, C1qGate_num=243, TGate_num=134,
                         CSUMGate_num=25, Toff_num=10,
                         given_state=0, given_measurement=2)

circ = QD_circuit(circuit)
circ.compress_circuit(m=3)
circ.show_connectivity(compressed=False)
print()
circ.show_connectivity()


# p1 = solve_qubit_circuit(circ.circuit) # Not implemented yet
# p2 = solve_qubit_circuit(circ.circuit_compressed) # Not implemented yet

# sample_size = int(1e2)
# x_list = np.linspace(1, sample_size, sample_size)

# circ.opt_x(method='Wigner')
# circ.get_QD_list(method='Wigner')
# wigner_out_list = circ.sample(method='Wigner', sample_size=sample_size)
# prob_wigner = np.cumsum(wigner_out_list)/x_list

# circ.opt_x(method='Opt', **{'niter':10})
# circ.get_QD_list(method = 'Opt')
# opt_out_list = circ.sample(method='Opt', sample_size=sample_size)
# prob_opt = np.cumsum(opt_out_list)/x_list

# circ.opt_x(method='Local_opt', **{'niter':10})
# circ.get_QD_list(method='Local_opt')
# local_opt_out_list = circ.sample(method='Local_opt',
#                                   sample_size=sample_size)
# prob_local_opt = np.cumsum(local_opt_out_list)/x_list


# plt.close('all')
# plt.plot(x_list, prob_wigner, linestyle='solid',
#           color='tab:blue', label='Wigner')
# plt.plot(x_list, prob_opt, linestyle='solid',
#           color='tab:orange', label='Optimised')
# plt.plot(x_list, prob_local_opt, linestyle='solid',
#           color='tab:red', label='Local_Optimised')
# plt.xlabel('sample #')
# plt.ylabel('p_estimate')
# plt.legend(loc='upper right')
# plt.show()
