from QUBIT_wig_neg import (wigner_neg_compressed)
from QUBIT_opt_neg import (optimize_neg_compressed)
from QUBIT_local_opt_neg import (local_opt_neg_compressed)
from QUBIT_get_prob import (get_prob_list)
from QUBIT_sample import (sample_circuit)

class QD_circuit(object):
#### Both circuit and compressed_circuit are labeled lists of the following form: {'state_list', 'gate_list', 'qudit_index_list','meas_list'} ####
    def __init__(self, circuit):
        self.circuit = circuit

#### Non-symbolic circuit compression####
    def compress_circuit(self, **kwargs):
        cc = self.circuit #Need to be added#
        self.compressed_circuit = cc
        return cc

#### Optimisation of quasi-probability distribution####
    def opt_x(self, Method = 'Wigner', **kwargs):
        options = {'opt_method': 'B','niter': 3}
        options.update(kwargs)
        if Method == 'Wigner':
            x_opt_list, log_neg_tot = wigner_neg_compressed(self.compressed_circuit, **kwargs)
            self.x_list_wig = x_opt_list
        elif Method == 'Local_opt':
            x_opt_list, log_neg_tot = local_opt_neg_compressed(self.compressed_circuit, **kwargs)    ### Old name: get_local_opt_x in QUBIT_local_opt_neg.py (old version)
            self.x_list_local_opt = x_opt_list
        elif Method == 'Opt':
            x_opt_list, log_neg_tot = optimize_neg_compressed(self.compressed_circuit, **kwargs)     ### Old function: optimize_neg_compressed in QUBIT_opt_neg.py (old version)
            self.x_list_opt = x_opt_list
        return x_opt_list, log_neg_tot

#### Get quasi-prob, negativity, sign, & normalised probabilit####
    def get_QD_list(self, Method = 'Wigner', **kwargs):
        if Method == 'Wigner':
            x_list = self.x_list_wig
        elif Method == 'Local_opt':
            x_list = self.x_list_local_opt
        elif Method == 'Opt':
            x_list = self.x_list_opt
        return get_prob_list(self.compressed_circuit,x_list, **kwargs)  ### Old function: See 'QUBIT_prob_estimation.py'

#### Sample the circuit: # Output is the all sampled outcome list ####
    def sample(self, Method = 'Wigner', sample_size=10000, **kwargs):
        QD_output = self.get_QD_list(Method, **kwargs)
        sample_list = sample_circuit(self.compressed_circuit, QD_output, sample_size, **kwargs) ### Old function: See 'QUBIT_prob_estimation.py'
        return sample_list


######################################################SAMPLE CODE###########################################################
if __name__ == "__main__":
    from QUBIT_temp_functions import (string_to_circuit, accum_sum)
    import autograd.numpy as np
    import matplotlib.pylab as plt

    Bernstein_Vazirani_circuit = ['001', [[[0],'H'], [[2],'H'], [[2],'H'],
                                          [[1,2],'C+'], [[2],'t'], [[0,2],'C+'],
                                          [[2],'T'], [[1,2],'C+'], [[2],'t'],
                                          [[0,2],'C+'], [[1],'T'], [[2],'T'],
                                          [[0,1],'C+'], [[2],'H'], [[0],'T'],
                                          [[1],'t'], [[0,1],'C+'], [[0],'H']],
                                  '0//']

    circuit = string_to_circuit(Bernstein_Vazirani_circuit)

    AA = QD_circuit(circuit)
    AA.compress_circuit()

    sample_size = 1000000

    AA.opt_x(Method = 'Wigner')
    AA.get_QD_list(Method = 'Wigner')
    wigner_out_list = AA.sample(Method = 'Wigner', sample_size= sample_size)

    AA.opt_x(Method = 'Opt',**{'niter':10})
    AA.get_QD_list(Method = 'Opt')
    opt_out_list = AA.sample(Method = 'Opt', sample_size= sample_size)

    AA.opt_x(Method = 'Local_opt',**{'niter':10})
    AA.get_QD_list(Method = 'Local_opt')
    local_opt_out_list = AA.sample(Method = 'Local_opt', sample_size = sample_size)
        
    x_list = np.linspace(1,sample_size,sample_size)
    plt.plot(x_list,accum_sum(wigner_out_list)/x_list, linestyle='solid', color='tab:blue', label='Wigner')
    plt.plot(x_list,accum_sum(opt_out_list)/x_list, linestyle='solid', color='tab:orange', label='Optimised')
    plt.plot(x_list,accum_sum(local_opt_out_list)/x_list, linestyle='solid', color='tab:red', label='Local_Optimised')
    plt.xlabel('sample #')
    plt.ylabel('p_estimate')
    plt.legend()
    plt.show()
