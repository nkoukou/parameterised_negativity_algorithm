from frame_opt import(wigner_neg_compressed, wigner_neg_compressed_3q,
        optimize_neg_compressed, optimize_neg_compressed_3q,
		local_opt_neg_compressed, local_opt_neg_compressed_3q)
from circuit_generator import(random_connected_circuit,
    	compress2q_circuit, string_to_circuit, show_connectivity,
    	solve_qubit_circuit, compress3q_circuit)
from sample import(get_prob_list, get_prob_list_3q,
        sample_circuit, sample_circuit_3q)


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
        	self.circuit_compressed = compress2q_circuit(self.circuit)
        	self.circuit_compressed = compress3q_circuit(
                                                    self.circuit_compressed)
        else:
            raise Exception('m must be 2 or 3')
        # !!! Change to compress_circuit

    def show_connectivity(self, compressed=True):
        circuit = self.circuit_compressed if compressed else self.circuit
        self.diagram = show_connectivity(circuit)

    def opt_x(self, method = 'Wigner', **kwargs):
        ''' Optimise quasi-probability distribution.

            method - string ('Wigner', 'Local_opt', 'Opt')
        '''
        options = {'opt_method': 'B', 'niter': 1}
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

    def opt_x_3q(self, method = 'Wigner', **kwargs):
        ''' Optimise quasi-probability distribution.

            method - string ('Wigner', 'Local_opt', 'Opt')
        '''
        options = {'opt_method': 'B', 'niter': 1}
        options.update(kwargs)
        if method=='Wigner':
            x_opt_list, log_neg_tot = wigner_neg_compressed_3q(
                self.circuit_compressed, **kwargs)
            self.x_list_wig = x_opt_list
        elif method=='Local_opt':
            x_opt_list, log_neg_tot = local_opt_neg_compressed_3q(
                self.circuit_compressed, **kwargs)
            self.x_list_local_opt = x_opt_list
        elif method=='Opt':
            x_opt_list, log_neg_tot = optimize_neg_compressed_3q(
                self.circuit_compressed, **kwargs)
            self.x_list_opt = x_opt_list
        else:
            raise Exception('Invalid optimisation method.')
        self.log_neg_tot = log_neg_tot
        return x_opt_list, log_neg_tot

    def get_QD_list(self, method='Wigner', m=2, **kwargs):
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

        if m==2:
            return get_prob_list(self.circuit_compressed, x_list, **kwargs)
        if m==3:
            return get_prob_list_3q(self.circuit_compressed, x_list, **kwargs)

    def sample(self, method='Wigner', m=2, sample_size=int(1e5), **kwargs):
        ''' Sample the circuit

            method - string ('Wigner', 'Local_opt', 'Opt')

            Output: the all sampled outcome list
        '''
        QD_output = self.get_QD_list(method, m, **kwargs)

        if m==2:
            sample_list = sample_circuit(self.circuit_compressed, QD_output, sample_size, **kwargs)
        if m==3:
            sample_list = sample_circuit_3q(self.circuit_compressed, QD_output, sample_size, **kwargs)
        
        return sample_list


################################ SAMPLE CODE ##################################
if __name__== "__main__":
    import time
    import autograd.numpy as np
    import matplotlib.pylab as plt



    circuit, Tcount = random_connected_circuit(qudit_num=4, circuit_length=10,
                Tgate_prob=2/3, given_state=None, given_measurement=2, method='c')

    p_born = solve_qubit_circuit(circuit)

    print("The actual Born probability: ", p_born)

    circ_2q = QD_circuit(circuit)
    circ_2q.show_connectivity(compressed=False)
    print("\n------------------2q compression-------------------\n")
    circ_2q.compress_circuit(m=2)
    circ_2q.show_connectivity()	

    print("\n------------------3q compression-------------------\n")
    circ_3q = QD_circuit(circuit)
    circ_3q.compress_circuit(m=3)
    circ_3q.show_connectivity()

    ## SAMPLING ##
    sample_size = int(1e3)
    x_list = np.linspace(1, sample_size, sample_size)

    # 2q-compression
    circ_2q.opt_x(method='Wigner')
    wigner_out_list_2q = circ_2q.sample(method='Wigner', m=2, sample_size=sample_size)
    prob_wigner_2q = np.cumsum(wigner_out_list_2q)/x_list

    circ_2q.opt_x(method='Opt', **{'niter':1})
    opt_out_list_2q = circ_2q.sample(method='Opt', m=2, sample_size=sample_size)
    prob_opt_2q = np.cumsum(opt_out_list_2q)/x_list

    circ_2q.opt_x(method='Local_opt', **{'niter':1})
    local_opt_out_list_2q = circ_2q.sample(method='Local_opt', m=2, 
                                      sample_size=sample_size)
    prob_local_opt_2q = np.cumsum(local_opt_out_list_2q)/x_list

    # 3q-compression
    circ_3q.opt_x_3q(method='Wigner')
    wigner_out_list_3q = circ_3q.sample(method='Wigner', m=3, sample_size=sample_size)
    prob_wigner_3q = np.cumsum(wigner_out_list_3q)/x_list

    circ_3q.opt_x_3q(method='Opt', **{'niter':1})
    opt_out_list_3q = circ_3q.sample(method='Opt', m=3, sample_size=sample_size)
    prob_opt_3q = np.cumsum(opt_out_list_3q)/x_list

    circ_3q.opt_x_3q(method='Local_opt', **{'niter':1})
    local_opt_out_list_3q = circ_3q.sample(method='Local_opt', m=3,
                                      sample_size=sample_size)
    prob_local_opt_3q = np.cumsum(local_opt_out_list_3q)/x_list

    p_born_list = [p_born]*len(x_list)

    ## Save the data
    path = 'prob_wigner_2q.txt'
    with open(path, 'a') as f:
        for n in range(len(prob_wigner_2q)):
            f.write(str(prob_wigner_2q[n])+"\n")
    f.close()

    path = 'prob_opt_2q.txt'
    with open(path, 'a') as f:
        for n in range(len(prob_opt_2q)):
            f.write(str(prob_opt_2q[n])+"\n")
    f.close()

    path = 'prob_local_opt_2q.txt'
    with open(path, 'a') as f:
        for n in range(len(prob_local_opt_2q)):
            f.write(str(prob_local_opt_2q[n])+"\n")
    f.close()

    path = 'prob_wigner_3q.txt'
    with open(path, 'a') as f:
        for n in range(len(prob_wigner_3q)):
            f.write(str(prob_wigner_3q[n])+"\n")
    f.close()

    path = 'prob_opt_3q.txt'
    with open(path, 'a') as f:
        for n in range(len(prob_opt_3q)):
            f.write(str(prob_opt_3q[n])+"\n")
    f.close()

    path = 'prob_local_opt_3q.txt'
    with open(path, 'a') as f:
        for n in range(len(prob_local_opt_3q)):
            f.write(str(prob_local_opt_3q[n])+"\n")
    f.close()

    ## Plot the data
    plt.close('all')
    plt.plot(x_list, prob_wigner_2q, linestyle=':', label='Wigner')
    plt.plot(x_list, prob_opt_2q, linestyle=':', label='Optimised')
    plt.plot(x_list, prob_local_opt_2q, linestyle=':', label='Local_Optimised')
    plt.plot(x_list, prob_wigner_3q, linestyle='solid', label='Wigner 3q-compression')
    plt.plot(x_list, prob_opt_3q, linestyle='solid', label='Optimised 3q-compression')
    plt.plot(x_list, prob_local_opt_3q, linestyle='solid', label='Local_Optimised 3q-compression')
    plt.plot(x_list, p_born_list, linestyle='solid', label='Born probablity')
    plt.xlabel('sample #')
    plt.ylabel('p_estimate')
    plt.legend(loc='upper right')
    plt.show()