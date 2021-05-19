from QUBIT_wig_neg import (wigner_neg_compressed, wigner_neg_compressed_3q)
from QUBIT_opt_neg import (optimize_neg_compressed, optimize_neg_compressed_3q)
from QUBIT_local_opt_neg import (local_opt_neg_compressed,
                                 local_opt_neg_compressed_3q)
from QUBIT_circuit_generator import (random_connected_circuit, random_circuit,
       compress2q_circuit, compress3q_circuit, string_to_circuit,
       show_connectivity, solve_qubit_circuit, random_connected_circuit_2q3q)
from QUBIT_get_prob import (get_prob_list,get_prob_list_3q)
from QUBIT_sample import (sample_circuit,sample_circuit_3q)
from QUBIT_BVcircuit import(BValg_circuit)
from QUBIT_circuit_components import(makeState, makeGate, makeMeas)

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

    def show_connectivity(self, compressed=True):
        circuit = self.circuit_compressed if compressed else self.circuit
        self.diagram = show_connectivity(circuit)

    def opt_x(self, method = 'Wigner', m=2, **kwargs):
        ''' Optimise quasi-probability distribution.

            method - string ('Wigner', 'Local_opt', 'Opt')
        '''
        options = {'opt_method': 'B', 'niter': 3}
        options.update(kwargs)
        if method=='Wigner':
            if m==2: x_opt_list, log_neg_tot = wigner_neg_compressed(
                    self.circuit_compressed, **kwargs)
            if m==3: x_opt_list, log_neg_tot = wigner_neg_compressed_3q(
                    self.circuit_compressed, **kwargs)
            self.x_list_wig = x_opt_list
        elif method=='Local_opt':
            if m==2: x_opt_list, log_neg_tot = local_opt_neg_compressed(
                     self.circuit_compressed, **kwargs)
            if m==3: x_opt_list, log_neg_tot = local_opt_neg_compressed_3q(
                     self.circuit_compressed, **kwargs)
            self.x_list_local_opt = x_opt_list
        elif method=='Opt':
            if m==2: x_opt_list, log_neg_tot = optimize_neg_compressed(
                     self.circuit_compressed, **kwargs)
            if m==3: x_opt_list, log_neg_tot = optimize_neg_compressed_3q(
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
            sample_list = sample_circuit(self.circuit_compressed, QD_output,
                                         sample_size, **kwargs)
        if m==3:
            sample_list = sample_circuit_3q(self.circuit_compressed,
                                            QD_output, sample_size, **kwargs)

        return sample_list


################################ SAMPLE CODE ##################################
if __name__== "__main__":
    import time
    import autograd.numpy as np
    import matplotlib.pylab as plt
    import pickle

    # circuit, Tcount = random_connected_circuit(
    #         qudit_num=7, circuit_length=11,
    #         Tgate_prob =(15/(2*11)),
    #         given_state=None, given_measurement=2, method='c')
    # circuit, Tcount, toffoli_num = random_connected_circuit_2q3q(
    #     qudit_num=4, circuit_length=10, Tgate_prob=1/3, prob_2q=1,
    #     given_state=None, given_measurement=4)
    circuit = random_circuit(qudit_num=3, C1qGate_num=1, TGate_num=2,
                             CSUMGate_num=4, Toff_num=5,
                   given_state=None, given_measurement=1)

    # np.save("test_circuit", circuit)
    # outfile = open("test_circuit", 'wb')
    # pickle.dump(circuit, outfile)
    # outfile.close()
    # infile = open("test_circuit",'rb')
    # circuit = pickle.load(infile)
    # infile.close()

    circ = QD_circuit(circuit)
    circ.show_connectivity(compressed=False)

    sample_size = int(5)
    x_list = np.linspace(1, sample_size, sample_size)


    # print("\n------------------ 2q-compression -------------------\n")
    # circ.compress_circuit(m=2)
    # # circ.show_connectivity()
    # pborn1 = solve_qubit_circuit(circ.circuit)
    # pborn2 = solve_qubit_circuit(circ.circuit_compressed)
    # print("(2q-compression) Probs:", np.allclose(pborn1, pborn2),
    #   "(%.4f, %.4f)"%(pborn1, pborn2))
    # if not np.allclose(pborn1, pborn2):
    #     raise Exception('(2q-compression) Probs: NOT equal')

    # t0 = time.time()
    # circ.opt_x(method='Wigner')
    # t1 = time.time()
    # circ.get_QD_list(method='Wigner')
    # t2 = time.time()
    # wigner_out_list = circ.sample(method='Wigner', sample_size=sample_size)
    # t3 = time.time()
    # prob_wigner = np.cumsum(wigner_out_list)/x_list
    # print("opt_x: %.2f"%(t1-t0))
    # print("get_QD_list: %.2f"%(t2-t1))
    # print("sample: %.2f"%(t3-t2))

    # circ.opt_x(method='Opt', **{'niter':10})
    # circ.get_QD_list(method = 'Opt')
    # opt_out_list = circ.sample(method='Opt', sample_size=sample_size)
    # prob_opt = np.cumsum(opt_out_list)/x_list

    # circ.opt_x(method='Local_opt', **{'niter':10})
    # circ.get_QD_list(method='Local_opt')
    # local_opt_out_list = circ.sample(method='Local_opt',
    #                                   sample_size=sample_size)
    # prob_local_opt = np.cumsum(local_opt_out_list)/x_list



    print("\n----------------- 3q-compression --------------------\n")
    circ.compress_circuit(m=3)
    circ.show_connectivity()
    pborn1 = solve_qubit_circuit(circ.circuit)
    pborn2 = solve_qubit_circuit(circ.circuit_compressed)
    print("(3q-compression) Probs:", np.allclose(pborn1, pborn2),
          "(%.4f, %.4f)"%(pborn1, pborn2))
    if not np.allclose(pborn1, pborn2):
        raise Exception('(2q-compression) Probs: NOT equal')

    # t0 = time.time()
    # circ.opt_x(method='Wigner', m=3)
    # t1 = time.time()
    # circ.get_QD_list(method='Wigner', m=3)
    # t2 = time.time()
    # wigner_out_list = circ.sample(method='Wigner', m=3,
    #                               sample_size=sample_size)
    # t3 = time.time()
    # print("opt_x: %.2f"%(t1-t0))
    # print("get_QD_list: %.2f"%(t2-t1))
    # print("sample: %.2f"%(t3-t2))
    # prob_wigner = np.cumsum(wigner_out_list)/x_list

    # circ.opt_x(method='Opt', m=3, **{'niter':10})
    # circ.get_QD_list(method = 'Opt', m=3)
    # opt_out_list = circ.sample(method='Opt', m=3, sample_size=sample_size)
    # prob_opt = np.cumsum(opt_out_list)/x_list

    # circ.opt_x(method='Local_opt', m=3, **{'niter':10})
    # circ.get_QD_list(method='Local_opt', m=3)
    # local_opt_out_list = circ.sample(method='Local_opt', m=3,
    #                                   sample_size=sample_size)
    # prob_local_opt = np.cumsum(local_opt_out_list)/x_list


    # plt.close('all')
    # plt.plot(x_list, prob_wigner, linestyle='solid',
    #           color='tab:blue', label='Wigner')
    # plt.plot(x_list, prob_opt, linestyle='solid',
    #           color='tab:orange', label='Optimised')
    # plt.plot(x_list, prob_local_opt, linestyle='solid',
    #           color='tab:red', label='Local_Optimised')
    # plt.hlines(y=pborn1, xmin=0, xmax= sample_size, c='k', ls='-')
    # plt.xlabel('sample #')
    # plt.ylabel('p_estimate')
    # plt.legend(loc='upper right')
    # plt.show()
