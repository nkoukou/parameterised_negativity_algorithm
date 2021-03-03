import numpy as np
import numpy.random as nr
import time
import itertools as it

def sample_circuit(compressed_circuit, QD_output, sample_size = 10000, **kwargs):
    options = {'option 1': True}
    options.update(kwargs)
    
    def sample_iter():
        current_PS_point = []
        p_estimate = 1.
       
        # Input states
        qudit_index = 0
        for PD_state in QD_output['PD_list_states']:
            prob = PD_state.flatten()
            neg = QD_output['neg_list_states'][qudit_index]
            sign_array = QD_output['sign_list_states'][qudit_index].flatten()
            
            PS_point = nr.choice(np.arange(len(prob)), p=prob)
            current_PS_point.append(PS_point)
            p_estimate *= neg*sign_array[PS_point]
            qudit_index +=1

        # Gates
        gate_index = 0
        for PD_gate in QD_output['PD_list_gates']:
            idx = compressed_circuit['qudit_index_list'][gate_index]
        
            p_in1 = current_PS_point[idx[0]]//2
            q_in1 = current_PS_point[idx[0]]%2
            p_in2 = current_PS_point[idx[1]]//2
            q_in2 = current_PS_point[idx[1]]%2

            prob = PD_gate[p_in1,q_in1,p_in2,q_in2].flatten()
            neg = QD_output['neg_list_gates'][gate_index][p_in1,q_in1,p_in2,q_in2]
            sign_array = QD_output['sign_list_gates'][gate_index][p_in1,q_in1,p_in2,q_in2].flatten()
            
            PS_point = nr.choice(np.arange(len(prob)), p=prob)
            current_PS_point[idx[0]] = PS_point//4
            current_PS_point[idx[1]] = PS_point%4
            p_estimate *= neg*sign_array[PS_point]
            
            gate_index += 1
        
        # Measurement
        meas_index = 0
        qudit_index = 0
        for meas in compressed_circuit['meas_list']:
            if str(meas) =='/':
                qudit_index += 1
            else:
                WF = QD_output['QD_list_meas'][meas_index].flatten()  #THIS SHOULD BE QD NOT PD
                meas_index += 1
                p_estimate *= WF[current_PS_point[qudit_index]]
                qudit_index += 1
        return p_estimate
    
    '''
    The function actually performing the Monte Carlo sampling as outlined in Pashayan et al. (2015)
    for the compressed version of the given circuit and the parameter list.
    It prints the method we used, the sampling result (p_estimate), and the computation time.
    It also returns the sampling result and the full list of p_estimate of each iteration ('plot').
    '''
    
    start_time = time.time()
    outcome_list = []
    for n in range(sample_size):
        outcome_list.append(sample_iter())
    sampling_time = time.time() - start_time

    # Print the final results
    print('==================Sampling Results=============================')
    print('p_estimate: ', np.sum(outcome_list)/sample_size)
    print('Sample size: ', sample_size)
    print('Sampling time: ', sampling_time)
    print('===============================================================')

    return np.array(outcome_list)
