import numpy as np
import numpy.random as nr
import time
import itertools as it

from state_functions import DIM
from phase_space import(x2Gamma, W_state_1q, neg_state_1q,
                              W_gate_1q, neg_gate_1q, W_gate_2q,
                              neg_gate_2q, W_meas_1q, neg_meas_1q,
                              W_gate_3q, neg_gate_3q)

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
            idx = compressed_circuit['index_list'][gate_index]

            p_in1 = current_PS_point[idx[0]]//DIM
            q_in1 = current_PS_point[idx[0]]%DIM
            p_in2 = current_PS_point[idx[1]]//DIM
            q_in2 = current_PS_point[idx[1]]%DIM

            prob = PD_gate[p_in1,q_in1,p_in2,q_in2].flatten()
            neg = QD_output['neg_list_gates'][gate_index][p_in1,q_in1,p_in2,q_in2]
            sign_array = QD_output['sign_list_gates'][gate_index][p_in1,q_in1,p_in2,q_in2].flatten()

            PS_point = nr.choice(np.arange(len(prob)), p=prob)
            current_PS_point[idx[0]] = PS_point//(DIM**2)
            current_PS_point[idx[1]] = PS_point%(DIM**2)
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
    print('====================== Sampling Results ======================')
    print('p_estimate:    ', np.sum(outcome_list)/sample_size)
    print('Sample size:   ', sample_size)
    print('Sampling time: ', sampling_time)
    print('==============================================================')

    return np.array(outcome_list)

def sample_circuit_3q(compressed_circuit, QD_output, sample_size = 10000, **kwargs):
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
            idx = compressed_circuit['index_list'][gate_index]

            p_in1 = current_PS_point[idx[0]]//DIM
            q_in1 = current_PS_point[idx[0]]%DIM
            p_in2 = current_PS_point[idx[1]]//DIM
            q_in2 = current_PS_point[idx[1]]%DIM
            p_in3 = current_PS_point[idx[2]]//DIM
            q_in3 = current_PS_point[idx[2]]%DIM

            prob = PD_gate[p_in1,q_in1,p_in2,q_in2,p_in3,q_in3].flatten()
            neg = QD_output['neg_list_gates'][gate_index][p_in1,q_in1,p_in2,q_in2,p_in3,q_in3]
            sign_array = QD_output['sign_list_gates'][gate_index][p_in1,q_in1,p_in2,q_in2,p_in3,q_in3].flatten()

            PS_point = nr.choice(np.arange(len(prob)), p=prob)
            current_PS_point[idx[0]] = PS_point//(DIM**4)
            current_PS_point[idx[1]] = (PS_point%(DIM**4))//(DIM**2)
            current_PS_point[idx[2]] = PS_point%(DIM**2)
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
    print('====================== Sampling Results ======================')
    print('p_estimate:    ', np.sum(outcome_list)/sample_size)
    print('Sample size:   ', sample_size)
    print('Sampling time: ', sampling_time)
    print('==============================================================')

    return np.array(outcome_list)

def get_prob_list(compressed_circuit, x_list, **kwargs):
    options = {'option1': True}
    options.update(kwargs)
    '''
    Calculate the quasi-probability distribution of each circuit element
    and return them as a list of [state part, gate part, measurement part].
    '''

    Gamma_index = 0
    Gammas = []

    QD_list_states = [] # The state part of the quasi-prob. distribution list
    QD_list_gates = [] # The gate part of the quasi-prob. distribution list
    QD_list_meas = [] # The measurement part of the quasi-prob. distribution list

    neg_list_states = [] # The state part of the quasi-prob. negativity list
    neg_list_gates = [] # The gate part of the quasi-prob. negativity list
    neg_list_meas = [] # The gate part of the quasi-prob. negativity list

    sign_list_states = [] # The state part of the sign of quasi-prob. distribution list
    sign_list_gates = [] # The gate part of the sign of the quasi-prob. distribution list
    sign_list_meas = [] # The measurement part of the sign of the quasi-prob. distribution list

    PD_list_states = [] # The state part of the normalised-prob. distribution list
    PD_list_gates = [] # The gate part of the normalised-prob. distribution list
    PD_list_meas = [] # The measurement part of the normalised-prob. distribution list

    # Input states
    for state in compressed_circuit['state_list']:
        Gamma = x2Gamma(x_list[8*Gamma_index:8*(Gamma_index+1)])
        QD_state = W_state_1q(state, Gamma) # W_\rho(p_in,q_in)
        PD_state = np.abs(QD_state)         # |W_\rho(p_in,q_in)|
        neg_state = PD_state.sum()          # sum_{p_in, q_in} |W(p_in,q_in)|
        sign_state = np.sign(QD_state)      # sign(W(p_in,q_in))
        PD_state /= neg_state               # |W_\rho(p_in,q_in)|/sum_{p_in, q_in} |W(p_in,q_in)| : Normalised Probability

        QD_list_states.append(QD_state)
        PD_list_states.append(PD_state)
        neg_list_states.append(neg_state)
        sign_list_states.append(sign_state)

        Gammas.append(Gamma)
        Gamma_index += 1

    # Gates
    gate_index = 0
    for gate in compressed_circuit['gate_list']:
        idx = compressed_circuit['index_list'][gate_index]
        gate_index += 1

        Gamma_in1 = Gammas[idx[0]]
        Gamma_in2 = Gammas[idx[1]]
        Gamma_out1 = x2Gamma(x_list[8*Gamma_index:8*(Gamma_index+1)])
        Gamma_out2 = x2Gamma(x_list[8*(Gamma_index+1):8*(Gamma_index+2)])

        QD_gate = W_gate_2q(gate,Gamma_in1,Gamma_in2,Gamma_out1,Gamma_out2)
        # W_U(p1_out,q1_out,p2_out,q2_out|p1_in,q1_in,p2_in,q2_in)
        PD_gate = np.abs(QD_gate)
        neg_gate = PD_gate.sum(axis=(4,5,6,7))
        sign_gate = np.sign(QD_gate)
        temp = PD_gate.swapaxes(0,4).swapaxes(1,5).swapaxes(2,6).swapaxes(3,7)
        PD_gate = (temp/neg_gate).swapaxes(0,4).swapaxes(1,5).swapaxes(2,6).swapaxes(3,7)

        QD_list_gates.append(QD_gate)
        PD_list_gates.append(PD_gate)
        neg_list_gates.append(neg_gate)
        sign_list_gates.append(sign_gate)

        Gammas[idx[0]] = Gamma_out1
        Gammas[idx[1]] = Gamma_out2
        Gamma_index += 2

    # Measurement
    qudit_index = 0
    for meas in compressed_circuit['meas_list']:
        if str(meas)=='/': continue
        Gamma = Gammas[qudit_index]
        qudit_index += 1

        QD_meas = W_meas_1q(meas, Gamma)
        PD_meas = np.abs(QD_meas)
        neg_meas = np.max(PD_meas)
        sign_meas = np.sign(QD_meas)

        QD_list_meas.append(QD_meas)
        PD_list_meas.append(PD_meas)
        neg_list_meas.append(neg_meas)
        sign_list_meas.append(sign_meas)

    QD_output = {
    'QD_list_states': QD_list_states, 'QD_list_gates': QD_list_gates, 'QD_list_meas': QD_list_meas,
    'PD_list_states': PD_list_states, 'PD_list_gates': PD_list_gates, 'PD_list_meas': PD_list_meas,
    'neg_list_states': neg_list_states, 'neg_list_gates': neg_list_gates, 'neg_list_meas': neg_list_meas,
    'sign_list_states': sign_list_states, 'sign_list_gates': sign_list_gates, 'sign_list_meas': sign_list_meas
        }
    return QD_output

def get_prob_list_3q(compressed_circuit, x_list, **kwargs):
    options = {'option1': True}
    options.update(kwargs)
    '''
    Calculate the quasi-probability distribution of each circuit element
    and return them as a list of [state part, gate part, measurement part].
    '''

    Gamma_index = 0
    Gammas = []

    QD_list_states = [] # The state part of the quasi-prob. distribution list
    QD_list_gates = [] # The gate part of the quasi-prob. distribution list
    QD_list_meas = [] # The measurement part of the quasi-prob. distribution list

    neg_list_states = [] # The state part of the quasi-prob. negativity list
    neg_list_gates = [] # The gate part of the quasi-prob. negativity list
    neg_list_meas = [] # The gate part of the quasi-prob. negativity list

    sign_list_states = [] # The state part of the sign of quasi-prob. distribution list
    sign_list_gates = [] # The gate part of the sign of the quasi-prob. distribution list
    sign_list_meas = [] # The measurement part of the sign of the quasi-prob. distribution list

    PD_list_states = [] # The state part of the normalised-prob. distribution list
    PD_list_gates = [] # The gate part of the normalised-prob. distribution list
    PD_list_meas = [] # The measurement part of the normalised-prob. distribution list

    # Input states
    for state in compressed_circuit['state_list']:
        Gamma = x2Gamma(x_list[8*Gamma_index:8*(Gamma_index+1)])
        QD_state = W_state_1q(state, Gamma) # W_\rho(p_in,q_in)
        PD_state = np.abs(QD_state)         # |W_\rho(p_in,q_in)|
        neg_state = PD_state.sum()          # sum_{p_in, q_in} |W(p_in,q_in)|
        sign_state = np.sign(QD_state)      # sign(W(p_in,q_in))
        PD_state /= neg_state               # |W_\rho(p_in,q_in)|/sum_{p_in, q_in} |W(p_in,q_in)| : Normalised Probability

        QD_list_states.append(QD_state)
        PD_list_states.append(PD_state)
        neg_list_states.append(neg_state)
        sign_list_states.append(sign_state)

        Gammas.append(Gamma)
        Gamma_index += 1

    # Gates
    gate_index = 0
    for gate in compressed_circuit['gate_list']:
        idx = compressed_circuit['index_list'][gate_index]
        gate_index += 1

        Gamma_in1 = Gammas[idx[0]]
        Gamma_in2 = Gammas[idx[1]]
        Gamma_in3 = Gammas[idx[2]]
        Gamma_out1 = x2Gamma(x_list[8*Gamma_index:8*(Gamma_index+1)])
        Gamma_out2 = x2Gamma(x_list[8*(Gamma_index+1):8*(Gamma_index+2)])
        Gamma_out3 = x2Gamma(x_list[8*(Gamma_index+2):8*(Gamma_index+3)])

        QD_gate = W_gate_3q(gate,Gamma_in1,Gamma_in2,Gamma_in3,Gamma_out1,Gamma_out2,Gamma_out3)
        PD_gate = np.abs(QD_gate)
        neg_gate = PD_gate.sum(axis=(6,7,8,9,10,11))
        sign_gate = np.sign(QD_gate)
        temp = PD_gate.swapaxes(0,6).swapaxes(1,7).swapaxes(2,8).swapaxes(3,9).swapaxes(4,10).swapaxes(5,11)
        PD_gate = (temp/neg_gate).swapaxes(0,6).swapaxes(1,7).swapaxes(2,8).swapaxes(3,9).swapaxes(4,10).swapaxes(5,11)

        QD_list_gates.append(QD_gate)
        PD_list_gates.append(PD_gate)
        neg_list_gates.append(neg_gate)
        sign_list_gates.append(sign_gate)

        Gammas[idx[0]] = Gamma_out1
        Gammas[idx[1]] = Gamma_out2
        Gammas[idx[2]] = Gamma_out3
        Gamma_index += 3

    # Measurement
    qudit_index = 0
    for meas in compressed_circuit['meas_list']:
        if str(meas)=='/': continue
        Gamma = Gammas[qudit_index]
        qudit_index += 1

        QD_meas = W_meas_1q(meas, Gamma)
        PD_meas = np.abs(QD_meas)
        neg_meas = np.max(PD_meas)
        sign_meas = np.sign(QD_meas)

        QD_list_meas.append(QD_meas)
        PD_list_meas.append(PD_meas)
        neg_list_meas.append(neg_meas)
        sign_list_meas.append(sign_meas)

    QD_output = {
    'QD_list_states': QD_list_states, 'QD_list_gates': QD_list_gates, 'QD_list_meas': QD_list_meas,
    'PD_list_states': PD_list_states, 'PD_list_gates': PD_list_gates, 'PD_list_meas': PD_list_meas,
    'neg_list_states': neg_list_states, 'neg_list_gates': neg_list_gates, 'neg_list_meas': neg_list_meas,
    'sign_list_states': sign_list_states, 'sign_list_gates': sign_list_gates, 'sign_list_meas': sign_list_meas
        }
    return QD_output