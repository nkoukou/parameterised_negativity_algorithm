import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt
import time
import itertools as it

from QUBIT_state_functions import DIM
from QUBIT_phase_space import(x2Gamma, W_state_1q, neg_state_1q, W_gate_1q, neg_gate_1q, W_gate_2q, neg_gate_2q, W_meas_1q, neg_meas_1q)

def get_prob_list(compressed_circuit,x_list,**kwargs):
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
        Gamma = x2Gamma(x_list[3*Gamma_index:3*(Gamma_index+1)])
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
        idx = compressed_circuit['qudit_index_list'][gate_index]
        gate_index += 1
        
        Gamma_in1 = Gammas[idx[0]]
        Gamma_in2 = Gammas[idx[1]]
        Gamma_out1 = x2Gamma(x_list[3*Gamma_index:3*(Gamma_index+1)])
        Gamma_out2 = x2Gamma(x_list[3*(Gamma_index+1):3*(Gamma_index+2)])
        
        QD_gate = W_gate_2q(gate,Gamma_in1,Gamma_in2,Gamma_out1,Gamma_out2) # W_U(p1_out,q1_out,p2_out,q2_out|p1_in,q1_in,p2_in,q2_in)
        PD_gate = np.abs(QD_gate)
        neg_gate = PD_gate.sum(axis=(4,5,6,7))
        sign_gate = np.sign(QD_gate)
        PD_gate = (PD_gate.swapaxes(0,4).swapaxes(1,5).swapaxes(2,6).swapaxes(3,7)/neg_gate).swapaxes(0,4).swapaxes(1,5).swapaxes(2,6).swapaxes(3,7)
        
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
