import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt
import time
import itertools as it

from state_functions import DIM
from phase_space import(x2Gamma, W_state_1q, neg_state_1q,
                        W_gate_1q, neg_gate_1q, W_gate_2q, neg_gate_2q, W_gate_3q, neg_gate_3q,
                        W_meas_1q, neg_meas_1q)

def get_prob_list(circuit, par_list, par0, W, **kwargs):
    options = {'option1': True}
    options.update(kwargs)
    '''
    Calculate the quasi-probability distribution of each circuit element
    and return them as a list of [state part, gate part, measurement part].
    '''
    W_state = W[0]
    W_gate = W[1]
    W_meas = W[2]

    par0_len = len(par0)

    frame_index = 0
    frames = []

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
    for state in circuit['state_list']:
        frame = par_list[par0_len*frame_index:par0_len*(frame_index+1)]
        QD_state = W_state(state, frame)    # W_\rho(p_in,q_in)
        PD_state = np.abs(QD_state)         # |W_\rho(p_in,q_in)|
        neg_state = PD_state.sum()          # sum_{p_in, q_in} |W(p_in,q_in)|
        sign_state = np.sign(QD_state)      # sign(W(p_in,q_in))
        PD_state /= neg_state               # |W_\rho(p_in,q_in)|/sum_{p_in, q_in} |W(p_in,q_in)| : Normalised Probability

        QD_list_states.append(QD_state)
        PD_list_states.append(PD_state)
        neg_list_states.append(neg_state)
        sign_list_states.append(sign_state)

        frames.append(frame)
        frame_index += 1

    # Gates
    gate_index = 0
    for gate in circuit['gate_list']:
        idx = circuit['index_list'][gate_index]
        gate_index += 1

        frames_in = [frames[i] for i in idx]
        frames_out = []
        for i in range(len(idx)):
            frames_out.append(par_list[par0_len*(frame_index+i):par0_len*(frame_index+i+1)])
        frame_index += len(idx)

        QD_gate = W_gate(gate, frames_in, frames_out)
        # W_U(p1_out,q1_out,p2_out,q2_out|p1_in,q1_in,p2_in,q2_in)
        PD_gate = np.abs(QD_gate)
        if len(idx)==1:
            neg_gate = PD_gate.sum(axis=(2,3))
            sign_gate = np.sign(QD_gate)
            temp = PD_gate.swapaxes(0,2).swapaxes(1,3)
            PD_gate = (temp/neg_gate).swapaxes(0,2).swapaxes(1,3)
        elif len(idx)==2:
            neg_gate = PD_gate.sum(axis=(4,5,6,7))
            sign_gate = np.sign(QD_gate)
            temp = PD_gate.swapaxes(0,4).swapaxes(1,5).swapaxes(2,6).swapaxes(3,7)
            PD_gate = (temp/neg_gate).swapaxes(0,4).swapaxes(1,5).swapaxes(2,6).swapaxes(3,7)
        elif len(idx)==3:
            neg_gate = PD_gate.sum(axis=(6,7,8,9,10,11))
            sign_gate = np.sign(QD_gate)
            temp = PD_gate.swapaxes(0,6).swapaxes(1,7).swapaxes(2,8).swapaxes(3,9).swapaxes(4,10).swapaxes(5,11)
            PD_gate = (temp/neg_gate).swapaxes(0,6).swapaxes(1,7).swapaxes(2,8).swapaxes(3,9).swapaxes(4,10).swapaxes(5,11)
        else:
            raise Exception('Cannot handle a gate larger than 3-qudit gates.')

        QD_list_gates.append(QD_gate)
        PD_list_gates.append(PD_gate)
        neg_list_gates.append(neg_gate)
        sign_list_gates.append(sign_gate)

        for i, index in enumerate(idx):
            frames[index] = frames_out[i]

    # Measurement
    qudit_index = 0
    for meas in circuit['meas_list']:
        if str(meas)=='/': continue
        frame = frames[qudit_index]
        qudit_index += 1

        QD_meas = W_meas(meas, frame)
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