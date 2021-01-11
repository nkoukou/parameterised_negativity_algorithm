import numpy as np
import numpy.random as nr

from circuit_components import(makeState, makeGate)
from phase_space import(x2Gamma, W_state_1q, neg_state_1q, W_gate_1q,
                        neg_gate_1q, W_gate_2q, neg_gate_2q, W_meas_1q,
                        neg_meas_1q)

def sample(circuit, x=None):
    ''' Performs Monte Carlo sampling (Pashayan et al.)
    '''
    state_string, gate_sequence, meas_string = circuit
    if x is None:
        gates = []
        for g in gate_sequence:
            gates.append(g[1])
        components = state_string + ''.join(gates) + meas_string
        x = [1,0,0,0,0,0,1,0]*len(components)
    # Sampling
    outcomes = np.zeros(len(state_string))
    gammas = np.zeros((len(state_string),3,3), dtype='complex_')

    for s in range(len(state_string)):
        Gamma = x2Gamma(x[8*s:8*(s+1)])
        state = makeState(state_string[s])
        prob_dist = np.abs(W_state_1q(state, Gamma).flatten()
                           )/neg_state_1q(state, Gamma)
        outcomes[s] = nr.choice(np.arange(len(prob_dist)), p=prob_dist)
        gammas[s] = Gamma

    running_idx = s
    for g in range(len(gate_sequence)):
        idx, gate = gate_sequence[g][0], gate_sequence[g][1]
        # print('Gate: ', gate)
        # print('------------------------------------------------------------')
        # print('Gammas: ', gammas)
        # print('------------------------------------------------------------')
        # print('Out: ', outcomes)
        # print()
        gate = makeGate(gate)
        if len(idx)==1:
            Gamma_in = gammas[idx[0]]
            Gamma_out = x2Gamma(x[8*running_idx:8*(running_idx+1)])

            row_p =int(outcomes[idx[0]]//3)
            row_q = int(outcomes[idx[0]]%3)

            w = W_gate_1q(gate, Gamma_in, Gamma_out)[row_p, row_q].flatten()
            neg = neg_gate_1q(gate, Gamma_in, Gamma_out)[row_p, row_q]

            prob_dist = np.abs(w)/neg
            outcomes[s] = nr.choice(np.arange(len(prob_dist)), p=prob_dist)
            gammas[idx[0]] = Gamma_out
            running_idx +=1
        elif len(idx)==2:
            running_idx +=2
        else:
            raise Exception('Too many gate indices')

    #measurement
    for m in range(len(meas_string)):
        if meas_string[m]=='1':
            outcomes[m] = -1
            continue
        Gamma = gammas[m]
        E = makeState(meas_string[m])

        row_p =int(outcomes[m]//3)
        row_q = int(outcomes[m]%3)
        outcomes[m] = W_meas_1q(E, Gamma)[row_p, row_q]

    return outcomes


    # init_state = circuit[0]
    # qudit_num = len(init_state)

    # circ_repr = [[init_state[i], ''] for i in range(qudit_num)]

    # gate_tracker = {}
    # for i in range(qudit_num):
    #     gate_tracker[i] = 0
    # for gate in circuit[1:-1][0]:
    #     idx, g = gate[0], gate[1]
    #     if len(idx)==1:
    #         circ_repr[idx[0]].append(g)
    #         gate_tracker[idx[0]] +=1
    #     elif len(idx)==2:
    #         idx_max = max(gate_tracker[idx[0]], gate_tracker[idx[1]])
    #         for i in [0,1]:
    #             circ_repr[idx[i]].extend(['1' for j in
    #               range(idx_max - gate_tracker[idx[i]])])
    #             gate_tracker[idx[i]] += idx_max - gate_tracker[idx[i]]
    #             circ_repr[idx[i]].append(g[i])
    #             gate_tracker[idx[i]] +=1
    #     else: raise Exception('Too many gate indices')
    # idx_max = max(gate_tracker.values())
    # for i in range(qudit_num):
    #     circ_repr[i].extend(['1' for j in
    #                               range(idx_max - gate_tracker[i])])

    # for i in range(qudit_num):
    #     circ_repr[i].append(circuit[-1][i])


    # circ_repr = [''.join(wire) for wire in circ_repr]
    # layers = []
    # for idx in range(len(circ_repr[0])):
    #     layer = []
    #     for wire in circ_repr:
    #         layer.append(wire[idx])
    #     layers.append(''.join(layer))

    # for wire in circ_repr:
    #     print(wire)
    # return circ_repr, layers
    # state_string, gate_sequence, meas_string = circuit

    # current_state_index = []
    # init_state_index = []
    # x_index = 0
    # for state_str in state_string:
    #     init_state_index.append([x_index, makeState(state_str)])
    #     current_state_index.append(x_index)
    #     x_index += 1

    # gate_1q_index = []
    # gate_2q_index = []
    # for gate_str in gate_sequence:
    #     if len(gate_str[0])==1:
    #         t_index = (gate_str[0])[0]
    #         gate_1q_index.append([[current_state_index[t_index], x_index],
    #                                makeGate(gate_str[1])])
    #         current_state_index[t_index] = x_index
    #         x_index += 1
    #     elif len(gate_str[0])==2:
    #         c_index = (gate_str[0])[0]
    #         t_index = (gate_str[0])[1]
            # gate_2q_index.append( [[current_state_index[c_index],
            #                         current_state_index[t_index],
            #                         x_index,x_index+1],
            #                        makeGate(gate_str[1])])
    #         current_state_index[c_index] = x_index
    #         current_state_index[t_index] = x_index+1
    #         x_index += 2

    # meas_index = []
    # for meas_str in meas_string:
    #     if meas_str != '1':
    #         meas_index.append([x_index, makeState(meas_str)])
    #         x_index += 1

    # x_len = x_index

    # return(x_len, current_state_index, init_state_index,
    #        gate_1q_index, gate_2q_index, meas_index)

def calc_prob(circuit):
    ''' Calculates exact Born probability of given circuit.
    '''
    pass