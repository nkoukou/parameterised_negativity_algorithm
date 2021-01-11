import numpy as np
import numpy.random as nr

from circuit_components import(makeState, makeGate)
from phase_space import(x2Gamma, W_state_1q, neg_state_1q, W_gate_1q,
                        neg_gate_1q, W_gate_2q, neg_gate_2q, W_meas_1q,
                        neg_meas_1q)

def sample(circuit, x=None, niters=1000):
    ''' Samples given circuit with given parameter list x. Number of sampling
        iterations is niters.
    '''
    outcomes = np.zeros(len(circuit[0]))
    for i in range(niters):
        if i%int(niters/10)==0: print(i)
        outcome = sample_iter(circuit, x)
        outcomes += outcome
    outcomes /= niters
    return outcomes

def sample_iter(circuit, x=None):
    ''' Performs Monte Carlo sampling (Pashayan et al.)
    '''
    state_string, gate_sequence, meas_string = circuit
    if x is None:
        gates = []
        for g in gate_sequence:
            gates.append(g[1])
        sampled_components = state_string + ''.join(gates)
        x = [1,0,0,0,0,0,1,0]*len(sampled_components)

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
        # print('---------------------------')
        # print('STATE')
        # print(prob_dist)
        # print(outcomes)

    running_idx = len(state_string)
    for g in range(len(gate_sequence)):
        idx, gate = gate_sequence[g][0], makeGate(gate_sequence[g][1])
        # print('Gate: ', gate)
        # print('------------------------------------------------------------')
        # print('Gammas: ', gammas)
        # print('------------------------------------------------------------')
        # print('Out: ', outcomes)
        # print()
        if len(idx)==1:
            Gamma_in = gammas[idx[0]]
            Gamma_out = x2Gamma(x[8*running_idx:8*(running_idx+1)])

            row_p =int(outcomes[idx[0]]//3)
            row_q = int(outcomes[idx[0]]%3)

            w = (W_gate_1q(gate, Gamma_in, Gamma_out)[row_p, row_q]).flatten()
            neg = neg_gate_1q(gate, Gamma_in, Gamma_out)[row_p, row_q]

            prob_dist = np.abs(w)/neg
            outcomes[idx[0]] = nr.choice(np.arange(len(prob_dist)),
                                         p=prob_dist)
            gammas[idx[0]] = Gamma_out
            running_idx +=1
        elif len(idx)==2:
            Gamma_in1 = gammas[idx[0]]
            Gamma_in2 = gammas[idx[1]]
            Gamma_out1 = x2Gamma(x[8*running_idx:8*(running_idx+1)])
            Gamma_out2 = x2Gamma(x[8*(running_idx+1):8*(running_idx+2)])

            row1_p = int(outcomes[idx[0]]//3)
            row2_p = int(outcomes[idx[0]]%3)
            row1_q = int(outcomes[idx[1]]//3)
            row2_q = int(outcomes[idx[1]]%3)

            w = (W_gate_2q(gate, Gamma_in1, Gamma_in2, Gamma_out1, Gamma_out2
                           )[row1_p,row2_p,row1_q,row2_q]).flatten()
            neg = neg_gate_2q(gate, Gamma_in1, Gamma_in2, Gamma_out1,
                              Gamma_out2)[row1_p,row2_p,row1_q,row2_q]

            prob_dist = np.abs(w)/neg
            outcome = nr.choice(np.arange(len(prob_dist)),
                                         p=prob_dist)
            outcomes[idx[0]] = outcome//9
            outcomes[idx[1]] = outcome%9

            gammas[idx[0]], gammas[idx[1]] = Gamma_out1, Gamma_out2
            running_idx +=2
        else:
            raise Exception('Too many gate indices')
        # print('---------------------------')
        # print('GATE')
        # print(prob_dist)
        # print(outcomes)

    #measurement
    for m in range(len(meas_string)):
        if meas_string[m]=='/':
            outcomes[m] = -1
            continue
        Gamma = gammas[m]
        E = makeState(meas_string[m])

        row_p = int(outcomes[m]//3)
        row_q = int(outcomes[m]%3)
        outcomes[m] = W_meas_1q(E, Gamma)[row_p, row_q]
        # print('---------------------------')
        # print('MEAS')
        # print(W_meas_1q(E, Gamma).flatten())
        # print(outcomes)

    return outcomes

def calc_prob(circuit):
    ''' Calculates exact Born probability of given circuit.
    '''
    pass