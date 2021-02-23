import numpy as np
import numpy.random as nr

from circuit_components import(makeState, makeGate)
from phase_space import(x2Gamma, W_state_1q, neg_state_1q, W_gate_1q,
                        neg_gate_1q, W_gate_2q, neg_gate_2q, W_meas_1q,
                        neg_meas_1q)

PD_list = []

def sample(circuit, x=None, niters=1000):
    ''' Samples given circuit with given parameter list x.
        x      - parameter list (x=0: Wigner, x=1: randomly perturbed
                                 distribution near Wigner)
        niters - number of sampling iterations.
    '''
    prob = 0
    prob_list = []

    if isinstance(x, int):
        x_length = n_Gammas(circuit)
        x=[1,0,0,0,0,0,1,0]*x_length
        if x==1: # Randomly perturbed Wigner distribution
            x += 0.01* 2*np.random.rand(8*x_length)-1

    global PD_list
    PD_list = get_prob_list(circuit, x)

    for i in range(niters):
#        if i%int(niters/10)==0: print(i)
        prob += sample_iter(circuit)
        prob_list.append(prob)
    prob /= niters
    return prob, np.array(prob_list)/np.linspace(1,niters,niters)

def sample_iter(circuit):
    ''' Performs Monte Carlo sampling as outlined in Pashayan et al. (2015)
        for given circuit and parameter list x.
        (x=0 uses Wigner, x=1 uses randomly perturbed distribution near Wigner)
    '''
    state_string, gate_sequence, meas_string = circuit

    # The list of Gamma quasi-probability distributions for circuit elements
    PD_list_states, PD_list_gates, PD_list_meas = PD_list

    # Sampling
    # Initial state
    outcomes = np.zeros(len(state_string))
    gammas = np.zeros((len(state_string),3,3), dtype='complex_')
    prob = 1

    for s in range(len(state_string)):
        state = makeState(state_string[s])

        w = PD_list_states[s].flatten()
        neg = np.abs(PD_list_states[s]).sum()

        prob_dist = np.abs(w)/neg
        outcome = nr.choice(np.arange(len(prob_dist)), p=prob_dist)

        outcomes[s] = outcome
        prob *= neg*np.sign(w[outcome])

    # Gates
    running_idx = len(state_string)
    for g in range(len(gate_sequence)):
        idx, gate = gate_sequence[g][0], makeGate(gate_sequence[g][1])
        if len(idx)==1:
            row_p = int(outcomes[idx[0]]//3)
            row_q = int(outcomes[idx[0]]%3)

            w = (PD_list_gates[g][row_p, row_q]).flatten()
            neg = (np.abs(PD_list_gates[g]).sum(axis=(2,3)))[row_p, row_q]

            prob_dist = np.abs(w)/neg
            outcome = nr.choice(np.arange(len(prob_dist)), p=prob_dist)

            outcomes[idx[0]] = outcome
            prob *= neg*np.sign(w[outcome])
            #running_idx +=1
        elif len(idx)==2:
            row1_p = int(outcomes[idx[0]]//3)
            row1_q = int(outcomes[idx[0]]%3)
            row2_p = int(outcomes[idx[1]]//3)
            row2_q = int(outcomes[idx[1]]%3)

            w = (PD_list_gates[g][row1_p,row1_q,row2_p,row2_q]).flatten()
            neg = (np.abs(PD_list_gates[g]).sum(axis=(4,5,6,7)))[row1_p,row1_q,row2_p,row2_q]

            prob_dist = np.abs(w)/neg
            outcome = nr.choice(np.arange(len(prob_dist)), p=prob_dist)
            outcomes[idx[0]] = outcome//9
            outcomes[idx[1]] = outcome%9

            prob *= neg*np.sign(w[outcome])
            #running_idx +=2
        else:
            raise Exception('Too many gate indices')

    # Measurement
    for m in range(len(meas_string)):
        if meas_string[m]=='/':
            outcomes[m] = -1
            continue
        E = makeState(meas_string[m])

        row_p = int(outcomes[m]//3)
        row_q = int(outcomes[m]%3)
        w = PD_list_meas[m][row_p, row_q]
        prob *= w
    return prob

def get_prob_list(circuit, x):
    '''
    For a given circuit and a list of Gammas, 
    calculate the Gamma probability distributions of each circuit element.
    '''
    state_string, gate_sequence, meas_string = circuit

    Gamma_index = 0
    Gammas = []

    PD_list_states = []
    PD_list_gates = []
    PD_list_meas = []
    
    # Input states
    for s in state_string:
        state = makeState(s)
        Gamma = x2Gamma(x[8*Gamma_index:8*(Gamma_index+1)])
        WF = W_state_1q(state, Gamma)
        
        PD_list_states.append(WF)
        Gammas.append(Gamma)
        Gamma_index += 1
    # Gates
    for g in gate_sequence:
        idx, gate = g[0], makeGate(g[1])
        if len(idx)==1:
            if g[1]=='H':
                Gamma_in = Gammas[idx[0]]
                Gamma_out = np.dot(np.dot(gate, Gamma_in),np.conjugate(gate.T))
            else:    
                Gamma_in = Gammas[idx[0]]
                Gamma_out = x2Gamma(x[8*Gamma_index:8*(Gamma_index+1)])
                Gamma_index += 1
            WF = W_gate_1q(gate, Gamma_in, Gamma_out)

            PD_list_gates.append(WF)
            Gammas[idx[0]] = Gamma_out    
        elif len(idx)==2:
            Gamma_in1 = Gammas[idx[0]]
            Gamma_in2 = Gammas[idx[1]]
            Gamma_out1 = x2Gamma(x[8*Gamma_index:8*(Gamma_index+1)])
            Gamma_out2 = x2Gamma(x[8*(Gamma_index+1):8*(Gamma_index+2)])
            WF = W_gate_2q(gate,Gamma_in1,Gamma_in2,Gamma_out1,Gamma_out2)

            PD_list_gates.append(WF)
            Gammas[idx[0]] = Gamma_out1
            Gammas[idx[1]] = Gamma_out2
            Gamma_index += 2
        else:
            raise Exception('Too many gate indices')
    # Measurement
    for m in range(len(meas_string)):
        if meas_string[m]!='/':
            E = makeState(meas_string[m])
            Gamma = Gammas[m]
            WF = W_meas_1q(E, Gamma)

            PD_list_meas.append(WF)
        else:
            PD_list_meas.append([])

    return [PD_list_states, PD_list_gates, PD_list_meas]

def n_Gammas(circuit):
    state_string, gate_sequence, meas_string = circuit
    Gamma_index = 0
    for s in state_string:
        Gamma_index += 1
    for g in gate_sequence:
        idx = g[0]
        if len(idx)==1:
            if g[1]=='H':
                continue
            Gamma_index += 1
        elif len(idx)==2:
            Gamma_index += 2
    return Gamma_index

def calc_prob(circuit):
    ''' Calculates exact Born probability of given circuit.
    '''
    pass
