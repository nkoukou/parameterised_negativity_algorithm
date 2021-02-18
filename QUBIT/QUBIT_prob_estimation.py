import numpy as np
import numpy.random as nr
import matplotlib.pyplot as plt
import time

from QUBIT_circuit_components import(makeState, makeGate)
from QUBIT_opt_neg import(optimize_neg)
from QUBIT_phase_space import(x2Gamma, W_state_1q, neg_state_1q, W_gate_1q,
                        neg_gate_1q, W_gate_2q, neg_gate_2q, W_meas_1q,
                        neg_meas_1q, n_Gammas)

def sample(circuit, x=0, niters=10000):
    ''' Samples given circuit with given parameter list x.
        x      - Gamma parameter list
               - if 0, Wigner distribution.
        niters - number of sampling iterations.
    '''
    p_sample = 0
    for n in range(niters):
        p_sample += sample_iter(circuit, x)
    return p_sample/niters


def compare_Wigner_para(circuit, niters=10000, save=1):
    '''Show the difference between sampling with Wigner distribution
       and sampling with the optimised parameter list 'opt_Gammas'.
       It prints the final p_estimate in each case
       and produces a plot showing the convergence of p_estimate 
       in each case as a function of niters.
    '''
    Wigner_Gammas = [1,1/2,1/2]*n_Gammas(circuit)
    opt_Gammas, Gamma_dist = optimize_neg(circuit)

    # Calculate the quasi-probability distributions of all circuit elements and save them
    PD_list_Wigner = get_prob_list(circuit, Wigner_Gammas)
    PD_list_Gammas = get_prob_list(circuit, opt_Gammas)

    p_sample_Wigner = 0
    p_sample_opt = 0

    plot_Wigner = []
    plot_opt = []

    path1 = 'Wigner.txt'
    path2 = 'Optimised.txt'

    start_time = time.time()
    for n in range(niters):
        p_sample_Wigner += sample_iter(circuit, PD_list_Wigner)
        p_sample_opt += sample_iter(circuit, PD_list_Gammas)

        plot_Wigner.append(p_sample_Wigner/(n+1))
        plot_opt.append(p_sample_opt/(n+1))

        if save==1:
            with open(path1, 'a') as f1:
                f1.write(str(n+1)+" "+str(p_sample_Wigner/(n+1)))
                f1.write("\n")
            with open(path2, 'a') as f2:
                f2.write(str(n+1)+" "+str(p_sample_opt/(n+1)))
                f2.write("\n")
    sampling_time = time.time() - start_time

    # Print the final results
    print('==================Sampling Results====================')
    print('Using Wigner: ', p_sample_Wigner/niters)
    print('Using optimised QP: ', p_sample_opt/niters)
    print('Sampling time: ', sampling_time)

    # Plot the results, both the Wigner and the optimised ones.
    x_axis = np.arange(niters)
    plt.plot(x_axis, plot_Wigner, linestyle='solid', color='tab:blue', label='Wigner')
    plt.plot(x_axis, plot_opt, linestyle='solid', color='tab:orange', label='Optimised')
    plt.xlabel("number of iterations")
    plt.ylabel("p_estimate")
    plt.legend()
    plt.savefig('BValg_niter.png')

    plt.plot(x_axis, plot_Wigner, linestyle='solid', color='tab:blue', label='Wigner')
    plt.plot(x_axis, plot_opt, linestyle='solid', color='tab:orange', label='Optimised')
    plt.xlabel("number of iterations")
    plt.ylabel("p_estimate")
    plt.legend()
    plt.ylim([-100,100])
    plt.savefig('BValg_niter_zoom1.png')

    plt.plot(x_axis, plot_Wigner, linestyle='solid', color='tab:blue', label='Wigner')
    plt.plot(x_axis, plot_opt, linestyle='solid', color='tab:orange', label='Optimised')
    plt.xlabel("number of iterations")
    plt.ylabel("p_estimate")
    plt.legend()
    plt.ylim([-10,10])
    plt.savefig('BValg_niter_zoom2.png')

    plt.plot(x_axis, plot_Wigner, linestyle='solid', color='tab:blue', label='Wigner')
    plt.plot(x_axis, plot_opt, linestyle='solid', color='tab:orange', label='Optimised')
    plt.xlabel("number of iterations")
    plt.ylabel("p_estimate")
    plt.legend()
    plt.ylim([-0.5,2.5])
    plt.savefig('BValg_niter_zoom3.png')

    f1.close()
    f2.close()


def sample_iter(circuit, PD_list):
    ''' Performs Monte Carlo sampling as outlined in Pashayan et al. (2015)
        for given circuit and the list of Gamma quasi-probability distributions
        of circuit elements.
    '''
    state_string, gate_sequence, meas_string = circuit

    # The list of Gamma quasi-probability distributions for circuit elements
    PD_list_states, PD_list_gates, PD_list_meas = PD_list

    current_PS_point = []
    p_estimate = 1.
   
    # Input states
    for s in range(len(state_string)):
        state = makeState(state_string[s])
        WF = PD_list_states[s].flatten()
        neg = np.abs(PD_list_states[s]).sum()
        prob = np.abs(WF)/neg

        PS_point = nr.choice(np.arange(len(prob)), p=prob)

        current_PS_point.append(PS_point)
        p_estimate *= neg*np.sign(WF[PS_point])

    # Gates
    for g in range(len(gate_sequence)):
        idx, gate = gate_sequence[g][0], makeGate(gate_sequence[g][1])
        if len(idx)==1:
            p_in = current_PS_point[idx[0]]//2
            q_in = current_PS_point[idx[0]]%2
            WF = PD_list_gates[g][p_in,q_in].flatten()
            neg = np.abs(PD_list_gates[g]).sum(axis=(2,3))[p_in,q_in]

            prob = np.abs(WF)/neg

            PS_point = nr.choice(np.arange(len(prob)), p=prob)
            current_PS_point[idx[0]] = PS_point            
            p_estimate *= neg*np.sign(WF[PS_point])
        elif len(idx)==2:
            p_in1 = current_PS_point[idx[0]]//2
            q_in1 = current_PS_point[idx[0]]%2
            p_in2 = current_PS_point[idx[1]]//2
            q_in2 = current_PS_point[idx[1]]%2
            WF = PD_list_gates[g][p_in1,q_in1,p_in2,q_in2].flatten()
            neg = np.abs(PD_list_gates[g]).sum(axis=(4,5,6,7))[p_in1,q_in1,p_in2,q_in2]
            prob = np.abs(WF)/neg

            PS_point = nr.choice(np.arange(len(prob)), p=prob)

            current_PS_point[idx[0]] = PS_point//4
            current_PS_point[idx[1]] = PS_point%4
            p_estimate *= neg*np.sign(WF[PS_point])
        else:
            raise Exception('Too many gate indices')

    # Measurement
    for m in range(len(meas_string)):
        if meas_string[m]!='/':
            E = makeState(meas_string[m])
            WF = PD_list_meas[m].flatten()

            p_estimate *= WF[current_PS_point[m]]

    return p_estimate

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
        Gamma = x2Gamma(x[3*Gamma_index:3*(Gamma_index+1)])
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
                Gamma_out = x2Gamma(x[3*Gamma_index:3*(Gamma_index+1)])
                Gamma_index += 1
            WF = W_gate_1q(gate, Gamma_in, Gamma_out)

            PD_list_gates.append(WF)
            Gammas[idx[0]] = Gamma_out    
        elif len(idx)==2:
            Gamma_in1 = Gammas[idx[0]]
            Gamma_in2 = Gammas[idx[1]]
            Gamma_out1 = x2Gamma(x[3*Gamma_index:3*(Gamma_index+1)])
            Gamma_out2 = x2Gamma(x[3*(Gamma_index+1):3*(Gamma_index+2)])
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
