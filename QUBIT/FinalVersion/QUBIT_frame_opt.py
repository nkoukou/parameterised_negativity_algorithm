import autograd.numpy as np
from autograd import(grad)

from scipy.optimize import(basinhopping)
from QUBIT_phase_space import(x2Gamma, neg_gate_1q_max, neg_gate_2q_max, neg_gate_3q_max,
                             neg_state_1q, neg_meas_1q)

x0 = [1,1/2,1/2]
Gamma0 = x2Gamma(x0)

def neg_gate_max(gate, Gamma_in, Gamma_out, n):
    '''
    Calculate the negativity of any n-qubit gate (TO DO: MAKE IT WORK FOR n>3)
    INPUT - gate: gate matrix
            Gamma_in: a list of input Gammas
            Gamma_out: a list of output Gammas
            n: the number of qubits to which the gate applies
    '''
    if n==1:
        return neg_gate_1q_max(gate, Gamma_in[0], Gamma_out[0])
    elif n==2:
        return neg_gate_2q_max(gate, Gamma_in[0], Gamma_in[1], Gamma_out[0], Gamma_out[1])
    elif n==3:
        return neg_gate_3q_max(gate, Gamma_in[0], Gamma_in[1], Gamma_in[2], Gamma_out[0], Gamma_out[1], Gamma_out[2])
    else:
        raise Exception('m>3 does not work at the moment.')

def make_x_list(full_x_list, small_x_list, x_index_list):
    '''
    Replace appropriate x-parameters in 'full_x_list' with the ones in 'small_x_list'.
    'x_index_list' contains the indices of x-parameters which are to be replaced. 
    INPUT - full_x_list: a longer vector of x-parameters (ex. a vector of x-parameters for a full circuit)
            small_x_list: a shorter vector of x-parameters (ex. a vector of optimised x-parameters for a sub-block of the full circuit)
            x_index_list: a list of indices for the x-parameters in 'small_x_list'
    '''
    full_x_list_out = full_x_list.copy()
    for i in range(len(x_index_list)):
        for j in range(len(x0)):
            full_x_list_out[len(x0)*x_index_list[i]+j] = small_x_list[len(x0)*i+j]
    return full_x_list_out

def arrange_gates(gates_set, indices_set, gate_numbers):
    '''
    Sort 'gates_set', 'indices_set', and 'gate_numbers' in ascending order of gate numbers.
    '''
    sorted_gates_set = []
    sorted_indices_set = []
    sorted_gate_numbers = gate_numbers.copy()
    sorted_gate_numbers.sort()
    
    for i in sorted_gate_numbers:
        sorted_gates_set.append(gates_set[gate_numbers.index(i)])
        sorted_indices_set.append(indices_set[gate_numbers.index(i)])
    return sorted_gates_set, sorted_indices_set, sorted_gate_numbers

def neg_circuit(circuit, x_list=0):
    '''
    Calculate the negativity of 'circuit' with the given x-parameters ('x_list').
    If 'x_list'=0, then generate Wigner x-vector, and calculate the Wigner negativity.
    '''
    if type(x_list)==int:
        x_len = len(circuit['state_list'])
        for indices in circuit['index_list']:
            x_len += len(indices)
        x_list = x0*x_len

    neg = 1.
    Gamma_index = 0
    Gammas = []
    for state in circuit['state_list']: ### State part
        Gamma = x2Gamma(x_list[3*Gamma_index:3*(Gamma_index+1)])
        neg = neg * neg_state_1q(state, Gamma)
        Gammas.append(Gamma)
        Gamma_index += 1

    gate_index = 0
    for gate in circuit['gate_list']: ### Gate part
        idx = circuit['index_list'][gate_index]
        gate_index += 1

        Gamma_in = [Gammas[i] for i in idx]
        Gamma_out = []
        for i in range(len(idx)):
            Gamma_out.append(x2Gamma(x_list[len(x0)*(Gamma_index+i):len(x0)*(Gamma_index+i+1)]))
        Gamma_index += len(idx)
        neg = neg * neg_gate_max(gate, Gamma_in, Gamma_out, len(idx))
        
        for i, index in enumerate(idx):
            Gammas[index] = Gamma_out[i]

    qudit_index = 0
    for meas in circuit['meas_list']: ### Measurement part
        if str(meas) == '/': continue
        Gamma = Gammas[qudit_index]
        qudit_index += 1
        neg = neg * neg_meas_1q(meas, Gamma)

    return np.log(neg)

def block_frame_opt(FO_gate_list, FO_index_list, FO_gate_numbers, n, niter=3):
    '''
    Carry out the frame optimisation for a given sub-block
    INPUT - FO_gate_list: the list of the gates in the sub-block
            FO_index_list: the list of qubit indices of the gates in the sub-block
            FO_gate_numbers: the gate-indices (in the full circuit) of the gates in the sub-block 
    '''
    merged = []
    for index in FO_index_list:
        merged.append(index)
    merged = list(np.unique(merged)) # Find out the number of qubits the sub-block applies to
    
    FO_x_len = len(merged) + sum([len(indices) for indices in FO_index_list]) # the number of different x-parameters in the given sub-block
    FO_x_list = x0*FO_x_len # initial vector of x-parameters
    
    ## Find out which x-paramemters are intermediate x-parameters (to be optimised).
    FO_x_opt_list = [] # The list only containing intermediate x-parameters
    FO_x_opt_index_list = [] # The list containing the indices of intermediate x-parameters
    for i, indices in enumerate(FO_index_list):
        for j, index in enumerate(indices):
            if index in np.unique(FO_index_list[i+1:]): # If there is another 'index' in the leftover gates, then it is a intermediate x-parameter.
                FO_x_opt_list = FO_x_opt_list+x0
                which_x = len(merged) + sum([len(a) for a in FO_index_list[:i]]) + j
                FO_x_opt_index_list.append(which_x)
                
    # Later we need to know the indices of optimised x-parameter in the full circuit.
    # We record [gate_index, i] for each intermediate x-parameter and return it with the optimised x-parameters.
    # 'gate_index' is the gate-index of the gate just before the correponding frame. 
    # 'i' indicates the index of corresponding wire (for the gate with 'gate_index') of the intermediate frame (x-parameter).
    x_index_list_to_return = []
    for i in FO_x_opt_index_list:
        find_corr_gate_n = len(merged)
        for j, gate_indices in enumerate(FO_index_list):
            find_corr_gate_n += len(gate_indices)
            if i < find_corr_gate_n:
                x_index_list_to_return.append([FO_gate_numbers[j], i-find_corr_gate_n+len(gate_indices)])
                break
    
    def cost_function(x): # The negativity of the sub-block for given vector 'x' of intermediate x-parameters.
        neg = 1.
        full_x_list = make_x_list(FO_x_list, x, FO_x_opt_index_list)
        # x only contains the intermediate x-parameters. We need to build 'full_x_list' for the whole sub-block from 'x'.

        Gammas = {}
        for i, index in enumerate(merged):
            Gammas[str(index)] = x2Gamma(full_x_list[len(x0)*i:len(x0)*(i+1)])

        gate_index = 0
        for gate in FO_gate_list:
            Gamma_in = [Gammas[str(y)] for y in FO_index_list[gate_index]]
            x_index = len(merged) + sum([len(a) for a in FO_index_list[:gate_index]])
            Gamma_out = []
            for i in range(len(FO_index_list[gate_index])):
                Gamma_out.append(x2Gamma(full_x_list[len(x0)*(x_index+i):len(x0)*(x_index+i+1)]))

            neg = neg*neg_gate_max(gate, Gamma_in, Gamma_out, n)

            for i,index in enumerate(FO_index_list[gate_index]):
                Gammas[str(index)] = Gamma_out[i]
            gate_index += 1
        return np.log(neg)
    
    grad_cost_function = grad(cost_function)
    def func(x):
        return cost_function(x), grad_cost_function(x)

    ## Optimise
    optimise_result = basinhopping(func, FO_x_opt_list, minimizer_kwargs={"method":"L-BFGS-B", "jac":True}, niter=niter)
    
    return optimise_result.x, x_index_list_to_return

def frame_opt(circuit, n, l, **kwargs):
    '''
    Perform frame optimisation with spatial parameter 'n' and temporal parameter 'l'.
    Divide the circuit into sub-blocks with at most 'l' gates and optimise the intermediate frames in each sub-block.
    Return the full list of x-parameters for the circuit, which includes the optimised intermidate frames.
    INPUT - circuit: given circuit
            n: spatial parameter (the minimum number of qubits to which a compressed gate applies)
            l: temporal parameter (the maximum number of gates included in the frame optimisation)
    '''
    options = {'niter': 3}
    options.update(kwargs)

    gate_list = circuit['gate_list']
    index_list = circuit['index_list']
    n_states = len(circuit['state_list'])
    
    ## Find out the full length of the vector of x-parameters for the circuit
    x_len = n_states
    for indices in index_list:
        x_len += len(indices)

    ## Initial vector or x-parameters
    x_list = x0*x_len
    
    ## Find a sub-block of at most 'l' connected gates.
    masks = [0]*len(index_list)
    for gate_n, indices in enumerate(index_list):
        if masks[gate_n]==1: continue
        masks[gate_n] = 1
        target_indices = indices
        gate_numbers = [gate_n]
        indices_set = [indices]
        gates_set = [gate_list[gate_n]]

        ## Starting from the next gate, find other 'l'-1 gates connected with the gate with gate-index 'gate_n'
        for i, next_indices in enumerate(index_list[gate_n+1:]):
            next_gate_n = gate_n+1+i
            if masks[next_gate_n]==1: continue

            merged = list(np.unique(target_indices+next_indices))
            if len(merged)==len(target_indices)+len(next_indices): continue # Not connected
            else: # If connected, put into the sub-block
                masks[next_gate_n] = 1
                target_indices = merged
                gate_numbers.append(next_gate_n)
                indices_set.append(next_indices)
                gates_set.append(gate_list[next_gate_n])

                # Then check whether there was a connected gate with the newly added gate before
                for j, double_check in enumerate(index_list[:next_gate_n]):
                    if masks[j]==1: continue
                    check_merged = list(np.unique(double_check+next_indices))
                    if len(check_merged)!=len(double_check)+len(next_indices): # There was a gate connected to the newly added gate.
                        masks[j] = 1
                        if len(gate_numbers)<l: # Add it to the sub-block only when the sub-block is not full (contains less than 'l' gates).
                            target_indices = list(np.unique(target_indices+double_check))
                            gate_numbers.append(j)
                            indices_set.append(double_check)
                            gates_set.append(gate_list[j])

            if len(gate_numbers)>=l: break # If we found all 'l' gates, then proceed to the next step.
                
        if len(gate_numbers)==1: continue # If we couldn't find any connected gate left, then just end the loop.

        gates_set, indices_set, gate_numbers = arrange_gates(gates_set, indices_set, gate_numbers) # Sort in order
        block_x_list, x_indicates_list = block_frame_opt(gates_set, indices_set, gate_numbers, n, niter=options['niter']) # Frame optimisation

        x_indices_list = [] # From 'x_indicates_list', find the indices of the optimised x-parameters in the full circuit.
        for gate_n, i in x_indicates_list:
            x_index = n_states + sum([len(indices) for indices in index_list[:gate_n]]) + i
            x_indices_list.append(x_index)

        x_list = make_x_list(x_list, block_x_list, x_indices_list) # Replace the corresponding x-parameters with the optimised ones.
    
    neg_circuit_opt = neg_circuit(circuit, x_list)
    print('--------------------- FRAME OPTIMISATION with l =',l,'----------------------')
    print('Wigner Log Neg:', neg_circuit(circuit, x0*x_len))
    print('Optimised Log Neg:', neg_circuit_opt)
    print('----------------------------------------------------------------------------')
        
    return x_list, neg_circuit_opt
