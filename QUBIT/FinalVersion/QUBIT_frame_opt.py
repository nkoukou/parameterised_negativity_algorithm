import autograd.numpy as np
from autograd import(grad)

from scipy.optimize import(basinhopping)


def neg_gate_max(W_gate, gate, par_list_in, par_list_out):
    n = len(par_list_in)
    return np.abs(W_gate(gate, par_list_in, par_list_out)
                      ).sum(axis=tuple(np.arange(2*n,4*n))).max()

def make_par_list(full_par_list, small_par_list, par_index_list, par0):
    '''
    Replace appropriate frame-parameters in 'full_par_list' with the ones in
    'small_par_list'.
    'par_index_list' contains the indices of frame-parameters which are to be
    replaced.
    INPUT - full_par_list: a longer vector of frame-parameters
              (ex. a vector of frame-parameters for a full circuit)
            small_par_list: a shorter vector of frame-parameters
              (ex. a vector of optimised frame-parameters for a circuit block)
            par_index_list: a list of indices for the frame-parameters in
                            'small_par_list'
    '''
    par0_len = len(par0)
    full_par_list_out = full_par_list.copy()
    for i in range(len(par_index_list)):
        for j in range(par0_len):
            full_par_list_out[par0_len*par_index_list[i]+j] = \
                small_par_list[par0_len*i+j]
    return full_par_list_out

def neg_circuit(circuit, W, par_list, par0):
    '''
    Calculate the negativity of 'circuit' with the given frame-parameters
    ('par_list').
    '''
    W_state = W[0]
    W_gate = W[1]
    W_meas = W[2]
    par0_len = len(par0)

    neg = 1.
    frame_index = 0
    frames = []
    for state in circuit['state_list']: ### State part
        frame = par_list[par0_len*frame_index:par0_len*(frame_index+1)]
        neg = neg * np.abs(W_state(state, frame)).sum()
        frames.append(frame)
        frame_index += 1

    gate_index = 0
    for gate in circuit['gate_list']: ### Gate part
        idx = circuit['index_list'][gate_index]
        gate_index += 1

        frames_in = [frames[i] for i in idx]
        frames_out = []
        for i in range(len(idx)):
            frames_out.append(par_list[par0_len*(frame_index+i):par0_len*(
                frame_index+i+1)])
        frame_index += len(idx)
        neg = neg * neg_gate_max(W_gate, gate, frames_in, frames_out)

        for i, index in enumerate(idx):
            frames[index] = frames_out[i]

    qudit_index = 0
    for meas in circuit['meas_list']: ### Measurement part
        if str(meas) == '/': continue
        frame = frames[qudit_index]
        qudit_index += 1
        neg = neg * np.abs(W_meas(meas, frame)).max()

    return np.log(neg)

def block_frame_opt(FO_gate_list, FO_index_list, FO_gate_numbers, par0,
                    W_gate, niter=3):
    '''
    Carry out the frame optimisation for a given block
    INPUT - FO_gate_list:    the list of the gates in the sub-block
            FO_index_list:   the list of qubit indices of the gates in the
                             block
            FO_gate_numbers: the gate-indices (in the full circuit) of the
                             gates in the block
    '''
    par0_len = len(par0)

    # Find out the number of qudits the block applies to
    merged = []
    for index in FO_index_list:
        merged.append(index)
    merged = list(np.unique(merged))

    # the number of different frame-parameters in the given sub-block
    FO_par_len = len(merged) + sum([len(indices) for indices in FO_index_list])

    # initial vector of frame-parameters
    FO_par_list = par0 * FO_par_len

    ## Find out which frame-paramemters are intermediate (to be optimised).
    FO_par_opt_list = [] # Intermediate frame-parameters
    FO_par_opt_index_list = [] # Indices of intermediate frame-parameters
    for i, indices in enumerate(FO_index_list):
        for j, index in enumerate(indices):
            if index in np.unique(FO_index_list[i+1:]):
                # If there is another 'index' in the leftover gates,
                # then it is an intermediate frame-parameter.
                FO_par_opt_list = FO_par_opt_list+par0
                which_par = j + len(merged) + sum([len(a)
                                                for a in FO_index_list[:i]])
                FO_par_opt_index_list.append(which_par)

    # Later we need to know the indices of optimised frame-parameter in the
    # full circuit. We record [gate_index, i] for each intermediate
    # frame-parameter and return it with the optimised frame-parameters.
    # 'gate_index' is the gate-index of the gate just before the corresponding
    # frame.
    # 'i' indicates the index of corresponding wire (for the gate with
    # 'gate_index') of the intermediate frame.
    par_index_list_to_return = []
    for i in FO_par_opt_index_list:
        find_corr_gate_n = len(merged)
        for j, gate_indices in enumerate(FO_index_list):
            find_corr_gate_n += len(gate_indices)
            if i < find_corr_gate_n:
                par_index_list_to_return.append([FO_gate_numbers[j],
                                       i-find_corr_gate_n+len(gate_indices)])
                break

    def cost_function(x):
        ''' Negativity of block for given vector x of intermediate
        frame-parameters.
        '''
        neg = 1.
        full_par_list = make_par_list(FO_par_list, x, FO_par_opt_index_list,
                                      par0)
        # x only contains the intermediate frame-parameters.
        # We need to build 'full_par_list' for the whole block from 'x'.

        frames = {}
        for i, index in enumerate(merged):
            frames[str(index)] = full_par_list[par0_len*i:par0_len*(i+1)]

        gate_index = 0
        for gate in FO_gate_list:
            frames_in = [frames[str(y)] for y in FO_index_list[gate_index]]
            par_index = len(merged) + sum([len(a)
                                        for a in FO_index_list[:gate_index]])
            frames_out = []
            for i in range(len(FO_index_list[gate_index])):
                frames_out.append(full_par_list[par0_len*(par_index+i):
                                                par0_len*(par_index+i+1)])

            neg = neg*neg_gate_max(W_gate, gate, frames_in, frames_out)

            for i,index in enumerate(FO_index_list[gate_index]):
                frames[str(index)] = frames_out[i]
            gate_index += 1
        return np.array(np.real(np.log(neg)))

    grad_cost_function = grad(cost_function)
    def func(x):
        return cost_function(x), grad_cost_function(x)

    if len(FO_par_opt_index_list)==0:
        ## If there is no connected wire, we don't need to optimise anything.
        return [], []
    else:
        ## Optimise
        optimise_result = basinhopping(func, FO_par_opt_list,
          minimizer_kwargs={"method":"L-BFGS-B", "jac":True}, niter=niter)
        return optimise_result.x, par_index_list_to_return

def frame_opt(circuit, l, par0, W, **kwargs):
    '''
    Perform frame optimisation with spatial parameter 'n' and temporal
    parameter 'l'.
    Divide the circuit into blocks with at most 'l' gates and optimise the
    intermediate frames in each block.
    Return the full list of frame-parameters for the circuit, which includes
    the optimised intermidate frames.
    INPUT - circuit: given circuit
            l:       max number of gates included in frame optimisation
            par0:    ???
            W:       ???
    '''
    options = {'niter': 3}
    options.update(kwargs)

    gate_list = circuit['gate_list']
    index_list = circuit['index_list']
    n_states = len(circuit['state_list'])

    W_state = W[0]
    W_gate = W[1]
    W_meas = W[2]

    ## Find out full length of the vector of frame-parameters for the circuit
    par_len = n_states
    for indices in index_list:
        par_len += len(indices)

    ## Initial vector of frame-parameters -> reference frames
    par_list = par0*par_len

    ## Find a block of at most 'l' connected gates.
    gate_set = []
    indices_set = []
    gate_numbers = []
    for gate_n, indices in enumerate(index_list):
        gate_set.append(gate_list[gate_n])
        indices_set.append(indices)
        gate_numbers.append(gate_n)

        if len(gate_numbers)>=l:
            ### Found a new block with 'l' gates. Perform frame optimisaiton.
            block_par_list, par_indicates_list = block_frame_opt(gate_set,
              indices_set, gate_numbers, par0, W_gate, niter=options['niter'])

            par_indices_list = []
            for gate_n, i in par_indicates_list:
                par_index = i + n_states + sum([len(indices)
                                       for indices in index_list[:gate_n]])
                par_indices_list.append(par_index)
            par_list = make_par_list(par_list, block_par_list,
                                     par_indices_list, par0)
            # Updating frame-parameters.

            gate_set = []
            indices_set = []
            gate_numbers = []

    if len(gate_numbers)>1:
        ## If there are more than 1 gates left, perform frame optimisation
        ## for the smaller last block.
        block_par_list, par_indicates_list = block_frame_opt(gate_set,
          indices_set, gate_numbers, par0, W_gate, niter=options['niter'])

        par_indices_list = []
        for gate_n, i in par_indicates_list:
            par_index = i + n_states + sum([len(indices)
                                   for indices in index_list[:gate_n]])
            par_indices_list.append(par_index)
        par_list = make_par_list(par_list, block_par_list, par_indices_list,
                                 par0)

    neg_circuit_opt = neg_circuit(circuit, W, par_list, par0)
    print('--------------- FRAME OPTIMISATION with l =',l,'----------------')
    print('Wigner Log Neg:', neg_circuit(circuit, W, par0*par_len, par0))
    print('Optimised Log Neg:', neg_circuit_opt)
    print('----------------------------------------------------------------')

    return par_list, neg_circuit_opt
