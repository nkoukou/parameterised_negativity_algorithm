import autograd.numpy as np
from autograd import(grad)
from scipy.optimize import(basinhopping)

def init_x_list(circuit, x0):
    N = len(circuit['state_list'])
    x_list = []
    x_index_gate = []
    x_index_meas = []

    for q_index in range(N):
        x_list.append(x0)

    running_idx = np.arange(N)
    idx_end = N-1

    index_list = circuit['index_list']
    for j in range(len(index_list)):
        q_index_list = index_list[j]
        x_idx_in = []
        x_idx_out = []
        for q_index in q_index_list:
            x_idx_in.append(running_idx[q_index])

            x_list.append(x0)
            idx_end = idx_end + 1
            x_idx_out.append(idx_end)
            running_idx[q_index] = idx_end

        x_index_gate.append([x_idx_in,x_idx_out])
    x_index_meas = running_idx
    return [x_list, x_index_gate, x_index_meas]
# x_index[gate_index] = [[x1_in_idx,x2_in_idx,...],
#                         [x1_out_idx,x2_out_idx,...]]
# x_list[x_idx] gives parameter x

def get_connected_index(x_circuit,target_circuit_index):
    [x_list, x_index_gate, x_index_meas] = x_circuit
    [target_state_index, target_gate_index, target_meas_index
     ] = target_circuit_index

    idx_connect = []

    for i in target_state_index:
        for j in target_gate_index:
            idx_connect = idx_connect + np.ndarray.tolist(np.intersect1d([i],
              x_index_gate[j][0]).flatten())

    for i in target_gate_index:
        for j in target_gate_index:
            if i != j:
                idx_connect = idx_connect + np.ndarray.tolist(np.intersect1d(
                  x_index_gate[i][1],x_index_gate[j][0]).flatten())

    for i in target_gate_index:
        for j in target_meas_index:
            idx_connect = idx_connect + np.ndarray.tolist(np.intersect1d(
              x_index_gate[i][1],x_index_meas[j]).flatten())

    return idx_connect

def replace_x_list(x_list,x_list_target,idx_connect):
    x_list_out = x_list.copy()
    i=0
    for idx in idx_connect:
        x_list_out[idx] = x_list_target[i]
        i += 1
    return x_list_out

def get_target_circuit_block(x_circuit,x_target_index):
    [x_list, x_index_gate, x_index_meas] = x_circuit
    target_state_index = []
    target_gate_index = []
    target_meas_index = []

    qubit_num = len(x_index_meas)
    for idx in range(qubit_num):
        if (idx in x_target_index):
            target_state_index.append(idx)
    for idx in range(len(x_index_gate)):
        if set(np.array(x_index_gate[idx]).flatten()).intersection(
                x_target_index):
            target_gate_index.append(idx)
    for idx in range(qubit_num):
        if (x_index_meas[idx] in x_target_index):
            target_meas_index.append(idx)

    target_circuit_index = [target_state_index,target_gate_index,
                            target_meas_index]
    return target_circuit_index

def neg_gate_max(W_gate, gate, par_list_in, par_list_out):
    n = len(par_list_out)
    return np.abs(W_gate(gate, par_list_in, par_list_out)).sum(axis=tuple(
                  np.arange(2*n,4*n))).max()
#     return np.abs(W_gate(gate, par_list_in, par_list_out)).sum(axis=0).max()


def get_negativity_block(W,circuit,x_circuit,target_circuit_index):
    W_state = W[0]
    W_gate = W[1]
    W_meas = W[2]

    state_list = circuit['state_list']
    gate_list = circuit['gate_list']
    meas_list = circuit['meas_list']

    [x_list, x_index_gate, x_index_meas] = x_circuit
    [target_state_index,target_gate_index,target_meas_index
     ] = target_circuit_index

    neg = 1
    for state_index in target_state_index:
        x = x_list[state_index]
        neg *= np.abs(W_state(state_list[state_index], x)).sum()
    for gate_index in target_gate_index:
        [x_idx_in, x_idx_out] = x_index_gate[gate_index]
        x_in = [x_list[x_idx] for x_idx in x_idx_in]
        x_out = [x_list[x_idx] for x_idx in x_idx_out]
        neg *= neg_gate_max(W_gate, gate_list[gate_index], x_in, x_out)
    for meas_index in target_meas_index:
        x = x_list[x_index_meas[meas_index]]
        neg *= np.abs(W_meas(meas_list[meas_index], x)).max()
    return neg

def get_negativity_circuit(W,circuit,x_circuit):
    target_circuit_index = [np.arange(len(circuit['state_list'])),
                            np.arange(len(circuit['gate_list'])),
                            np.arange(len(circuit['meas_list']))]
    return get_negativity_block(W,circuit,x_circuit,target_circuit_index)

def opt_negativity_block(W,circuit,x_circuit,target_circuit_index,niter=3,
                         show_log=False):
    [x_list, x_index_gate, x_index_meas] = x_circuit
    idx_connect = get_connected_index(x_circuit,target_circuit_index)
    len_x = len(x_list[0])

    def cost_function(x):
        x_list_target = np.reshape(x,(-1,len_x))
        x_replaced_list = replace_x_list(x_list, x_list_target, idx_connect)
        x_replaced_circuit = [x_replaced_list, x_index_gate, x_index_meas]
        return get_negativity_block(W, circuit, x_replaced_circuit,
                                    target_circuit_index)

    grad_cost_function = grad(cost_function)
    def func(x):
        return cost_function(x), grad_cost_function(x)

    if len(idx_connect)==0:
        ## If there is no connected wire, we don't need to optimise anything.
        return x_circuit
    else:
        ## Optimise
        x_ref_list = [x_list[idx] for idx in idx_connect]

        optimise_result = basinhopping(func, x_ref_list,
          minimizer_kwargs={"method":"L-BFGS-B", "jac":True}, niter=niter)
        x_list_target_opt = np.reshape(optimise_result.x,(-1,len_x))
        x_list_all_opt = replace_x_list(x_list,x_list_target_opt,idx_connect)
        x_circuit_opt = [x_list_all_opt,x_index_gate, x_index_meas]

        if show_log==True:
            neg_init = get_negativity_circuit(W,circuit,x_circuit)
            print('Initial log-negativity:', np.log(neg_init))
            print('target_state_index',target_circuit_index[0])
            print('target_gate_index',target_circuit_index[1])
            print('target_meas_index',target_circuit_index[2])
            print('Connected index:',idx_connect)
            # print('Optimized log-negativity:',np.log(neg_out))
        return x_circuit_opt

def random_circuit_opt(W, circuit, x_circuit, l_state=2, l_gate=5, l_meas=2,
                       niter=3, show_log=False):
    target_state_index = np.random.choice(np.arange(
                           len(circuit['state_list'])),l_state,replace=False)
    target_gate_index = np.random.choice(np.arange(len(circuit['gate_list'])),
                                         l_gate,replace=False)
    target_meas_index = np.random.choice(np.arange(len(circuit['meas_list'])),
                                         l_meas,replace=False)

    target_circuit_index = [target_state_index,target_gate_index,
                            target_meas_index]
    idx_connect = get_connected_index(x_circuit,target_circuit_index)

    get_negativity_block(W,circuit,x_circuit,target_circuit_index)
    x_circuit_out = opt_negativity_block(W, circuit, x_circuit,
                                         target_circuit_index,niter=niter)
    neg_out = get_negativity_circuit(W,circuit,x_circuit_out)

    return x_circuit_out, neg_out

def random_para_opt(W,circuit,x_circuit,l=3,niter=3,show_log=False):
    neg_init = get_negativity_circuit(W,circuit,x_circuit)
    if show_log == True:
        print('Initial parameters\n',np.real(x_circuit[0]))
        print('Initial log-negativity:\t', np.log(neg_init))

    neg_list = [neg_init]
    itr_count = 0

    x_range = np.arange(len(x_circuit[0]))
    np.random.shuffle(x_range)

    x_circuit_out = x_circuit

    for x_init_idx in range(int(len(x_circuit[0])/l)):
#     for x_init_idx in range(len(x_circuit[0])):
        x_target_index = x_range[l*x_init_idx:np.min([l*(x_init_idx + 1),
                                                      len(x_circuit[0])])]
        # x_target_index = x_range[x_init_idx:np.min([x_init_idx + l,
        #                                             len(x_circuit[0])])]
        target_circuit_index = get_target_circuit_block(x_circuit_out,
                                                        x_target_index)
        idx_connect = get_connected_index(x_circuit_out, target_circuit_index)

        get_negativity_block(W,circuit,x_circuit_out,target_circuit_index)
        x_circuit_out = opt_negativity_block(W,circuit,x_circuit_out,
                                             target_circuit_index,niter=niter)
        neg_out = get_negativity_circuit(W,circuit,x_circuit_out)
        if show_log == True:
            print('Optimized log-negativity:\t', np.log(neg_out))
        neg_list.append(neg_out)
    return x_circuit_out, neg_list

def sequential_para_opt(W, circuit, x_circuit, l=3, niter=3, show_log=False):
    neg_init = get_negativity_circuit(W,circuit,x_circuit)
    if show_log == True:
        print('Initial parameters\n',np.real(x_circuit[0]))
        print('Initial log-negativity:\t', np.log(neg_init))

    neg_list = [neg_init]
    itr_count = 0

    x_circuit_out = x_circuit
    for x_init_idx in range(len(x_circuit[0])):
        x_target_index = np.arange(x_init_idx,np.min([x_init_idx + l,
                                                      len(x_circuit[0])]))
        target_circuit_index = get_target_circuit_block(x_circuit_out,
                                                        x_target_index)
        idx_connect = get_connected_index(x_circuit_out, target_circuit_index)

        get_negativity_block(W,circuit,x_circuit_out,target_circuit_index)
        x_circuit_out = opt_negativity_block(W, circuit, x_circuit_out,
                                             target_circuit_index,niter=niter)
        neg_out = get_negativity_circuit(W, circuit, x_circuit_out)
        if show_log == True:
            print('Optimized log-negativity:\t', np.log(neg_out))
        neg_list.append(neg_out)
    return x_circuit_out, neg_list
