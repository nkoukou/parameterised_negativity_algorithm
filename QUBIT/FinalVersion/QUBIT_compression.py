from functools import (reduce)
import numpy as np

def compress_circuit(circuit, n):
    '''
        circuit = {'state_list': states, 'gate_list': gates,
                   'index_list': indices, 'meas_list': measurements}
        n : spatial parameter
    '''
    target_index_list = circuit['index_list']
    target_gate_list = circuit['gate_list']

    disjoint_index_list = []
    disjoint_gate_list = []
    compressed_index_list = []
    compressed_gate_list = []

    for i in range(len(target_index_list)):
        target_index = target_index_list[i]
        target_gate = target_gate_list[i]
        d_loc = -1

        while target_index:
            disjoint_index_list, disjoint_gate_list, compressed_index_list, \
            compressed_gate_list, target_index, target_gate, d_loc = \
            compress_subroutine(disjoint_index_list, disjoint_gate_list,
            compressed_index_list, compressed_gate_list, target_index,
            target_gate, d_loc, n)

    for k in range(len(disjoint_index_list)):
        disjoint_index = disjoint_index_list[k]
        disjoint_gate = disjoint_gate_list[k]
        compressed_index_list.append(match_m_index(disjoint_index,n))
        compressed_gate_list.append(aligned_gate(disjoint_gate,disjoint_index,
                                             match_m_index(disjoint_index,n)))

    circuit['index_list'] = compressed_index_list
    circuit['gate_list'] = compressed_gate_list

    return circuit

def compress_subroutine(disjoint_index_list_in, disjoint_gate_list_in,
                        compressed_index_list_in, compressed_gate_list_in,
                        target_index_in, target_gate_in, d_loc_in, n):
    if d_loc_in == -1-len(disjoint_index_list_in):
        disjoint_index_list_out = disjoint_index_list_in
        disjoint_index_list_out.append(target_index_in)
        disjoint_gate_list_out = disjoint_gate_list_in
        disjoint_gate_list_out.append(target_gate_in)
        compressed_index_list_out = compressed_index_list_in
        compressed_gate_list_out = compressed_gate_list_in
        target_index_out = []
        target_gate_out = []
        d_loc_out = -1

    else:
        t_len = len(target_index_in)
        disjoint_index = disjoint_index_list_in[d_loc_in]
        disjoint_gate = disjoint_gate_list_in[d_loc_in]
        d_len = len(disjoint_index)
        merged_index = np.ndarray.tolist(np.unique(np.append(target_index_in,
                                                             disjoint_index)))
        m_len = len(merged_index)

        if m_len == t_len + d_len: ### disjointed index
            disjoint_index_list_out = disjoint_index_list_in
            disjoint_gate_list_out = disjoint_gate_list_in
            compressed_index_list_out = compressed_index_list_in
            compressed_gate_list_out = compressed_gate_list_in
            target_index_out = target_index_in
            target_gate_out = target_gate_in
            d_loc_out = d_loc_in -1
        elif m_len > n: ###
            disjoint_index_list_out = np.ndarray.tolist(np.delete(
                disjoint_index_list_in,d_loc_in,axis=0))
            disjoint_gate_list_out = np.ndarray.tolist(np.delete(
                disjoint_gate_list_in,d_loc_in,axis=0))
            compressed_index_list_out = compressed_index_list_in
            compressed_index_list_out.append(match_m_index(disjoint_index,n))
            ### when the final index length is smaller than 'n',
            ### find some dummy index to append.
            compressed_gate_list_out = compressed_gate_list_in
            compressed_gate_list_out.append(aligned_gate(disjoint_gate,
                disjoint_index,match_m_index(disjoint_index, n)))
            target_index_out = target_index_in
            target_gate_out = target_gate_in
            d_loc_out = d_loc_in
        elif m_len <= n: ###
            disjoint_index_list_out = np.ndarray.tolist(np.delete(
                disjoint_index_list_in,d_loc_in,axis=0))
            disjoint_gate_list_out = np.ndarray.tolist(np.delete(
                disjoint_gate_list_in,d_loc_in,axis=0))
            compressed_index_list_out = compressed_index_list_in
            compressed_gate_list_out = compressed_gate_list_in
            target_index_out = merged_index
            target_gate_out = merge_gate(disjoint_gate, target_gate_in,
                                disjoint_index, target_index_in, merged_index)
            d_loc_out = d_loc_in
        else:
            raise Exception('Something went wrong')

    return(disjoint_index_list_out, disjoint_gate_list_out,
           compressed_index_list_out, compressed_gate_list_out,
           target_index_out, target_gate_out, d_loc_out)

def aligned_gate(gate, index, target_index):
    ''' Converts gate with index so that it matches target_index.
        gate         - array
        index        - list of int
        target_index - list of int
    '''
    if not set(index).issubset(target_index):
        raise Exception('Indices do not match')

    if len(index)!=len(target_index):
        index_dup = [index[i] for i in range(len(index))]
        added_dim = 0
        for i in list(set(target_index).difference(index)):
            index_dup.append(i)
            added_dim += 1
        gate = np.kron(gate, np.eye(2**added_dim))
    else:
        index_dup = index

    new_target_index = [index_dup.index(target_index[i])
                        for i in range(len(target_index))]
    P_matrix = permutation_matrix(list(range(len(index_dup))),
                                  new_target_index, [2]*len(target_index))
    gate = np.dot(P_matrix, np.dot(gate, P_matrix.T))

    return gate

def permutation_matrix(initial_order,final_order,dimension_subsystems):

    subsystems_number = len(initial_order)
    # print(initial_order, final_order)
    # check that order and dimension tuples have the same length
    if ( subsystems_number != len(final_order) or
         subsystems_number != len(dimension_subsystems) ):
        raise RuntimeError("The length of the tuples passed to the function"+
                           "needs to be the same")

    # Create the list of basis for each subsystem
    initial_basis_list = list(map(lambda dim : basis(dim),
                                  dimension_subsystems))

    # Create all possible indices for the global basis
    indices = indices_list(dimension_subsystems)
    # Create permutation matrix P
    total_dim = np.product(np.array(dimension_subsystems))
    P_matrix = np.zeros((total_dim,total_dim))

    for index in indices:
        initial_vector_list = [initial_basis_list[n][i]
                               for n,i in enumerate(index)]
        final_vector_list = [initial_vector_list[i] for i in final_order]
        initial_vector = tensor(initial_vector_list)
        final_vector = tensor(final_vector_list)

        P_matrix += np.outer(final_vector,initial_vector)

    return P_matrix

def basis(dim):
    basis = []
    for i in range(dim):
        vec = np.zeros(dim)
        vec[i] = 1
        basis.append(vec)
    return basis

def tensor(array_list):
    return reduce(lambda x,y : np.kron(x,y), array_list)

def indices_list(dimension_tuple):
    number_subsystems = len(dimension_tuple)
    if number_subsystems != 0:
        subindices = [range(dim) for dim in dimension_tuple]
        return np.array(np.meshgrid(*subindices)).T.reshape(-1,
                                                            number_subsystems)
    else:
        return np.array([])

def match_m_index(index, m):
    index_out = index.copy()
    if len(index_out) > m:
        raise Exception('index length is longer than m')
    jj = 0
    while len(index_out) < m:
        if jj not in index_out:
            index_out.append(jj)
        jj += 1
    return index_out

def merge_gate(U1, U2, index1, index2, m_index):
    U1_match = aligned_gate(U1,index1,m_index)
    U2_match = aligned_gate(U2,index2,m_index)
    U_merged = np.dot(U2_match,U1_match)
    return U_merged
