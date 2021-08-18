import numpy as np
from QUBIT_circuit_generator import aligned_gate

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

def compress_subroutine(disjoint_index_list_in, disjoint_gate_list_in, compressed_index_list_in, compressed_gate_list_in, target_index_in, target_gate_in, d_loc_in, n):
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
        merged_index = np.ndarray.tolist(np.unique(np.append(target_index_in,disjoint_index)))
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
            disjoint_index_list_out = np.ndarray.tolist(np.delete(disjoint_index_list_in,d_loc_in,axis=0))
            disjoint_gate_list_out = np.ndarray.tolist(np.delete(disjoint_gate_list_in,d_loc_in,axis=0))
            compressed_index_list_out = compressed_index_list_in
            compressed_index_list_out.append(match_m_index(disjoint_index,n)) 
            ### when the final index length is smaller than 'n', find some dummy index to append.
            compressed_gate_list_out = compressed_gate_list_in
            compressed_gate_list_out.append(aligned_gate(disjoint_gate,disjoint_index,match_m_index(disjoint_index, n)))
            target_index_out = target_index_in
            target_gate_out = target_gate_in
            d_loc_out = d_loc_in
        elif m_len <= n: ###
            disjoint_index_list_out = np.ndarray.tolist(np.delete(disjoint_index_list_in,d_loc_in,axis=0))
            disjoint_gate_list_out = np.ndarray.tolist(np.delete(disjoint_gate_list_in,d_loc_in,axis=0))
            compressed_index_list_out = compressed_index_list_in
            compressed_gate_list_out = compressed_gate_list_in
            target_index_out = merged_index
            target_gate_out = merge_gate(disjoint_gate,target_gate_in,disjoint_index,target_index_in,merged_index)
            d_loc_out = d_loc_in
        else:
            raise Exception('Something went wrong')
        
    return disjoint_index_list_out, disjoint_gate_list_out, compressed_index_list_out, compressed_gate_list_out, target_index_out, target_gate_out, d_loc_out

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
            disjoint_index_list, disjoint_gate_list, compressed_index_list, compressed_gate_list, target_index, target_gate, d_loc = \
            	compress_subroutine(disjoint_index_list, disjoint_gate_list, compressed_index_list, compressed_gate_list, target_index, target_gate, d_loc, n)

    for k in range(len(disjoint_index_list)):
        disjoint_index = disjoint_index_list[k]
        disjoint_gate = disjoint_gate_list[k]
        compressed_index_list.append(match_m_index(disjoint_index,n))
        compressed_gate_list.append(aligned_gate(disjoint_gate,disjoint_index,match_m_index(disjoint_index,n)))

    circuit['index_list'] = compressed_index_list
    circuit['gate_list'] = compressed_gate_list

    return circuit

