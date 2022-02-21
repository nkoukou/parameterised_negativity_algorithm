import os
from numba import (jit, prange, types)
import numpy as np
import numpy.random as nr
import time

@jit(nopython=True, cache=True) # Comment out this line to ignore numba
def sample_fast(sample_size, meas_list, index_list,
          qd_list_states, qd_list_gates, qd_list_meas, pd_list_states,
          pd_list_gates, pd_list_meas, sign_list_states, sign_list_gates,
          sign_list_meas, neg_list_states, neg_list_gates, neg_list_meas):
    sample_size = np.int64(sample_size)
    N = meas_list.shape[0]
    p_out = 0 # np.zeros(sample_size) #
    for n in prange(sample_size):
        if n%(sample_size//10)==0:
            print("------")
            print((n/sample_size)*100, "%")

        current_ps_point = np.zeros(N, dtype=np.int64)
        p_estimate = 1.

        # Input states
        for s in range(N):
            ps_point = np.arange(len(pd_list_states[s]), dtype=np.int64
                         )[np.searchsorted(np.cumsum(pd_list_states[s]),
                                           np.random.random(), side="right")]
            current_ps_point[s] = ps_point
            p_estimate *= neg_list_states[s]*sign_list_states[s][ps_point]

        # Gates
        for g in range(index_list.shape[0]):
            idx = index_list[g]

            arr_dim = np.log2(len(pd_list_gates[g]))/2
            pq_in = 0
            for i in range(len(idx)):
                pq_in += current_ps_point[idx[i]]//2 * 2**(2*(arr_dim-i)-1)
                pq_in += current_ps_point[idx[i]]%2 * 2**(2*(arr_dim-i)-2)
            pq_in = np.int64(pq_in)

            prob = pd_list_gates[g,pq_in:pq_in+np.int64(2**arr_dim)]
            prob = prob/prob.sum()
            sign = sign_list_gates[g,pq_in:pq_in+np.int64(2**arr_dim)]
            neg = neg_list_gates[g,np.int64(pq_in//(2**arr_dim))]

            # The next line is the cause of the segmentation fault.
            # I suspect it's related to the type/values/normalisation of
            # array prob.
            ps_point = np.arange(len(prob), dtype=np.int64)[np.searchsorted(
              np.cumsum(prob), np.random.random(), side="right")]

            prob_dim = np.log2(len(prob))/2
            for i in range(len(idx)):
                current_ps_point[idx[i]] = int(ps_point/4**(prob_dim-1-i))%4
            p_estimate *= neg*sign[ps_point]


        # Measurement
        for m in range(N):
            p_estimate *= qd_list_meas[m,current_ps_point[m]]
        p_out += 1./sample_size * p_estimate
        # p_out[n] = p_estimate

        # exp_qaoa = p_estimate
        # temp = 0
        # for m in range(N):
        #     p_estimate *= qd_list_meas[m,current_ps_point[m]]
        #     m_next = (m+1)%N
        #     temp += 0.5 * qd_list_meas[m_next,current_ps_point[m_next]]*\
        #                   qd_list_meas[m,current_ps_point[m]]
        # exp_qaoa *= temp
        # p_out += 1./sample_size * exp_qaoa
    # p_out = (1./sample_size) * np.cumsum(p_out)
    return p_out

def get_qd_output(circuit, par_list, ps):
    '''
    Calculate the quasi-probability distribution of each circuit element
    and return them as a list of [state qds, gate qds, measurement qds].
    '''
    par_idx_states = np.arange(0, len(circuit["state_list"]))
    par_idx_gates = par_list[1]
    par_idx_meas = par_list[2]
    par_vals = par_list[0]

    qd_list_states = []   # qd lists
    qd_list_gates = []
    qd_list_meas = []

    neg_list_states = []  # qd negativity lists
    neg_list_gates = []
    neg_list_meas = []

    sign_list_states = [] # sign of qd lists
    sign_list_gates = []
    sign_list_meas = []

    pd_list_states = []   # normalised prob dist lists
    pd_list_gates = []
    pd_list_meas = []

    # States
    for s, state in enumerate(circuit['state_list']):
        x = par_vals[par_idx_states[s]]

        qd_state = ps.W_state(state, x)
        pd_state = np.abs(qd_state)
        neg_state = pd_state.sum()
        pd_state /= neg_state
        sign_state = np.sign(qd_state)

        qd_list_states.append(qd_state)
        pd_list_states.append(pd_state)
        neg_list_states.append(neg_state)
        sign_list_states.append(sign_state)

    # Gates
    for g, gate in enumerate(circuit['gate_list']):
        idx = circuit['index_list'][g]

        x_in, x_out = [], []
        for k in par_idx_gates[g][0]:
            x_in.append(par_vals[k])
        for k in par_idx_gates[g][1]:
            x_out.append(par_vals[k])

        qd_gate = ps.W_gate(gate, x_in, x_out)
        pd_gate = np.abs(qd_gate)
        neg_gate = pd_gate.sum(axis=tuple(range(2*len(idx),4*len(idx))))
        for i in range(2*len(idx)):
            pd_gate = pd_gate.swapaxes(i,2*len(idx)+i)
        pd_gate = pd_gate/neg_gate
        for i in range(2*len(idx)):
            pd_gate = pd_gate.swapaxes(i,2*len(idx)+i)
        sign_gate = np.sign(qd_gate)

        qd_list_gates.append(qd_gate)
        pd_list_gates.append(pd_gate)
        neg_list_gates.append(neg_gate)
        sign_list_gates.append(sign_gate)

    # Measurements
    for m, meas in enumerate(circuit['meas_list']):
        x = par_vals[par_idx_meas[m]]

        qd_meas = ps.W_meas(meas, x)
        pd_meas = np.abs(qd_meas)
        neg_meas = np.max(pd_meas)
        sign_meas = np.sign(qd_meas)

        qd_list_meas.append(qd_meas)
        pd_list_meas.append(pd_meas)
        neg_list_meas.append(neg_meas)
        sign_list_meas.append(sign_meas)

    output = {
    'qd_list_states': qd_list_states, 'qd_list_gates': qd_list_gates,
    'qd_list_meas': qd_list_meas, 'pd_list_states': pd_list_states,
    'pd_list_gates': pd_list_gates, 'pd_list_meas': pd_list_meas,
    'neg_list_states': neg_list_states, 'neg_list_gates': neg_list_gates,
    'neg_list_meas': neg_list_meas, 'sign_list_states': sign_list_states,
    'sign_list_gates': sign_list_gates, 'sign_list_meas': sign_list_meas
    }
    return output

def prepare_sampler(circuit, par_list, ps):
    meas_list = np.stack(circuit["meas_list"]).astype(np.float64)
    index_list = np.array(circuit["index_list"]).astype(np.int64)

    output = get_qd_output(circuit, par_list, ps)

    qd_list_states = np.stack([dist.flatten().astype(np.float64)
                               for dist in output["qd_list_states"]])
    qd_list_gates = np.stack([dist.flatten().astype(np.float64)
                              for dist in output["qd_list_gates"]])
    qd_list_meas = np.stack([dist.flatten().astype(np.float64)
                             for dist in output["qd_list_meas"]])

    neg_list_states = np.array(output["neg_list_states"]).astype(np.float64)
    neg_list_gates = np.stack([dist.flatten().astype(np.float64)
                               for dist in output["neg_list_gates"]])
    neg_list_meas = np.array(output["neg_list_meas"]).astype(np.float64)

    pd_list_states = np.stack([dist.flatten().astype(np.float64)
                               for dist in output["pd_list_states"]])
    pd_list_gates = np.stack([dist.flatten().astype(np.float64)
                              for dist in output["pd_list_gates"]])
    pd_list_meas = np.stack([dist.flatten().astype(np.float64)
                             for dist in output["pd_list_meas"]])

    sign_list_states = np.stack([dist.flatten().astype(np.float64)
                                 for dist in output["sign_list_states"]])
    sign_list_gates = np.stack([dist.flatten().astype(np.float64)
                                for dist in output["sign_list_gates"]])
    sign_list_meas = np.stack([dist.flatten().astype(np.float64)
                               for dist in output["sign_list_meas"]])

    return(meas_list, index_list, qd_list_states, qd_list_gates, qd_list_meas,
           pd_list_states, pd_list_gates, pd_list_meas, sign_list_states,
           sign_list_gates, sign_list_meas, neg_list_states, neg_list_gates,
           neg_list_meas)







