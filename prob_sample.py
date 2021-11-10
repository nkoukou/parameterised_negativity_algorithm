import numpy as np
import numpy.random as nr
import time
import itertools as it

from phase_space import(PhaseSpace)


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

def sample_circuit(circuit, qd_output, sample_size=int(1e4)):
    '''
    The function performing the Monte Carlo sampling as outlined in
    Pashayan et al. (2015) for the compressed version of the given circuit
    and the parameter list. It prints the method we used, the sampling result
    (p_estimate), and the computation time. It also returns the sampling
    result and the full list of p_estimate of each iteration ('plot').
    '''
    def sample_iter():
        current_ps_point = []
        p_estimate = 1.

        # Input states - PD NOT QD
        for s, pd_state in enumerate(qd_output['pd_list_states']):
            prob = pd_state.flatten()
            neg = qd_output['neg_list_states'][s]
            sign = qd_output['sign_list_states'][s].flatten()

            ps_point = nr.choice(np.arange(len(prob)), p=prob)
            current_ps_point.append(ps_point)
            p_estimate *= neg*sign[ps_point]

        # Gates
        for g, pd_gate in enumerate(qd_output['pd_list_gates']):
            idx = circuit['index_list'][g]

            pq_in = [[current_ps_point[idx[i]]//2, current_ps_point[idx[i]]%2]
                     for i in range(len(idx))]
            pq_in = tuple([item for sublist in pq_in for item in sublist])

            prob = pd_gate[pq_in].flatten()
            neg = qd_output['neg_list_gates'][g][pq_in]
            sign = qd_output['sign_list_gates'][g][pq_in].flatten()

            ps_point = nr.choice(np.arange(len(prob)), p=prob)
            str_repr = np.base_repr(ps_point, 4).zfill(len(idx))
            for i in range(len(idx)):
                current_ps_point[idx[i]] = int(str_repr[i])
            p_estimate *= neg*sign[ps_point]

        # Measurement - QD NOT PD
        exp_qaoa = p_estimate
        temp = 0
        for m, meas in enumerate(circuit['meas_list']):
            wf = qd_output['qd_list_meas'][m].flatten()
            p_estimate *= wf[current_ps_point[m]]

            idx_temp = (m+1)%len(circuit['meas_list'])
            wf_next = qd_output['qd_list_meas'][idx_temp].flatten()
            temp += 0.5 * wf_next[current_ps_point[m]] * \
                          wf[current_ps_point[m]]

        exp_qaoa *= temp

        return p_estimate, exp_qaoa

    start_time = time.time()
    p_estimate = np.zeros(sample_size)
    exp_qaoa = np.zeros(sample_size)
    print("\n"+"SAMPLING")
    for n in range(sample_size):
        if n%(sample_size//10)==0: print(n/sample_size*100, "%")
        p_estimate[n], exp_qaoa[n] = sample_iter()
    sampling_time = time.time() - start_time

    print('====================== Sampling Results ======================')
    print('p_estimate:    ', np.average(p_estimate))
    print('exp_qaoa:      ', np.average(exp_qaoa))
    print('Sample size:   ', sample_size)
    print('Sampling time: ', sampling_time)
    print('==============================================================')

    return p_estimate, exp_qaoa















