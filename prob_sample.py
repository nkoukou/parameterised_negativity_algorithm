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
        sign_state = np.sign(qd_state)
        pd_state /= neg_state

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
        sign_gate = np.sign(qd_gate)
        for i in range(2*len(idx)):
            pd_gate = pd_gate.swapaxes(i,2*len(idx)+i)
        pd_gate = pd_gate/neg_gate
        for i in range(2*len(idx)):
            pd_gate = pd_gate.swapaxes(i,2*len(idx)+i)

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

def sample_circuit(circuit, qd_output, sample_size = int(1e4)):
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

        # Input states
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

            p_in1 = current_ps_point[idx[0]]//2
            q_in1 = current_ps_point[idx[0]]%2
            p_in2 = current_ps_point[idx[1]]//2
            q_in2 = current_ps_point[idx[1]]%2

            prob = pd_gate[p_in1,q_in1,p_in2,q_in2].flatten()
            neg = qd_output['neg_list_gates'][g][p_in1, q_in1, p_in2, q_in2]
            sign = qd_output['sign_list_gates'][g][p_in1, q_in1,
                                                   p_in2, q_in2].flatten()

            ps_point = nr.choice(np.arange(len(prob)), p=prob)
            current_ps_point[idx[0]] = ps_point//4
            current_ps_point[idx[1]] = ps_point%4
            p_estimate *= neg*sign[ps_point]

        exp_qaoa = 0
        # Measurement
        for m, meas in enumerate(circuit['meas_list']):
            wf = qd_output['qd_list_meas'][m].flatten() # QD NOT PD
            p_estimate *= wf[current_ps_point[m]]

            temp = (m+1)%len(circuit['meas_list'])
            wf_next = qd_output['qd_list_meas'][temp].flatten() # QD NOT PD
            exp_qaoa += 0.5*wf_next[current_ps_point[m]]*wf[current_ps_point[m]]

        return p_estimate, exp_qaoa

    start_time = time.time()
    p_estimate = np.zeros(sample_size)
    exp_qaoa = np.zeros(sample_size)
    for n in range(sample_size):
        est, test = sample_iter()
        p_estimate[n] = est
        exp_qaoa[n] = test
    sampling_time = time.time() - start_time

    print('====================== Sampling Results ======================')
    print('p_estimate:    ', np.average(p_estimate))
    print('exp_qaoa:    ', np.average(exp_qaoa))
    print('Sample size:   ', sample_size)
    print('Sampling time: ', sampling_time)
    print('==============================================================')

    return p_estimate, exp_qaoa


def sample_circuit_2q(compressed_circuit, qd_output,
                   sample_size = 10000, **kwargs):
    options = {'option 1': True}
    options.update(kwargs)

    def sample_iter():
        current_ps_point = []
        p_estimate = 1.

        # Input states
        qudit_index = 0
        for pd_state in qd_output['pd_list_states']:
            prob = pd_state.flatten()
            neg = qd_output['neg_list_states'][qudit_index]
            sign_array = qd_output['sign_list_states'][qudit_index].flatten()

            ps_point = nr.choice(np.arange(len(prob)), p=prob)
            current_ps_point.append(ps_point)
            p_estimate *= neg*sign_array[ps_point]
            qudit_index +=1

        # Gates
        gate_index = 0
        for pd_gate in qd_output['pd_list_gates']:
            idx = compressed_circuit['index_list'][gate_index]

            p_in1 = current_ps_point[idx[0]]//2
            q_in1 = current_ps_point[idx[0]]%2
            p_in2 = current_ps_point[idx[1]]//2
            q_in2 = current_ps_point[idx[1]]%2

            prob = pd_gate[p_in1,q_in1,p_in2,q_in2].flatten()
            neg = qd_output['neg_list_gates'][gate_index][p_in1,q_in1,
                                                          p_in2,q_in2]
            sign_array = qd_output['sign_list_gates'][gate_index][p_in1,
                                                q_in1,p_in2,q_in2].flatten()

            ps_point = nr.choice(np.arange(len(prob)), p=prob)
            current_ps_point[idx[0]] = ps_point//4
            current_ps_point[idx[1]] = ps_point%4
            p_estimate *= neg*sign_array[ps_point]

            gate_index += 1

        # Measurement
        meas_index = 0
        qudit_index = 0
        for meas in compressed_circuit['meas_list']:
            if str(meas) =='/':
                qudit_index += 1
            else:
                WF = qd_output['qd_list_meas'][meas_index].flatten()
                #THIS SHOULD BE QD NOT PD
                meas_index += 1
                p_estimate *= WF[current_ps_point[qudit_index]]
                qudit_index += 1
        return p_estimate

    start_time = time.time()
    outcome_list = []
    for n in range(sample_size):
        outcome_list.append(sample_iter())
    sampling_time = time.time() - start_time

    # Print the final results
    print('====================== Sampling Results ======================')
    print('p_estimate:    ', np.sum(outcome_list)/sample_size)
    print('Sample size:   ', sample_size)
    print('Sampling time: ', sampling_time)
    print('==============================================================')

    return np.array(outcome_list)

def sample_circuit_3q(compressed_circuit, qd_output, sample_size = 10000,
                      **kwargs):
    options = {'option 1': True}
    options.update(kwargs)

    def sample_iter():
        current_ps_point = []
        p_estimate = 1.

        # Input states
        qudit_index = 0
        for pd_state in qd_output['pd_list_states']:
            prob = pd_state.flatten()
            neg = qd_output['neg_list_states'][qudit_index]
            sign_array = qd_output['sign_list_states'][qudit_index].flatten()

            ps_point = nr.choice(np.arange(len(prob)), p=prob)
            current_ps_point.append(ps_point)
            p_estimate *= neg*sign_array[ps_point]
            qudit_index +=1

        # Gates
        gate_index = 0
        for pd_gate in qd_output['pd_list_gates']:
            idx = compressed_circuit['index_list'][gate_index]

            p_in1 = current_ps_point[idx[0]]//2
            q_in1 = current_ps_point[idx[0]]%2
            p_in2 = current_ps_point[idx[1]]//2
            q_in2 = current_ps_point[idx[1]]%2
            p_in3 = current_ps_point[idx[2]]//2
            q_in3 = current_ps_point[idx[2]]%2

            prob = pd_gate[p_in1,q_in1,p_in2,q_in2,p_in3,q_in3].flatten()
            neg = qd_output['neg_list_gates'][gate_index][p_in1,q_in1,
                    p_in2,q_in2,p_in3,q_in3]
            sign_array = qd_output['sign_list_gates'][gate_index][p_in1,
                           q_in1,p_in2,q_in2,p_in3,q_in3].flatten()

            ps_point = nr.choice(np.arange(len(prob)), p=prob)
            current_ps_point[idx[0]] = ps_point//16
            current_ps_point[idx[1]] = (ps_point%16)//4
            current_ps_point[idx[2]] = ps_point%4
            p_estimate *= neg*sign_array[ps_point]

            gate_index += 1

        # Measurement
        meas_index = 0
        qudit_index = 0
        for meas in compressed_circuit['meas_list']:
            if str(meas) =='/':
                qudit_index += 1
            else:
                WF = qd_output['qd_list_meas'][meas_index].flatten()
                # THIS SHOULD BE QD NOT PD
                meas_index += 1
                p_estimate *= WF[current_ps_point[qudit_index]]
                qudit_index += 1
        return p_estimate

    '''
    The function actually performing the Monte Carlo sampling as outlined in
    Pashayan et al. (2015) for the compressed version of the given circuit
    and the parameter list. It prints the method we used, the sampling result
    (p_estimate), and the computation time. It also returns the sampling
    result and the full list of p_estimate of each iteration ('plot').
    '''

    start_time = time.time()
    outcome_list = []
    for n in range(sample_size):
        outcome_list.append(sample_iter())
    sampling_time = time.time() - start_time

    # Print the final results
    print('====================== Sampling Results ======================')
    print('p_estimate:    ', np.sum(outcome_list)/sample_size)
    print('Sample size:   ', sample_size)
    print('Sampling time: ', sampling_time)
    print('==============================================================')

    return np.array(outcome_list)















