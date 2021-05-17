from functools import(reduce)
from itertools import(permutations)
from random import(shuffle)
import numpy as np
import numpy.random as nr
import functools as fc
from QUBIT_circuit_components import(makeState, makeGate, makeMeas)
from QUBIT_state_functions import(evolve)

IDC  = makeGate('1')
SWAP = makeGate('S')

def random_circuit(qudit_num, C1qGate_num, TGate_num, CSUMGate_num, Toff_num,
                   given_state=None, given_measurement=1):
    ''' Inputs:
        qudit_num         - int
        C1qGate_num       - int
        TGate_num         - int
        CSUMGate_num      - int
        given_state       - None or 0 (all zeros) or string
        given_measurement - string or int (number of measurement modes)

        Output:
        circuit = {'state_list': states, 'gate_list': gates,
                   'index_list': indices, 'meas_list': measurements}
    '''
    # States
    if given_state is None:
        char = ['0', '1'] # Full list: ['0','1','2','+','m','S','N','T']
        prob = [1/len(char)]*len(char)

        given_state = ''
        for i in range(qudit_num):
            given_state += nr.choice(char, p=prob)
    elif given_state==0:
        given_state = '0'*qudit_num
    else:
        if len(given_state)!=qudit_num:
            raise Exception('Number of qudits must be %d'%(qudit_num))
    states = []
    for s in given_state:
        states.append(makeState(s))

    # Gates
    char1q = ['H', 'K'] # Full list: ['H', 'K', 'X', 'Z', '1']
    prob1q = [1/len(char1q)]*len(char1q)
    gates_seq = []
    for i in range(C1qGate_num):
        gate = makeGate(nr.choice(char1q, p=prob1q))
        index = [nr.randint(qudit_num)]
        gates_seq.append((gate, index))
    for i in range(TGate_num):
        gate = makeGate('T')
        index = [nr.randint(qudit_num)]
        gates_seq.append((gate, index))
    for i in range(CSUMGate_num):
        gate = makeGate('C+')
        index = list(nr.choice(qudit_num, size=2, replace=False))
        gates_seq.append((gate, index))
    for i in range(Toff_num):
        gate = makeGate('A')
        index = list(nr.choice(qudit_num, size=3, replace=False))
        gates_seq.append((gate, index))
    nr.shuffle(gates_seq)
    gates, indices = zip(*gates_seq)
    gates, indices = list(gates), list(indices)

    # Measurements
    if type(given_measurement)==int:
        char = ['0'] # Full list: ['0','1','2','+','m','S','N','T']
        prob = [1/len(char)]*len(char)

        meas = ['/']*qudit_num
        for i in range(given_measurement):
            meas[i] = nr.choice(char, p=prob)

        given_measurement = ''
        for m in meas:
            given_measurement += m
    else:
        if len(given_measurement)!=qudit_num:
            raise Exception('Number of qudits is %d'%(qudit_num))
    measurements = []
    for m in given_measurement:
        measurements.append(makeMeas(m))

    circuit = {'state_list': states, 'gate_list': gates,
               'index_list': indices, 'meas_list': measurements}
    return circuit

def random_connected_circuit(qudit_num, circuit_length, Tgate_prob=1/3,
                   given_state=None, given_measurement=1, method='r'):
    ''' Inputs:
        qudit_num         - int
        circuit_length    - int
        Tgate_prob        - float
        given_state       - None or 0 (all zeros) or string
        given_measurement - string or int (number of measurement modes)

        Output:
        circuit = {'state_list': states, 'gate_list': gates,
                   'index_list': indices, 'meas_list': measurements}

        circuit is fully connected, i.e. there are no disentangled wires.
    '''
    # States
    if given_state is None:
        char = ['0', '1']
        prob = [1/len(char)]*len(char)

        given_state = ''
        for i in range(qudit_num):
            given_state += nr.choice(char, p=prob)
    elif given_state==0:
        given_state = '0'*qudit_num
    else:
        if len(given_state)!=qudit_num:
            raise Exception('Number of qudits must be %d'%(qudit_num))
    states = []
    for s in given_state:
        states.append(makeState(s))

    # Indices
    indices = get_index_list(circuit_length, qudit_num, method='c')

    # Gates
    gates = []
    if type(Tgate_prob)==int:
        char = ['1', 'H', 'K']
        prob = [1/len(char)]*len(char)
        chars = ['T']*Tgate_prob + [nr.choice(char, p=prob)
                      for g in range(2*circuit_length - Tgate_prob)]
        shuffle(chars)
        for g in range(circuit_length):
            U1qA = makeGate(chars[2*g])
            U1qB = makeGate(chars[2*g+1])
            U_AB_loc = np.kron(U1qA, U1qB)
            csum = 'C+' if indices[g][0]>indices[g][1] else '+C'
            csum = makeGate(csum)
            U_AB_tot = np.dot(U_AB_loc, csum)
            gates.append(U_AB_tot)
        Tcount = Tgate_prob
    else:
        char = ['1', 'H', 'K', 'T']
        prob_list = [(1-Tgate_prob)/3]*(len(char)-1) + [Tgate_prob]
        Tcount = 0
        for g in range(circuit_length):
            U1qA = nr.choice(char, p=prob_list)
            U1qB = nr.choice(char, p=prob_list)
            Tcount +=(U1qA=='T')+(U1qB=='T')
            U1qA = makeGate(U1qA)
            U1qB = makeGate(U1qB)
            U_AB_loc = np.kron(U1qA, U1qB)
            csum = 'C+' if indices[g][0]>indices[g][1] else '+C'
            csum = makeGate(csum)
            U_AB_tot = np.dot(U_AB_loc, csum)
            gates.append(U_AB_tot)

    # Measurements
    if type(given_measurement)==int:
        char = ['0']
        prob = [1/len(char)]*len(char)

        meas = ['/']*qudit_num
        for i in range(given_measurement):
            meas[i] = nr.choice(char, p=prob)

        given_measurement = ''
        for m in meas:
            given_measurement += m
    else:
        if len(given_measurement)!=qudit_num:
            raise Exception('Number of qudits is %d'%(qudit_num))
    measurements = []
    for m in given_measurement:
        measurements.append(makeMeas(m))

    circuit = {'state_list': states, 'gate_list': gates,
               'index_list': indices, 'meas_list': measurements}

    return circuit, Tcount

def get_index_list(circuit_length, qudit_num, method='r'):
    ''' Creates index_list for circuit of given circuit_length, qudit_num
        with given method ('r': random, 'c':canonical)
    '''
    gate_qudit_index_list = []
    if method=='r':
        for gate_index in range(circuit_length):
            rng = nr.default_rng()
            gate_qudit_index = rng.choice(qudit_num, size=2, replace=False)
            gate_qudit_index_list.append(list(gate_qudit_index))

    elif method=='c':
        qudit_index = 0
        for gate_index in range(circuit_length):
            gate_qudit_index_list.append([qudit_index, qudit_index+1])
            qudit_index += 2
            if qudit_index == qudit_num and qudit_num%2 == 0:
                qudit_index = 1
            elif qudit_index == qudit_num-1 and qudit_num%2 == 0:
                qudit_index = 0
            elif qudit_index == qudit_num and qudit_num%2 == 1:
                qudit_index = 0
            elif qudit_index == qudit_num-1 and qudit_num%2 == 1:
                qudit_index = 1
    return gate_qudit_index_list

def random_connected_circuit_2q3q(qudit_num, circuit_length, Tgate_prob=1/3,
                                  prob_2q=1/2, given_state=None,
                                  given_measurement=1):
    ''' Inputs:
        qudit_num         - int
        circuit_length    - int
        Tgate_prob        - float
        prob_2q           - float (the probability of 2-qubit gates;
                                   others are 3-qubit gates.)
        given_state       - None or 0 (all zeros) or string
        given_measurement - string or int (number of measurement modes)

        Output:
        circuit = {'state_list': states, 'gate_list': gates,
                   'index_list': indices, 'meas_list': measurements}

        circuit is fully connected, i.e. there are no disentangled wires.
    '''
    Toffoli_matrix = np.array([[1., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 1., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 1., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 1., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 1., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 1., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 1.],
                               [0., 0., 0., 0., 0., 0., 1., 0.]])

    # States
    if given_state is None:
        char = ['0', '1']
        prob = [1/len(char)]*len(char)

        given_state = ''
        for i in range(qudit_num):
            given_state += nr.choice(char, p=prob)
    elif given_state==0:
        given_state = '0'*qudit_num
    else:
        if len(given_state)!=qudit_num:
            raise Exception('Number of qudits must be %d'%(qudit_num))
    states = []
    for s in given_state:
        states.append(makeState(s))

    # Indices
    indices = get_index_list_2q3q(circuit_length, qudit_num, prob_2q)

    # Gates
    char = ['1', 'H', 'K', 'T']
    prob_list = [(1-Tgate_prob)/3]*(len(char)-1) + [Tgate_prob]
    gates = []
    Tcount = 0
    toffoli_count = 0
    for i in range(len(indices)):
        if len(indices[i])==2:
            U1qA = nr.choice(char, p=prob_list)
            U1qB = nr.choice(char, p=prob_list)
            Tcount +=(U1qA=='T')+(U1qB=='T')
            U1qA = makeGate(U1qA)
            U1qB = makeGate(U1qB)
            U_AB_loc = np.kron(U1qA, U1qB)
            csum = 'C+'
            csum = makeGate(csum)
            U_tot = np.dot(U_AB_loc, csum)
        elif len(indices[i])==3:
            U1qA = nr.choice(char, p=prob_list)
            U1qB = nr.choice(char, p=prob_list)
            U1qC = nr.choice(char, p=prob_list)
            Tcount +=(U1qA=='T')+(U1qB=='T')+(U1qC=='T')
            U1qA = makeGate(U1qA)
            U1qB = makeGate(U1qB)
            U1qC = makeGate(U1qC)
            U_ABC_loc = np.kron(np.kron(U1qA, U1qB), U1qC)
            U_tot = np.dot(U_ABC_loc, Toffoli_matrix)
            toffoli_count += 1
        gates.append(U_tot)

    # Measurements
    if type(given_measurement)==int:
        char = ['0']
        prob = [1/len(char)]*len(char)

        meas = ['/']*qudit_num
        for i in range(given_measurement):
            meas[i] = nr.choice(char, p=prob)

        given_measurement = ''
        for m in meas:
            given_measurement += m
    else:
        if len(given_measurement)!=qudit_num:
            raise Exception('Number of qudits is %d'%(qudit_num))
    measurements = []
    for m in given_measurement:
        measurements.append(makeMeas(m))

    circuit = {'state_list': states, 'gate_list': gates,
               'index_list': indices, 'meas_list': measurements}

    return circuit, Tcount, toffoli_count

def get_index_list_2q3q(circuit_length, qudit_num, prob_2q=1/2):
    ''' Creates fully connected index_list for circuit of given
        circuit_length, qudit_num.
    '''
    if circuit_length<(qudit_num-1):
        raise Exception("The length of the circuit is not enought to \
                        entangle all qubits in the circuit.")

    gate_qudit_index_list = []
    qudit_masked = list(range(qudit_num))
    for i in range(circuit_length):
        # Randomly choose between 2- or 3-qubit gates
        gate_type = nr.choice([2,3],p=[prob_2q,1-prob_2q])

        rng = nr.default_rng()
        if len(qudit_masked)>2:
            gate_qudit_index = rng.choice(qudit_masked, size=gate_type,
                                          replace=False)
            gate_qudit_index_list.append(list(gate_qudit_index))
            for i in gate_qudit_index:
                qudit_masked.remove(i)
        elif len(qudit_masked)!=0:
            choice_pool = list(range(qudit_num))
            for i in qudit_masked:
                choice_pool.remove(i)
            gate_qudit_index = list(rng.choice(choice_pool,
                          size=(gate_type-len(qudit_masked)), replace=False))
            for i in qudit_masked:
                gate_qudit_index.append(i)
            gate_qudit_index_list.append(list(gate_qudit_index))
            qudit_masked.clear()
        elif len(qudit_masked)==0:
            gate_qudit_index = rng.choice(qudit_num, size=gate_type,
                                          replace=False)
            gate_qudit_index_list.append(list(gate_qudit_index))

    return gate_qudit_index_list

def compress2q_circuit(circuit):
    ''' Returns an equivalent circuit that contains only 2- or 3-qudit gates
    '''
    qudit_num = len(circuit['state_list'])
    gates, indices = circuit['gate_list'], circuit['index_list']
    gate_num = len(indices)

    if isinstance(gates[0], str):
        raise ValueError("Gates should be arrays")

    gates_compressed = []
    indices_compressed = []
    u2or3q_counts = []
    gate_masked = [0]*gate_num
    disentangled_wires = list(range(qudit_num))

    for count, gate in enumerate(gates):
        if len(indices[count])==1: continue

        gates_compressed.append(gate)
        indices_compressed.append(indices[count])
        u2or3q_counts.append(count)
        gate_masked[count] = 1
        for i in range(len(indices[count])):
            if indices[count][i] in disentangled_wires:
                disentangled_wires.remove(indices[count][i])

    for k in range(len(indices_compressed)):
        u_gate, u_index = gates_compressed[k], indices_compressed[k]
        if len(u_index)==2:
            u1q = [IDC for i in range(2)]
            for i in range(u2or3q_counts[k]):
                idx, gate = indices[i], gates[i]
                if gate_masked[i]: continue
                if idx[0] not in u_index: continue

                if idx[0]==u_index[0]:
                    u1q[0] = np.dot(gate, u1q[0])
                if idx[0]==u_index[1]:
                    u1q[1] = np.dot(gate, u1q[1])
                gate_masked[i] +=1
            gates_compressed[k] = np.dot(u_gate, np.kron(u1q[0], u1q[1]))
        elif len(u_index)==3:
            u1q = [IDC for i in range(3)]
            for i in range(u2or3q_counts[k]):
                idx, gate = indices[i], gates[i]
                if gate_masked[i]: continue
                if idx[0] not in u_index: continue

                if idx[0]==u_index[0]:
                    u1q[0] = np.dot(gate, u1q[0])
                if idx[0]==u_index[1]:
                    u1q[1] = np.dot(gate, u1q[1])
                if idx[0]==u_index[2]:
                    u1q[2] = np.dot(gate, u1q[2])
                gate_masked[i] +=1
            gates_compressed[k] = np.dot(u_gate, np.kron(np.kron(u1q[0],
                                                            u1q[1]), u1q[2]))

    for k in range(len(indices_compressed)-1, -1, -1):
        u_gate, u_index = gates_compressed[k], indices_compressed[k]
        if len(u_index)==2:
            u1q = [IDC for i in range(2)]
            for i in range(gate_num-1, u2or3q_counts[k], -1):
                idx, gate = indices[i], gates[i]
                if gate_masked[i]: continue
                if idx[0] not in u_index: continue

                if idx[0]==u_index[0]:
                    u1q[0] = np.dot(u1q[0], gate)
                if idx[0]==u_index[1]:
                    u1q[1] = np.dot(u1q[1], gate)
                gate_masked[i] +=1
            gates_compressed[k] = np.dot(np.kron(u1q[0], u1q[1]), u_gate)
        elif len(u_index)==3:
            u1q = [IDC for i in range(3)]
            for i in range(gate_num-1, u2or3q_counts[k], -1):
                idx, gate = indices[i], gates[i]
                if gate_masked[i]: continue
                if idx[0] not in u_index: continue

                if idx[0]==u_index[0]:
                    u1q[0] = np.dot(u1q[0], gate)
                if idx[0]==u_index[1]:
                    u1q[1] = np.dot(u1q[1], gate)
                if idx[0]==u_index[2]:
                    u1q[2] = np.dot(u1q[2], gate)
                gate_masked[i] +=1
            gates_compressed[k] = np.dot(np.kron(np.kron(u1q[0], u1q[1]),
                                                 u1q[2]), u_gate)

    u1qs = [IDC for i in range(len(disentangled_wires))]
    for i in range(len(gates)):
        if indices[i] not in disentangled_wires: continue

        idx = disentangled_wires.index(indices[i])
        u1qs[idx] = np.dot(makeGate(gates[i]), u1qs[idx])

        gate_masked[i] +=1

    for i, wire in enumerate(disentangled_wires):
        gates_compressed.append(u1qs[i])
        indices_compressed.append([wire])

    duplicates = []
    for i in range(len(gates_compressed)-1):
        gate, gate_next = gates_compressed[i], gates_compressed[i+1]
        idx, idx_next = indices_compressed[i], indices_compressed[i+1]
        if set(idx)!=set(idx_next): continue

        if idx==idx_next:
            gate_next = np.dot(gate_next, gate)
        else:
            gate_next = np.dot(gate_next, aligned_gate(gate, idx, idx_next))
            # gate_next = np.dot(gate_next, np.dot(SWAP, np.dot(gate, SWAP)))
        gates_compressed[i+1] = gate_next
        duplicates.append(i)

    for i in duplicates[::-1]:
        gates_compressed.pop(i)
        indices_compressed.pop(i)
    #     gates_compressed[i] = None
    #     indices_compressed[i] = None
    # gates_compressed = [val for val in gates_compressed if val is not None]
    # indices_compressed=[val for val in indices_compressed if val is not None]

    circuit_compressed = {'state_list': circuit['state_list'],
                          'gate_list': gates_compressed,
                          'index_list': indices_compressed,
                          'meas_list': circuit['meas_list']}
    return circuit_compressed

def compress3q_circuit(circuit):
    ''' Returns an equivalent circuit that contains only 3-qudit gates
    '''
    qudit_num = len(circuit['state_list'])
    gates, indices = circuit['gate_list'], circuit['index_list']
    gate_num = len(indices)

    if isinstance(gates[0], str): raise ValueError("Gates should be arrays")
    if qudit_num<3: raise Exception("qudit_num must be at least 3")

    gates_mask = [gates[i] for i in range(gate_num)]
    indices_mask = [indices[i] for i in range(gate_num)]
    gates_compressed = []
    indices_compressed = []

    while len(gates_mask)!=0:
        gate, idx = gates_mask[0], indices_mask[0]
        grouped_gates = [gate]
        grouped_idx = [idx]
        idx_set = [idx[i] for i in range(len(idx))]
        to_be_removed = [0]

        # If it is the last gate, append an identity matrix to make a 3-qubit gate
        # And end the while loop
        if len(gates_mask)==1:
            if len(idx)==2: # If the last gate is a 2-qubit gate
                # Get a random wire
                choice_pool = list(set(range(qudit_num)).difference(idx_set))
                rng = nr.default_rng()
                idx_set.append(rng.choice(choice_pool, size=1, replace=False)[0])

                u3q = makeGate('111')
                gates_compressed.append(np.dot(aligned_gate(gate, idx, idx_set), u3q))
                indices_compressed.append(idx_set)

                for i in to_be_removed: # Remove the gate from the gate_list
                    gates_mask.pop(i)
                    indices_mask.pop(i)
            elif len(idx)==3:# If the last gate is a 3-qubit gate (convenient!)
                gates_compressed.append(gate)
                indices_compressed.append(idx)
            break

        # Finding the appropriate 3 indices when the gate is a 2-qubit gate
        if len(idx)==2: # Only when the gate is a 2-qubit gate
            for s in range(1,len(gates_mask)): # Check all the gates coming next
                check_idx = indices_mask[s]
                if set(idx)==set(check_idx):
                    continue
                elif len(check_idx)==2:
                    if check_idx[0] in idx:
                        # If we found another connected wire, check whether there is any preceding gate
                        for d in range(1, s):
                            if check_idx[1] in indices_mask[d]:
                                break
                            if d==(s-1):
                                idx_set.append(check_idx[1])
                        if s==1: idx_set.append(check_idx[1]) # For the case when the code does not go into the above for-loop because s=1.
                    elif check_idx[1] in idx:
                        # If we found another connected wire, check whether there is any preceding gate
                        for d in range(1, s):
                            if check_idx[0] in indices_mask[d]: break
                            if d==(s-1): idx_set.append(check_idx[0])
                        if s==1: idx_set.append(check_idx[0]) # For the case when the code does not go into the above for-loop because s=1.
                elif len(check_idx)==3:
                    if idx[0] in check_idx and idx[1] in check_idx:
                        # If we found an overlapped 3-qubit gate, check whether there is any preceding gate
                        for d in range(1,s):
                            if check_idx[0] in indices_mask[d] or check_idx[1] in indices_mask[d] or check_idx[2] in indices_mask[d]:
                                break
                            if d==(s-1):
                                idx_set.append(list(set(check_idx).difference(idx))[0])
                        if s==1:
                                idx_set.append(list(set(check_idx).difference(idx))[0])
                if len(idx_set)==3: break # End the loop after finding an appropriate 3 indices

        # If there is no appropriate connected wire, just append an identity matrix to make a 3-qubit gate
        # And continue to the next while step
        if len(idx_set)==2:
            # print("Came here!")
            choice_pool = list(set(range(qudit_num)).difference(idx_set))
            # print(idx_set)
            # print(choice_pool)
            rng = nr.default_rng()
            idx_set.append(rng.choice(choice_pool, size=1, replace=False)[0])
            u3q = makeGate('111')
            gates_compressed.append(np.dot(aligned_gate(gate, idx, idx_set), u3q))
            indices_compressed.append(idx_set)
            for i in to_be_removed: # Remove the gate from the gate_list
                gates_mask.pop(i)
                indices_mask.pop(i)
            continue

        # Group the gates: group the gates which can be combined together for the obtained idx_set
        for s in range(1, len(gates_mask)): # Check for all gates coming next
            check_idx = indices_mask[s]
            if len(check_idx)==2:
                if check_idx[0] in idx_set and check_idx[1] in idx_set: # If a gate is applied to idx_set
                    for d in range(1, s): # Check whether there is any preceding gate
                        if d in to_be_removed:
                            if d!=(s-1): continue
                            else:
                                grouped_gates.append(gates_mask[s])
                                grouped_idx.append(check_idx)
                                to_be_removed.append(s)
                                break
                        elif check_idx[0] in indices_mask[d] or check_idx[1] in indices_mask[d]:
                            break
                        if d==(s-1): # If there is no preceding gate then put it into the group
                            grouped_gates.append(gates_mask[s])
                            grouped_idx.append(check_idx)
                            to_be_removed.append(s)
                            break
                    if s==1: # If it is the next gate, put it into the group
                        grouped_gates.append(gates_mask[s])
                        grouped_idx.append(check_idx)
                        to_be_removed.append(s)
                else: continue
            elif len(check_idx)==3: # Do the same for the case of 3-qubit gates
                if set(check_idx)==set(idx_set):
                    for d in range(1, s):
                        if d in to_be_removed:
                            if d!=(s-1): continue
                            else:
                                grouped_gates.append(gates_mask[s])
                                grouped_idx.append(check_idx)
                                to_be_removed.append(s)
                                break
                        elif check_idx[0] in indices_mask[d] or check_idx[1] in indices_mask[d] or check_idx[2] in indices_mask[d]:
                            break
                        if d==(s-1):
                            grouped_gates.append(gates_mask[s])
                            grouped_idx.append(check_idx)
                            to_be_removed.append(s)
                            break
                    if s==1:
                        grouped_gates.append(gates_mask[s])
                        grouped_idx.append(check_idx)
                        to_be_removed.append(s)
            else: raise Exception("A gate must be a 2- or 3-qubit gates.")

        # Add the grouped gates into the compressed sequence
        u3q_combined = makeGate('111')
        for k in range(len(grouped_gates)):
            u3q_combined = np.dot(aligned_gate(grouped_gates[k], grouped_idx[k], idx_set), u3q_combined)
        gates_compressed.append(u3q_combined)
        indices_compressed.append(idx_set)

        # Remove the compressed gates from the gate sequence
        for k in to_be_removed[::-1]:
            # print(k, to_be_removed[::-1])
            # print(len(gates_mask), len(indices_mask))
            gates_mask.pop(k)
            indices_mask.pop(k)

    circuit_compressed = {'state_list': circuit['state_list'],
                          'gate_list': gates_compressed,
                          'index_list': indices_compressed,
                          'meas_list': circuit['meas_list']}
    return circuit_compressed

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

    new_target_index = [index_dup.index(target_index[i]) for i in range(len(target_index))]
    P_matrix = permutation_matrix(list(range(len(index_dup))), new_target_index, [2]*len(target_index))
    gate = np.dot(P_matrix, np.dot(gate, P_matrix.T))

    return gate

def basis(dim):
    basis = []
    for i in range(dim):
        vec = np.zeros(dim)
        vec[i] = 1
        basis.append(vec)
    return basis

def tensor(array_list):
    return fc.reduce(lambda x,y : np.kron(x,y), array_list)

def indices_list(dimension_tuple):
    number_subsystems = len(dimension_tuple)
    if number_subsystems != 0:
        subindices = [range(dim) for dim in dimension_tuple]
        return np.array(np.meshgrid(*subindices)).T.reshape(-1,number_subsystems)
    else:
        return np.array([])

def permutation_matrix(initial_order,final_order,dimension_subsystems):

    subsystems_number = len(initial_order)
    # print(initial_order, final_order)
    # check that order and dimension tuples have the same length
    if subsystems_number != len(final_order) or subsystems_number != len(dimension_subsystems):
        raise RuntimeError("The length of the tuples passed to the function needs to be the same")

    # Create the list of basis for each subsystem
    initial_basis_list = list(map(lambda dim : basis(dim) , dimension_subsystems))

    # Create all possible indices for the global basis
    indices = indices_list(dimension_subsystems)
    # Create permutation matrix P
    total_dim = np.product(np.array(dimension_subsystems))
    P_matrix = np.zeros((total_dim,total_dim))

    for index in indices:
        initial_vector_list = [initial_basis_list[n][i] for n,i in enumerate(index)]
        final_vector_list = [initial_vector_list[i] for i in final_order]
        initial_vector = tensor(initial_vector_list)
        final_vector = tensor(final_vector_list)

        P_matrix += np.outer(final_vector,initial_vector)

    return P_matrix

def show_connectivity(circuit):
    ''' Prints a visual circuit representation.
    '''
    qudit_num = len(circuit['state_list'])
    indices = circuit['index_list']
    meas = circuit['meas_list']

    circ_repr = [['> '] for i in range(qudit_num)]
    for i in range(len(indices)):
        idx = indices[i]
        if len(idx)==1: continue
        elif len(idx) in [2,3]:
            idle_wires = np.delete(np.arange(qudit_num), idx)
            for j in idle_wires: circ_repr[j].extend(['-'])
            if len(idx)==3:
                circ_repr[idx[-3]].append('O')
                circ_repr[idx[-2]].append('|')
                circ_repr[idx[-1]].append('+')
            if len(idx)==2:
                circ_repr[idx[-2]].append('c')
                circ_repr[idx[-1]].append('z')
        else: raise Exception('show_connectivity not implemented for m>3')
    for i in range(qudit_num):
        m = '/' if np.allclose(meas[i], IDC) else 'D'
        circ_repr[i].append(' '+m)


    circ_repr = [''.join(wire) for wire in circ_repr]
    for wire in circ_repr:
        print(wire)
    # return circ_repr


def string_to_circuit(circuit_string):
    ''' Converts symbolic circuit to:

        circuit = {'state_list': states, 'gate_list': gates,
                   'index_list': indices, 'meas_list': measurements}
    '''
    state_string_list = circuit_string[0]
    gate_string_list  = circuit_string[1]
    meas_string_list  = circuit_string[2]

    state_list = []
    for state_string in state_string_list:
        state_list.append(makeState(state_string))

    gate_list, index_list = [], []
    for gate in gate_string_list:
        index_list.append(gate[0])
        gate_list.append(makeGate(gate[1]))

    meas_list = []
    for meas_string in meas_string_list:
        # if meas_string=='/': continue
        meas_list.append(makeMeas(meas_string))

    circuit = {'state_list': state_list, 'gate_list': gate_list,
               'index_list': index_list, 'meas_list': meas_list}

    return circuit

def solve_qubit_circuit(circuit):
    state = reduce(np.kron, circuit['state_list'])
    D, N = 2, int(np.log2(state.shape[0]))

    for i in range(len(circuit['index_list'])):
        idx, gate = circuit['index_list'][i], circuit['gate_list'][i]

        order = np.concatenate((idx, np.delete(np.arange(N),idx)))
        perm  = np.argsort(np.concatenate((order, order+N)))
        gate  = np.kron(gate, np.eye(2**(N-len(idx))))
        gate  = gate.reshape(N*2*(D,)).transpose(perm).reshape(2*(D**N,))

        state = evolve(state, gate)
        # print('\n---------------\n', 'gate:\n', gate,  '\n---------------')
        # print('\n---------------\n','state:\n', state, '\n---------------')
    meas = reduce(np.kron, circuit['meas_list'])
    prob = np.trace(np.dot(meas, state))
    if not np.isclose(prob.imag, 0):
        print(prob)
        raise Exception('Probability is not real')
    return prob.real

def test_solver():
    truths = []
    tests = []

    tests.append( string_to_circuit( ['00', [
                    [[0],'H'],
                    [[0],'K'],
                    [[1],'K']
                    ], '10'] ))
    truths.append(1/2)


    tests.append( string_to_circuit( ['00', [
                    [[0], 'H'],
                    [[1,0], 'C+'],
                    [[0,1], 'C+']
                    ], '0/'] ))
    truths.append(1/2)

    q = 2
    tests.append( string_to_circuit( ['0000', [
                    [[q], 'H'],
                    [[q], 'T'],
                    [[q], 'H'],
                    [[1], 'K'],
                    [[1,q], 'C+'],
                    [[q,1], 'C+'],
                    [[q], 'H']
                    ], '0100'] ))
    truths.append(0.07322330470336305)

    tests.append( string_to_circuit( ['110', [
                    [[0,1,2],'A'],
                    # [[2], 'H']
                    ], '111'] ))
    truths.append(1)

    tests.append( string_to_circuit( ['011', [
                    [[1,2,0],'A'],
                    # [[2], 'H']
                    ], '111'] ))
    truths.append(1)

    tests.append( string_to_circuit( ['011', [
                    [[2,1,0],'A'],
                    # [[2], 'H']
                    ], '111'] ))
    truths.append(1)

    checks = [solve_qubit_circuit(tests[i])
              for i in range(len(tests))]
    all_check = np.isclose(checks, truths)
    if np.all(all_check):
        print(True)
        return(all_check)
    else:
        print(all_check, '\n', checks)
        return(all_check)





