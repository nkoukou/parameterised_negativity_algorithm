from functools import reduce
from itertools import permutations
import numpy as np
import numpy.random as nr
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
    char = ['1', 'H', 'K', 'T']
    prob_list = [(1-Tgate_prob)/3]*(len(char)-1) + [Tgate_prob]
    gates = []
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

def compress2q_circuit(circuit):
    ''' Returns an equivalent circuit that contains only 2-qudit gates
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
            gates_compressed[k] = np.dot(u_gate, np.kron(np.kron(u1q[0], u1q[1]), u1q[2]))

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
            gates_compressed[k] = np.dot(np.kron(np.kron(u1q[0], u1q[1]), u1q[2]), u_gate)

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
            gate_next = np.dot(gate_next, np.dot(SWAP, np.dot(gate, SWAP)))
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
                # Find the nearest wire for the identity matrix
                if idx_set[1]==(qudit_num-1): idx_set.append(idx_set[0]-1)
                else: idx_set.append(idx_set[1]+1)

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
                elif check_idx[0] in idx:
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

                if len(idx_set)==3: break # End the loop after finding an appropriate 3 indices

        # If there is no appropriate connected wire, just append an identity matrix to make a 3-qubit gate
        # And continue to the next while step
        if len(idx_set)==2:
            # print("Came here!")
            if idx_set[1]==(qudit_num-1): idx_set.append(idx_set[0]-1)
            else: idx_set.append(idx_set[1]+1)
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
                            continue
                        elif check_idx[0] in indices_mask[d] or check_idx[1] in indices_mask[d]:
                            break
                        if d==(s-1): # If there is no preceding gate then put it into the group
                            grouped_gates.append(gates_mask[s])
                            grouped_idx.append(check_idx)
                            to_be_removed.append(s)
                    if s==1: # If it is the next gate, put it into the group
                        grouped_gates.append(gates_mask[s])
                        grouped_idx.append(check_idx)
                        to_be_removed.append(s)
                else: continue
            elif len(check_idx)==3: # Do the same for the case of 3-qubit gates
                if set(check_idx)==set(idx_set):
                    for d in range(1, s):
                        if d in to_be_removed: continue
                        elif check_idx[0] in indices_mask[d] or check_idx[1] in indices_mask[d] or check_idx[2] in indices_mask[d]:
                            break
                        if d==(s-1):
                            grouped_gates.append(gates_mask[s])
                            grouped_idx.append(check_idx)
                            to_be_removed.append(s)
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
        index        - list of int (len = 1 or 2)
        target_index - list of int (len = 3)
    '''
    if not set(index).issubset(target_index):
        raise Exception('Indices do not match')
    if len(index)==1:
        temp = target_index.index(index[0])
        gate = np.kron(np.eye(2**temp), np.kron(gate, np.eye(2**(2-temp))))
    if len(index)==2:
        if target_index.index(index[0])>target_index.index(index[1]):
            gate = gate.reshape((2,2,2,2)).swapaxes(0,1).swapaxes(2,3
                                ).reshape((4,4))
        temp = target_index.index(list(set(target_index
                                           ).difference(index))[0])
        if temp==0:
            gate = np.kron(IDC, gate)
        if temp==1:
            gate = np.kron(gate, IDC)
            gate = gate.reshape((2,2,2,2,2,2)).swapaxes(0,3).swapaxes(1,2
                                 ).swapaxes(4,5).reshape((8,8))
        if temp==2:
            gate = np.kron(gate, IDC)
    return gate

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
    ''' Solves small qubit circuits.
    '''
    dim = 2
    state = reduce(np.kron, circuit['state_list'])
    qudit_num = int(np.log2(state.shape[0]))
    # print('\n---------------\n','state:\n', state, '\n---------------')

    for i in range(len(circuit['index_list'])):
        idx, gate = circuit['index_list'][i], circuit['gate_list'][i]
        # print('\n---------------\n','gate:\n', gate, '\n---------------')

        order = np.array([sorted(idx).index(i) for i in idx])
        order_rhs = order+len(idx)
        perm = np.concatenate((order, order_rhs))
        if len(order)==3:
            if np.all(order==np.array([1,2,0])):
                perm = np.array([0, 1, 3, 2, 5, 4])
            if np.all(order==np.array([2,0,1])):
                perm = np.array([0, 1, 3, 2, 5, 4])
        gate = gate.reshape(len(idx)*2*(dim,)).transpose(perm
                            ).reshape(2*(dim**len(idx),))
        id_count = np.cumsum(np.diff(np.sort(np.append(idx,
                                               (-1,qudit_num))))-1)
        perm = np.arange(len(idx),len(idx)+id_count[0])
        for i in range(len(idx)):
            next_idc = np.arange(len(idx)+id_count[i],
                                 len(idx)+id_count[i+1])
            perm = np.concatenate((perm,np.insert(next_idc, 0, i)))
        perm = np.concatenate((perm,perm+qudit_num))
        gate = np.kron(gate, np.eye(2**(qudit_num-len(idx))))
        gate = gate.reshape(qudit_num*2*(dim,)).transpose(perm
                                  ).reshape(2*(dim**qudit_num,))

        # print(idx, order)
        # print(id_count)
        # print(perm)
        # print('\n---------------\n','gate:\n', gate, '\n---------------')

        state = evolve(state, gate)
        # print('\n---------------\n','state:\n', state, '\n---------------')

    meas = reduce(np.kron, circuit['meas_list'])
    prob = np.trace(np.dot(meas, state))
    if not np.isclose(prob.imag, 0):
        print(prob)
        raise Exception('Probability is not real')
    return prob.real

def find_perms():
    D, N = 2, 3
    gate = np.arange(1,(D*D)**N+1).reshape((D**N,D**N))
    sub_states = [np.array([1,2]), np.array([3,5]), np.array([7,11])]
    indices = [np.array(idx) for idx in list(permutations(range(N)))]

    state0 = reduce(np.kron, sub_states)

    all_perms = []
    for idx in indices:
        state = reduce(np.kron, [sub_states[idx[i]] for i in range(N)])

        valid_perms = []
        for perm in permutations(range(2*N)):
            temp = gate.reshape((D,D,)*N).transpose(perm).reshape((D**N,D**N))
            target = np.sort(evolve(state0, gate))
            if np.allclose(np.sort(evolve(state, temp)), target):
                valid_perms.append(list(perm))
        all_perms.append(valid_perms)
    return indices, all_perms

def find_perms_A(last):
    D, N = 2, 3
    gate = makeGate('A')
    if last==0:
        index = [1,2,0]
        state0 = makeState('011')
    elif last==1:
        index = [2,0,1]
        state0 = makeState('101')
    elif last==2:
        index = [0,1,2]
        state0 = makeState('110')
    else: raise Exception('ERROR ON last')
    target = makeState('111')

    all_perms = []
    for perm in permutations(range(2*N)):
        if set(perm[:3]) != set(np.array(index)): continue
        temp = gate.reshape((D,D,)*N).transpose(perm).reshape((D**N,D**N))
        if np.allclose(evolve(state0, temp), target):
            all_perms.append(list(perm))
    return index, all_perms

def find_perms_test():
    all_perms = []
    N = 3
    count = 0
    pool = [[0, 1, 3, 2, 5, 4],  [0, 1, 3, 5, 2, 4],  [0, 4, 3, 2, 5, 1],
            [0, 4, 3, 5, 2, 1],  [1, 0, 4, 2, 5, 3],  [1, 0, 4, 5, 2, 3],
            [1, 3, 4, 2, 5, 0],  [1, 3, 4, 5, 2, 0],  [3, 1, 0, 2, 5, 4],
            [3, 1, 0, 5, 2, 4],  [3, 4, 0, 2, 5, 1],  [3, 4, 0, 5, 2, 1],
            [4, 0, 1, 2, 5, 3],  [4, 0, 1, 5, 2, 3],  [4, 3, 1, 2, 5, 0],
            [4, 3, 1, 5, 2, 0]]
    all_perms = []
    for perm1 in pool:
        for perm2 in pool:
            # print(perm1, perm2)
            all_checks = test_solver(perm1, perm2)
            if np.all(all_checks):
                all_perms.append(perm1+perm2)
            if count%25==0: print(count)
            count +=1
    return count, all_perms

def test_solver():
    truths = []
    tests = []

    # tests.append( string_to_circuit( ['00', [
    #                 [[0],'H'],
    #                 [[0],'K'],
    #                 [[1],'K']
    #                ], '10'] ))
    # truths.append(1/2)


    # tests.append( string_to_circuit( ['00', [
    #                 [[0], 'H'],
    #                 [[1,0], 'C+'],
    #                 [[0,1], 'C+']
    #                ], '0/'] ))
    # truths.append(1/2)

    # q = 2
    # tests.append( string_to_circuit( ['0000', [
    #                 [[q], 'H'],
    #                 [[q], 'T'],
    #                 [[q], 'H'],
    #                 [[1], 'K'],
    #                 [[1,q], 'C+'],
    #                 [[q,1], 'C+'],
    #                 [[q], 'H']
    #                ], '0100'] ))
    # truths.append(0.07322330470336305)

    gate = makeGate('11H')
    gate = np.dot(makeGate('11T'), gate)
    gate = np.dot(makeGate('11H'), gate)
    gate = np.dot(makeGate('1K1'), gate)
    gate = np.dot(makeGate('H11'), gate)
    gate = np.dot(makeGate('1C+'), gate)
    gate = np.dot(makeGate('1+C'), gate)
    tests.append( {'state_list': [makeState('0') for i in range(4)],
                    'gate_list': [gate, makeGate('H')],
                    'index_list': [[0,1,2], [2]],
                    'meas_list': [makeState('+'), makeState('1'),
                                  makeState('0'), makeState('0')]} )
    truths.append(0.07322330470336305)

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

    tests.append( string_to_circuit( ['101', [
                    [[2,0,1],'A'],
                    # [[2], 'H']
                    ], '111'] ))
    truths.append(1)

    tests.append( string_to_circuit( ['000000', [
                    [[0], 'H'],
                    [[1], 'H'],
                    [[0,2], 'C+'],
                    [[2,0,1],'A'],
                    # [[2], 'H'],
                    # [[2,4,0],'A'] # Comment-out = 1/8 ('001010')
                    ], '000000'] ))
    truths.append(1/4)

    tests.append( string_to_circuit( ['000000', [
                    [[0], 'H'],
                    [[1], 'H'],
                    [[0,2], 'C+'],
                    [[2,0,1],'A'],
                    # [[2], 'H'],
                    # [[2,4,0],'A'] # Comment-out = 1/8 ('001010')
                    ], '010000'] ))
    truths.append(1/4)

    tests.append( string_to_circuit( ['000000', [
                    [[0], 'H'],
                    [[1], 'H'],
                    [[0,2], 'C+'],
                    [[2,0,1],'A'],
                    # [[2], 'H'],
                    # [[2,4,0],'A'] # Comment-out = 1/8 ('001010')
                    ], '111000'] ))
    truths.append(1/4)

    tests.append( string_to_circuit( ['000000', [
                    [[0], 'H'],
                    [[1], 'H'],
                    [[0,2], 'C+'],
                    [[2,0,1],'A'],
                    # [[2], 'H'],
                    # [[2,4,0],'A'] # Comment-out = 1/8 ('001010')
                    ], '101000'] ))
    truths.append(1/4)

    tests.append( string_to_circuit( ['000000', [
                    [[0], 'H'],
                    [[1], 'H'],
                    [[0,2], 'C+'],
                    [[2,0,1],'A'],
                    [[2], 'H'],
                    [[4,2,0],'A']
                    ], '0++000'] ))
    truths.append(1/2)

    # tests.append( string_to_circuit( ['000000', [
    #                 [[0], 'H'],
    #                 [[1], 'H'],
    #                 [[0,2], 'C+'],
    #                 [[2,0,1],'A'],
    #                 [[2], 'H'],
    #                 [[2,4,0],'A'] # Comment-out = 1/8 ('001010')
    #                 ], '1+0010'] ))
    # truths.append(1/4)

    # tests.append( string_to_circuit( ['000000', [
    #                 [[0], 'H'],
    #                 [[1], 'H'],
    #                 [[0,2], 'C+'],
    #                 [[2,0,1],'A'],
    #                 [[2], 'H'],
    #                 [[2,4,0],'A'] # Comment-out = 1/8 ('001010')
    #                 ], '0+1010'] ))
    # truths.append(1/4)

    checks = [solve_qubit_circuit(tests[i])
              for i in range(len(tests))]
    all_check = np.isclose(checks, truths)
    if np.all(all_check):
        print(True)
        return(all_check)
    else:
        print(all_check, '\n', checks)
        return(all_check)
# test_solver()





