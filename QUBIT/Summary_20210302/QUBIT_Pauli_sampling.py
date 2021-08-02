import time
import autograd.numpy as np
import itertools as it
import numpy.random as nr

from autograd import(grad)
from scipy.optimize import(Bounds, minimize)
from scipy.optimize import(basinhopping)

sigma_x = np.array([[0,1],[1,0]],dtype='complex')
sigma_y = np.array([[0,-1.j],[1.j,0]],dtype='complex')
sigma_z = np.array([[1,0],[0,-1]],dtype='complex')
Pauli_list = [np.eye(2),sigma_x,sigma_y,sigma_z]

def Rot_matrix(a,b,c):
    ca = np.cos(a)
    sa = np.sin(a)
    cb = np.cos(b)
    sb = np.sin(b)
    cc = np.cos(c)
    sc = np.sin(c)

    R_out = np.array(
    [[1,     0,              0,              0],
     [0, ca*cb, ca*sb*sc-sa*cc, ca*sb*cc+sa*sc],
     [0, sa*cb, sa*sb*sc+ca*cc, sa*sb*cc-ca*sc],
     [0,   -sb, cb*sc,          cb*cc         ]
    ]
    )
    return R_out

def evol_U(U,A):
    return np.dot(np.dot(U,A),np.conjugate(U).T)


### Pauli-Transformation Matrix ###
def T_matrix_1q(U1q):
    T_out = np.zeros((4,4))
    for w in it.product(range(4),repeat=2):
        [w_in, w_out] = w
        T_out[w_out,w_in] = np.real(1./2.*np.trace(np.dot(evol_U(U1q,Pauli_list[w_in]),Pauli_list[w_out])))
    return T_out

def T_matrix_2q(U2q):
    T_out = np.zeros((16,16))
    for w in it.product(range(4),repeat=4):
        [w1_in, w2_in, w1_out, w2_out] = w
        Pauli_in = np.kron(Pauli_list[w1_in],Pauli_list[w2_in])
        Pauli_out = np.kron(Pauli_list[w1_out],Pauli_list[w2_out])
        T_out[4*w1_out+w2_out][4*w1_in+w2_in] = np.real(1./4.*np.trace(np.dot(evol_U(U2q,Pauli_in),Pauli_out)))
    return T_out

def T_matrix_3q(U3q):
    T_out = np.zeros((64,64))
    for w in it.product(range(4),repeat=6):
        [w1_in, w2_in, w3_in, w1_out, w2_out, w3_out] = w
        Pauli_in = np.kron(np.kron(Pauli_list[w1_in],Pauli_list[w2_in]),Pauli_list[w3_in])
        Pauli_out = np.kron(np.kron(Pauli_list[w1_out],Pauli_list[w2_out]),Pauli_list[w3_out])
        T_out[16*w1_out+4*w2_out+w3_out][16*w1_in+4*w2_in+w3_in] = np.real(1./8.*np.trace(np.dot(evol_U(U3q,Pauli_in),Pauli_out)))
    return T_out

def T_matrix_4q(U4q):
    T_out = np.zeros((256,256))
    for w in it.product(range(4),repeat=8):
        [w1_in, w2_in, w3_in, w4_in, w1_out, w2_out, w3_out, w4_out] = w
        Pauli_in = np.kron(np.kron(np.kron(Pauli_list[w1_in],Pauli_list[w2_in]),Pauli_list[w3_in]),Pauli_list[w4_in])
        Pauli_out = np.kron(np.kron(np.kron(Pauli_list[w1_out],Pauli_list[w2_out]),Pauli_list[w3_out]),Pauli_list[w4_out])
        T_out[64*w1_out+16*w2_out+4*w3_out+w4_out][64*w1_in+16*w2_in+4*w3_in+w4_in] = np.real(1./16.*np.trace(np.dot(evol_U(U4q,Pauli_in),Pauli_out)))
    return T_out

### Rotated-Pauli-Transformation Matrix###
def RTR_matrix_1q(T1q,x_in,x_out):
    R_in = Rot_matrix(x_in[0],x_in[1],x_in[2])
    R_out = Rot_matrix(x_out[0],x_out[1],x_out[2])
    return np.dot(np.dot(R_out,T1q),R_in.T)

def RTR_matrix_2q(T2q,x_in,x_out):
    R1_in = Rot_matrix(x_in[0],x_in[1],x_in[2])
    R2_in = Rot_matrix(x_in[3],x_in[4],x_in[5])
    R_in = np.kron(R1_in,R2_in)

    R1_out = Rot_matrix(x_out[0],x_out[1],x_out[2])
    R2_out = Rot_matrix(x_out[3],x_out[4],x_out[5])
    R_out = np.kron(R1_out,R2_out)

    return np.dot(np.dot(R_out,T2q),R_in.T)

def RTR_matrix_3q(T3q,x_in,x_out):
    R1_in = Rot_matrix(x_in[0],x_in[1],x_in[2])
    R2_in = Rot_matrix(x_in[3],x_in[4],x_in[5])
    R3_in = Rot_matrix(x_in[6],x_in[7],x_in[8])
    R_in = np.kron(np.kron(R1_in,R2_in),R3_in)

    R1_out = Rot_matrix(x_out[0],x_out[1],x_out[2])
    R2_out = Rot_matrix(x_out[3],x_out[4],x_out[5])
    R3_out = Rot_matrix(x_out[6],x_out[7],x_out[8])
    R_out = np.kron(np.kron(R1_out,R2_out),R3_out)

    return np.dot(np.dot(R_out,T3q),R_in.T)

### Negativity Calculation ###
def negativity(M):
    return np.abs(M).sum(axis=0)
def max_negativity(M):
    return np.max(negativity(M))
def get_NSP(M):
    N_list = negativity(M)
    S_matrix = np.sign(M)*N_list
    P_matrix = np.abs(M)/N_list
    return N_list,S_matrix,P_matrix
def get_NSP_sequence(gate_T_sequence):
    gate_N_sequence = []
    gate_S_sequence = []
    gate_P_sequence = []
    for gate_T in gate_T_sequence:
        N_list, S_matrix, P_matrix = get_NSP(gate_T)
        gate_N_sequence.append(N_list)
        gate_S_sequence.append(S_matrix)
        gate_P_sequence.append(P_matrix)
    return gate_N_sequence, gate_S_sequence, gate_P_sequence

### Update w ###
def update_1q(w_in,s_in,P_matrix,S_matrix):
    prob = P_matrix[:,w_in]
    w_out = nr.choice(4, p=prob)
    s_out = s_in*S_matrix[w_out,w_in]
    return w_out, s_out

def update_2q(w1_in,w2_in,s_in,P_matrix,S_matrix):
    w_in = 4*w1_in + w2_in
    prob = P_matrix[:,w_in]
    w_out = nr.choice(16, p=prob)
    w1_out = w_out//4
    w2_out = w_out%4
    s_out = s_in*S_matrix[w_out,w_in]
    return w1_out, w2_out, s_out

def update_3q(w1_in,w2_in,w3_in,s_in,P_matrix,S_matrix):
    w_in = 16*w1_in + 4*w2_in + w3_in
    prob = P_matrix[:,w_in]
    w_out = nr.choice(64, p=prob)
    w1_out = w_out//16
    w2_out = (w_out%16)//4
    w3_out = (w_out%16)%4
    s_out = s_in*S_matrix[w_out,w_in]
    return w1_out, w2_out, w3_out, s_out

def update_4q(w1_in,w2_in,w3_in,w4_in,s_in,P_matrix,S_matrix):
    w_in = 64*w1_in + 16*w2_in + 4*w3_in + w4_in
    prob = P_matrix[:,w_in]
    w_out = nr.choice(256, p=prob)
    w1_out = w_out//64
    w2_out = (w_out%64)//16
    w3_out = ((w_out%64)%16)//4
    w4_out = ((w_out%64)%16)%4
    s_out = s_in*S_matrix[w_out,w_in]
    return w1_out, w2_out, w3_out, w4_out, s_out

### Calculate Pauli-Transform matrix from circuit###
def get_prob_Pauli_2q(circuit):
    state_T_list = get_state_T_list(circuit['state_list'])
    gate_T_2q_list = get_gate_T_list_2q(circuit['gate_list'])
    index_list = circuit['index_list']
    meas_T_list, neg_meas = get_meas_T_list(circuit['meas_list'])
    gate_N_2q_list, gate_S_2q_list, gate_P_2q_list = get_NSP_sequence(gate_T_2q_list)

    neg_gate = 1
    for gate_T_2q in gate_T_2q_list:
        neg_gate *= max_negativity(gate_T_2q)
    print('Log_2 negativity (gate):',np.log2(neg_gate))
    print('Log_2 negativity (meas):',np.log2(neg_meas))
    print('Log_2 negativity:',np.log2(neg_gate) + np.log2(neg_meas))


    prob_Pauli_output = {
        'state_T_list': state_T_list, 'gate_T_list': gate_T_2q_list, 'gate_P_list': gate_P_2q_list,
        'gate_S_list': gate_S_2q_list, 'gate_N_list': gate_N_2q_list, 'index_list': index_list, 'meas_T_list': meas_T_list,
        'neg_gate': neg_gate
        }
    return prob_Pauli_output

def get_prob_Pauli_3q(circuit):
    state_T_list = get_state_T_list(circuit['state_list'])
    gate_T_3q_list = get_gate_T_list_3q(circuit['gate_list'])
    index_list = circuit['index_list']
    meas_T_list, neg_meas = get_meas_T_list(circuit['meas_list'])
    gate_N_3q_list, gate_S_3q_list, gate_P_3q_list = get_NSP_sequence(gate_T_3q_list)

    neg_gate = 1
    for gate_T_3q in gate_T_3q_list:
        neg_gate *= max_negativity(gate_T_3q)
    print('Log_2 negativity (gate):',np.log2(neg_gate))
    print('Log_2 negativity (meas):',np.log2(neg_meas))
    print('Log_2 negativity:',np.log2(neg_gate) + np.log(neg_meas))

    prob_Pauli_output = {
        'state_T_list': state_T_list, 'gate_T_list': gate_T_3q_list, 'gate_P_list': gate_P_3q_list,
        'gate_S_list': gate_S_3q_list, 'gate_N_list': gate_N_3q_list, 'index_list': index_list, 'meas_T_list': meas_T_list,
        'neg_gate': neg_gate
        }
    return prob_Pauli_output

def get_prob_Pauli_4q(circuit):
    state_T_list = get_state_T_list(circuit['state_list'])
    gate_T_4q_list = get_gate_T_list_4q(circuit['gate_list'])
    index_list = circuit['index_list']
    meas_T_list, neg_meas = get_meas_T_list(circuit['meas_list'])
    gate_N_4q_list, gate_S_4q_list, gate_P_4q_list = get_NSP_sequence(gate_T_4q_list)

    neg_gate = 1
    for gate_T_4q in gate_T_4q_list:
        neg_gate *= max_negativity(gate_T_4q)
    print('Log_2 negativity (gate):',np.log2(neg_gate))
    print('Log_2 negativity (meas):',np.log2(neg_meas))
    print('Log_2 negativity:',np.log2(neg_gate) + np.log(neg_meas))

    prob_Pauli_output = {
        'state_T_list': state_T_list, 'gate_T_list': gate_T_4q_list, 'gate_P_list': gate_P_4q_list,
        'gate_S_list': gate_S_4q_list, 'gate_N_list': gate_N_4q_list, 'index_list': index_list, 'meas_T_list': meas_T_list,
        'neg_gate': neg_gate
        }
    return prob_Pauli_output

def get_state_T_list(state_list):
    T_list = []
    for state in state_list:
        T = []
        for index in range(4):
            T.append(np.real(np.trace(np.dot(state,Pauli_list[index])))/2.)
        T_list.append(T)
    return T_list

def get_gate_T_list_2q(gate_list):
    T_list = []
    for gate in gate_list:
        T_list.append(T_matrix_2q(gate))
    return T_list

def get_gate_T_list_3q(gate_list):
    T_list = []
    for gate in gate_list:
        T_list.append(T_matrix_3q(gate))
    return T_list

def get_gate_T_list_4q(gate_list):
    T_list = []
    for gate in gate_list:
        T_list.append(T_matrix_4q(gate))
    return T_list

def get_meas_T_list(meas_list):
    T_list = []
    neg_meas = 1
    for meas in meas_list:
        T = []
        for index in range(4):
            T.append(np.real(np.trace(np.dot(meas,Pauli_list[index]))))
        T_list.append(T)
        neg_meas *= np.max(np.abs(T))
    return T_list, neg_meas

def get_meas_RT_list(meas_T_list,x_list):
    RT_list = []
    meas_index = 0
    neg_meas = 1
    for meas_T in meas_T_list:
        R = Rot_matrix(x_list[meas_index][0],x_list[meas_index][1],x_list[meas_index][2])
        RT = np.dot(meas_T,R.T)
        RT_list.append(RT)
        neg_meas *= np.max(np.abs(RT))
        meas_index += 1
    return RT_list, neg_meas

### Sample circuit ###
def sample_circuit_2q(prob_Pauli,sample_size = int(1e5)):
    state_T_list = prob_Pauli['state_T_list']
    gate_T_2q_list = prob_Pauli['gate_T_list'] ### Not used
    gate_P_2q_list = prob_Pauli['gate_P_list']
    gate_S_2q_list = prob_Pauli['gate_S_list']
    gate_N_2q_list = prob_Pauli['gate_N_list'] ### Not used
    index_list = prob_Pauli['index_list']
    meas_T_list = prob_Pauli['meas_T_list']

    def sample_itr_2q():
        w_list, s = sample_state(state_T_list)

        for index in range(len(gate_P_2q_list)):
            w_index_1 = index_list[index][0]
            w_index_2 = index_list[index][1]
            w1_temp_in = w_list[w_index_1]
            w2_temp_in = w_list[w_index_2]
            w1_temp_out, w2_temp_out, s = update_2q(w1_temp_in,w2_temp_in,s,gate_P_2q_list[index],gate_S_2q_list[index])
            w_list[w_index_1] = w1_temp_out
            w_list[w_index_2] = w2_temp_out

        for index in range(len(w_list)):
            w = w_list[index]
            s *= meas_T_list[index][w]
#         print(s)
        return s

    t = time.time()
    out_list = []
    for sample in range(sample_size):
        out_list.append(sample_itr_2q())
    print('p_estimate:',np.mean(out_list),'\t(sampling time:',time.time()-t,')')

    return out_list

def sample_circuit_3q(prob_Pauli_output,sample_size = int(1e5)):
    state_T_list = prob_Pauli_output['state_T_list']
#     gate_T_3q_list = prob_Pauli_output['gate_T_list'] ### Not used
    gate_P_3q_list = prob_Pauli_output['gate_P_list']
    gate_S_3q_list = prob_Pauli_output['gate_S_list']
#     gate_N_3q_list = prob_Pauli_output['gate_N_list'] ### Not used
    index_list = prob_Pauli_output['index_list']
    meas_T_list = prob_Pauli_output['meas_T_list']

    def sample_itr_3q():
        w_list, s = sample_state(state_T_list)

        for index in range(len(gate_P_3q_list)):
            w_index_1 = index_list[index][0]
            w_index_2 = index_list[index][1]
            w_index_3 = index_list[index][2]
            w1_temp_in = w_list[w_index_1]
            w2_temp_in = w_list[w_index_2]
            w3_temp_in = w_list[w_index_3]
            w1_temp_out, w2_temp_out, w3_temp_out, s = update_3q(w1_temp_in,w2_temp_in,w3_temp_in,s,gate_P_3q_list[index],gate_S_3q_list[index])
            w_list[w_index_1] = w1_temp_out
            w_list[w_index_2] = w2_temp_out
            w_list[w_index_3] = w3_temp_out

        for index in range(len(w_list)):
            w = w_list[index]
            s *= meas_T_list[index][w]

        return s

    t = time.time()
    out_list = []
    for sample in range(sample_size):
        out_list.append(sample_itr_3q())
    print('p_estimate:',np.mean(out_list),'\t(sampling time:',time.time()-t,')')
    return out_list

def sample_circuit_4q(prob_Pauli_output,sample_size = int(1e5)):
    state_T_list = prob_Pauli_output['state_T_list']
#     gate_T_4q_list = prob_Pauli_output['gate_T_list'] ### Not used
    gate_P_4q_list = prob_Pauli_output['gate_P_list']
    gate_S_4q_list = prob_Pauli_output['gate_S_list']
#     gate_N_4q_list = prob_Pauli_output['gate_N_list'] ### Not used
    index_list = prob_Pauli_output['index_list']
    meas_T_list = prob_Pauli_output['meas_T_list']

    def sample_itr_4q():
        w_list, s = sample_state(state_T_list)

        for index in range(len(gate_P_4q_list)):
            w_index_1 = index_list[index][0]
            w_index_2 = index_list[index][1]
            w_index_3 = index_list[index][2]
            w_index_4 = index_list[index][3]
            w1_temp_in = w_list[w_index_1]
            w2_temp_in = w_list[w_index_2]
            w3_temp_in = w_list[w_index_3]
            w4_temp_in = w_list[w_index_4]
            w1_temp_out, w2_temp_out, w3_temp_out, w4_temp_out, s = update_4q(w1_temp_in,w2_temp_in,w3_temp_in,w4_temp_in,s,gate_P_4q_list[index],gate_S_4q_list[index])
            w_list[w_index_1] = w1_temp_out
            w_list[w_index_2] = w2_temp_out
            w_list[w_index_3] = w3_temp_out
            w_list[w_index_4] = w4_temp_out

        for index in range(len(w_list)):
            w = w_list[index]
            s *= meas_T_list[index][w]

        return s

    t = time.time()
    out_list = []
    for sample in range(sample_size):
        out_list.append(sample_itr_4q())
    print('p_estimate:',np.mean(out_list),'\t(sampling time:',time.time()-t,')')
    return out_list

# def sample_circuit_2q_all(prob_Pauli,sample_size = int(1e5)):
#     state_T_list = prob_Pauli['state_T_list']
#     gate_T_2q_list = prob_Pauli['gate_T_list'] ### Not used 
#     gate_P_2q_list = prob_Pauli['gate_P_list']
#     gate_S_2q_list = prob_Pauli['gate_S_list'] 
#     gate_N_2q_list = prob_Pauli['gate_N_list'] ### Not used 
#     index_list = prob_Pauli['index_list']
#     meas_T_list = prob_Pauli['meas_T_list']
    
#     def sample_itr_2q_all():
#         w_list, s = sample_state(state_T_list)

#         for index in range(len(gate_P_2q_list)):
#             w_index_1 = index_list[index][0]
#             w_index_2 = index_list[index][1]
#             w1_temp_in = w_list[w_index_1]
#             w2_temp_in = w_list[w_index_2]
#             w1_temp_out, w2_temp_out, s = update_2q(w1_temp_in,w2_temp_in,s,gate_P_2q_list[index],gate_S_2q_list[index])
#             w_list[w_index_1] = w1_temp_out
#             w_list[w_index_2] = w2_temp_out
        
#         meas_T_eye = np.array([2,0,0,0])
# #         meas_T_target = np.array([1.0, 0.0, 0.0, -1.0])
#         val_list = []
#         for index in range(len(w_list)):
#             s_out = s
#             meas_T_target = meas_T_list[index]
#             for jj in range(len(w_list)):
#                 w = w_list[jj]
#                 if jj==index:
#                     s_out *= meas_T_target[w]
#                 else:
#                     s_out *= meas_T_eye[w]
#             val_list.append(s_out)
#         return val_list

#     t = time.time()
#     out_list = []
#     for sample in range(sample_size):
#         out_list.append(sample_itr_2q_all())
    
#     print('p_estimate:',np.mean(out_list,axis=0),'\t(sampling time:',time.time()-t,')')

#     return out_list

# def sample_circuit_3q_all(prob_Pauli_output,sample_size = int(1e5)):
#     state_T_list = prob_Pauli_output['state_T_list']
# #     gate_T_3q_list = prob_Pauli_output['gate_T_list'] ### Not used 
#     gate_P_3q_list = prob_Pauli_output['gate_P_list']
#     gate_S_3q_list = prob_Pauli_output['gate_S_list'] 
# #     gate_N_3q_list = prob_Pauli_output['gate_N_list'] ### Not used 
#     index_list = prob_Pauli_output['index_list']
#     meas_T_list = prob_Pauli_output['meas_T_list']

#     def sample_itr_3q_all():
#         w_list, s = sample_state(state_T_list)

#         for index in range(len(gate_P_3q_list)):
#             w_index_1 = index_list[index][0]
#             w_index_2 = index_list[index][1]
#             w_index_3 = index_list[index][2]
#             w1_temp_in = w_list[w_index_1]
#             w2_temp_in = w_list[w_index_2]
#             w3_temp_in = w_list[w_index_3]
#             w1_temp_out, w2_temp_out, w3_temp_out, s = update_3q(w1_temp_in,w2_temp_in,w3_temp_in,s,gate_P_3q_list[index],gate_S_3q_list[index])
#             w_list[w_index_1] = w1_temp_out
#             w_list[w_index_2] = w2_temp_out
#             w_list[w_index_3] = w3_temp_out

#         meas_T_eye = np.array([2,0,0,0])
# #         meas_T_target = np.array([1.0, 0.0, 0.0, -1.0])
#         val_list = []
#         for index in range(len(w_list)):
#             s_out = s
#             meas_T_target = meas_T_list[index]
#             for jj in range(len(w_list)):
#                 w = w_list[jj]
#                 if jj==index:
#                     s_out *= meas_T_target[w]
#                 else:
#                     s_out *= meas_T_eye[w]
#             val_list.append(s_out)
#         return val_list
    
#     t = time.time()
#     out_list = []
#     for sample in range(sample_size):
#         out_list.append(sample_itr_3q_all())
#     print('p_estimate:',np.mean(out_list,axis=0),'\t(sampling time:',time.time()-t,')')
#     return out_list

def sample_state(T_list):
    w_list = []
    s = 1.
    for T in T_list:
        prob = np.abs(T)
        sign = np.sign(T)
        w = nr.choice(4, p=prob)
        w_list.append(w)
        s *= sign[w]
    return w_list,s

### Gate Optimization ###
def opt_Pauli_2q(prob_Pauli_output,**kwargs):
    t = time.time()
    options = {'opt_method': 'B', 'niter': 3}
    options.update(kwargs)

    state_T_list = prob_Pauli_output['state_T_list']
    gate_T_2q_list = prob_Pauli_output['gate_T_list']
    gate_P_2q_list = prob_Pauli_output['gate_P_list']
    gate_S_2q_list = prob_Pauli_output['gate_S_list']
    gate_N_2q_list = prob_Pauli_output['gate_N_list']
    index_list = prob_Pauli_output['index_list']
    meas_T_list = prob_Pauli_output['meas_T_list']

    state_num = len(state_T_list)
    x_list = []
    for index in range(state_num):
        x_list.append(np.array([0,0,0]))

    gate_T_2q_opt_list = []
    for index in range(len(gate_T_2q_list)):
        gate_T_2q = gate_T_2q_list[index]
        x1_index = index_list[index][0]
        x2_index = index_list[index][1]
        x1_temp_in = x_list[x1_index]
        x2_temp_in = x_list[x2_index]
        x_in = np.append(x1_temp_in,x2_temp_in)
        optimized_x, optimized_value = opt_neg_2q(gate_T_2q,x_in,**kwargs)
        x1_temp_out = optimized_x[0:3]
        x2_temp_out = optimized_x[3:6]
        x_out = np.append(x1_temp_out,x2_temp_out)
        gate_T_2q_opt_list.append(RTR_matrix_2q(gate_T_2q,x_in,x_out))

        x_list[x1_index] = x1_temp_out
        x_list[x2_index] = x2_temp_out

    meas_RT_list, neg_meas = get_meas_RT_list(meas_T_list,x_list)
    gate_N_2q_opt_list, gate_S_2q_opt_list, gate_P_2q_opt_list = get_NSP_sequence(gate_T_2q_opt_list)

    neg_gate = 1
    for gate_T_2q in gate_T_2q_opt_list:
        neg_gate *= max_negativity(gate_T_2q)
    print('Log_2 negativity (gate):',np.log2(neg_gate))
    print('Log_2 negativity (meas):',np.log2(neg_meas))
    print('Log_2 negativity:',np.log2(neg_gate) + np.log2(neg_meas),'\t(optimization time:)', time.time() -t,')')

    prob_Pauli_output_opt = {
        'state_T_list': state_T_list, 'gate_T_list': gate_T_2q_opt_list, 'gate_P_list': gate_P_2q_opt_list,
        'gate_S_list': gate_S_2q_opt_list, 'gate_N_list': gate_N_2q_opt_list, 'index_list': index_list, 'meas_T_list': meas_RT_list,
        'neg_gate': neg_gate
        }
    return prob_Pauli_output_opt

def opt_Pauli_3q(prob_Pauli_output,**kwargs):
    t = time.time()
    options = {'opt_method': 'B', 'niter': 3}
    options.update(kwargs)

    state_T_list = prob_Pauli_output['state_T_list']
    gate_T_3q_list = prob_Pauli_output['gate_T_list']
    gate_P_3q_list = prob_Pauli_output['gate_P_list']
    gate_S_3q_list = prob_Pauli_output['gate_S_list']
    gate_N_3q_list = prob_Pauli_output['gate_N_list']
    index_list = prob_Pauli_output['index_list']
    meas_T_list = prob_Pauli_output['meas_T_list']

    state_num = len(state_T_list)
    x_list = []
    for index in range(state_num):
        x_list.append(np.array([0,0,0]))

    gate_T_3q_opt_list = []
    for index in range(len(gate_T_3q_list)):
        gate_T_3q = gate_T_3q_list[index]
        x1_index = index_list[index][0]
        x2_index = index_list[index][1]
        x3_index = index_list[index][2]
        x1_temp_in = x_list[x1_index]
        x2_temp_in = x_list[x2_index]
        x3_temp_in = x_list[x3_index]
        x_in = np.append(np.append(x1_temp_in,x2_temp_in),x3_temp_in)
        optimized_x, optimized_value = opt_neg_3q(gate_T_3q,x_in,**kwargs)
        x1_temp_out = optimized_x[0:3]
        x2_temp_out = optimized_x[3:6]
        x3_temp_out = optimized_x[6:9]
        x_out = np.append(np.append(x1_temp_out,x2_temp_out),x3_temp_out)
        gate_T_3q_opt_list.append(RTR_matrix_3q(gate_T_3q,x_in,x_out))

        x_list[x1_index] = x1_temp_out
        x_list[x2_index] = x2_temp_out
        x_list[x3_index] = x3_temp_out

    meas_RT_list, neg_meas = get_meas_RT_list(meas_T_list,x_list)
    gate_N_3q_opt_list, gate_S_3q_opt_list, gate_P_3q_opt_list = get_NSP_sequence(gate_T_3q_opt_list)

    neg_gate = 1
    for gate_T_3q in gate_T_3q_opt_list:
        neg_gate *= max_negativity(gate_T_3q)
    print('Log_2 negativity (gate):',np.log2(neg_gate))
    print('Log_2 negativity (meas):',np.log2(neg_meas))
    print('Log_2 negativity:',np.log2(neg_gate) + np.log2(neg_meas),'\t(optimization time:)', time.time() -t,')')

    prob_Pauli_output_opt = {
        'state_T_list': state_T_list, 'gate_T_list': gate_T_3q_opt_list, 'gate_P_list': gate_P_3q_opt_list,
        'gate_S_list': gate_S_3q_opt_list, 'gate_N_list': gate_N_3q_opt_list, 'index_list': index_list, 'meas_T_list': meas_RT_list,
        'neg_gate': neg_gate
        }
    return prob_Pauli_output_opt


def opt_neg_1q(T1q,x_in,**kwargs):
    options = {'opt_method': 'B', 'niter': 3}
    options.update(kwargs)

    def cost_function(x):
        RTR = RTR_matrix_1q(T1q,x_in,x)
        return np.log(max_negativity(RTR))

    x0 = 2*np.random.rand(3)-1
    optimize_result, dt = optimizer(cost_function, x0, options['opt_method'], niter = options['niter'])
    optimized_x = optimize_result.x
    optimized_value = cost_function(optimized_x)
#     print('--------------------- OPTIMIZATION LOG--------------------\n', options)
#     print('Optimized log negativity:', optimized_value)
#     print('Computation time: ', dt)
#     print('--------------------------------------------------------------')
    return optimized_x, optimized_value

def opt_neg_2q(T2q,x_in,**kwargs):
    options = {'opt_method': 'B', 'niter': 3}
    options.update(kwargs)

    def cost_function(x):
        RTR = RTR_matrix_2q(T2q,x_in,x)
        return np.log(max_negativity(RTR))

    x0 = 2*np.random.rand(6)-1
    optimize_result, dt = optimizer(cost_function, x0, options['opt_method'], niter = options['niter'])
    optimized_x = optimize_result.x
    optimized_value = cost_function(optimized_x)
#     print('--------------------- OPTIMIZATION LOG--------------------\n', options)
#     print('Optimized log negativity:', optimized_value)
#     print('Computation time: ', dt)
#     print('--------------------------------------------------------------')
    return optimized_x, optimized_value

def opt_neg_3q(T3q,x_in,**kwargs):
    options = {'opt_method': 'B', 'niter': 3}
    options.update(kwargs)

    def cost_function(x):
        RTR = RTR_matrix_3q(T3q,x_in,x)
        return np.log(max_negativity(RTR))

    x0 = 2*np.random.rand(9)-1
    optimize_result, dt = optimizer(cost_function, x0, options['opt_method'], niter = options['niter'])
    optimized_x = optimize_result.x
    optimized_value = cost_function(optimized_x)
#     print('--------------------- OPTIMIZATION LOG--------------------\n', options)
#     print('Optimized log negativity:', optimized_value)
#     print('Computation time: ', dt)
#     print('--------------------------------------------------------------')
    return optimized_x, optimized_value



def opt_Pauli_2q_global(prob_Pauli_output,**kwargs): #### Not working ###
    options = {'opt_method': 'G', 'niter': 3}
    options.update(kwargs)

    state_T_list = prob_Pauli_output['state_T_list']
    gate_T_2q_list = prob_Pauli_output['gate_T_list']
    gate_P_2q_list = prob_Pauli_output['gate_P_list']
    gate_S_2q_list = prob_Pauli_output['gate_S_list']
    gate_N_2q_list = prob_Pauli_output['gate_N_list']
    index_list = prob_Pauli_output['index_list']
    meas_T_list = prob_Pauli_output['meas_T_list']

    state_num = len(state_T_list)
    x_list = []
    for index in range(state_num):
        x_list.append(np.array([0,0,0]))
    x_len = 2*len(gate_T_2q_list)

    def cost_function(x):
        neg = 1.
        for index in range(len(gate_T_2q_list)):
            gate_T_2q = gate_T_2q_list[index]
            x1_index = index_list[index][0]
            x2_index = index_list[index][1]
            x1_temp_in = x_list[x1_index]
            x2_temp_in = x_list[x2_index]
            x1_temp_out = x[6*index:6*index+3]
            x2_temp_out = x[6*index+3:6*index+6]
            x_in = np.append(x1_temp_in,x2_temp_in)
            x_out = np.append(x1_temp_out,x2_temp_out)
            neg *= max_negativity(RTR_matrix_2q(gate_T_2q,x_in,x_out))
            x_list[x1_index] = x1_temp_out
            x_list[x2_index] = x2_temp_out
        meas_RT_list, neg_meas = get_meas_RT_list(meas_T_list,x_list)
        return np.log(neg*neg_meas)

    x0 = np.zeros(3*x_len)
    optimize_result, dt = optimizer(cost_function, x0, options['opt_method'], niter = options['niter'])
    optimized_x = optimize_result.x
    optimized_value = cost_function(optimized_x)
    print('--------------------- GLOBAL OPTIMIZATION --------------------\n', options)
    print('Optimized Log Neg:', optimized_value)
    print('Computation time: ', dt)
    print('--------------------------------------------------------------')


    x_list = []
    for index in range(state_num):
        x_list.append(np.array([0,0,0]))
    gate_T_2q_opt_list = []
    for index in range(len(gate_T_2q_list)):
        gate_T_2q = gate_T_2q_list[index]
        x1_index = index_list[index][0]
        x2_index = index_list[index][1]
        x1_temp_in = x_list[x1_index]
        x2_temp_in = x_list[x2_index]
        x1_temp_out = optimized_x[3*(2*index):3*(2*index+1)]
        x2_temp_out = optimized_x[3*(2*index+2):3*(2*index+3)]
        x_in = np.append(x1_temp_in,x2_temp_in)
        x_out = np.append(x1_temp_out,x2_temp_out)
        gate_T_2q_opt_list.append(RTR_matrix_2q(gate_T_2q,x_in,x_out))
        x_list[x1_index] = x1_temp_out
        x_list[x2_index] = x2_temp_out

    meas_RT_list, neg_meas = get_meas_RT_list(meas_T_list,x_list)
    gate_N_2q_opt_list, gate_S_2q_opt_list, gate_P_2q_opt_list = get_NSP_sequence(gate_T_2q_opt_list)

    neg_gate = 1
    for gate_T_2q in gate_T_2q_opt_list:
        neg_gate *= max_negativity(gate_T_2q)
    print('Log_2 negativity (gate):',np.log2(neg_gate))
    print('Log_2 negativity (meas):',np.log2(neg_meas))
    print('Log_2 negativity:',np.log2(neg_gate) + np.log2(neg_meas))

    prob_Pauli_output = {
        'state_T_list': state_T_list, 'gate_T_list': gate_T_2q_opt_list, 'gate_P_list': gate_P_2q_opt_list,
        'gate_S_list': gate_S_2q_opt_list, 'gate_N_list': gate_N_2q_opt_list, 'index_list': index_list, 'meas_T_list': meas_RT_list,
        'neg_gate': neg_gate
        }
    return prob_Pauli_output

def optimizer(cost_function, x0, opt_method='B', niter = 10):
    start_time = time.time()

    if opt_method=='B': # autograd
        grad_cost_function = grad(cost_function)
        def func(x):
            return cost_function(x), grad_cost_function(x)
        optimize_result = basinhopping(func, x0, minimizer_kwargs={"method":"L-BFGS-B","jac":True},niter=niter)

    elif opt_method=='NG': # Powell
        optimize_result = minimize(cost_function, x0, method='Powell')

    elif opt_method=='G': # Without autograd
        optimize_result = minimize(cost_function, x0, method='L-BFGS-B',jac=grad_cost_function)

    else:
        raise Exception('Invalid optimisation method')

    dt = time.time()-start_time

    return optimize_result, dt


################ Test Code ##############################################
if __name__== "__main__":
    import time
    import matplotlib.pylab as plt

    from QUBIT_circuit_components import makeGate, makeCsum
    from QUBIT_circuit_generator import (random_connected_circuit, random_circuit,
           compress2q_circuit, compress3q_circuit, string_to_circuit,
           show_connectivity, solve_qubit_circuit, random_connected_circuit_2q3q)
    from QUBIT_BVcircuit import(BValg_circuit)
    from QUBIT_QD_circuit import QD_circuit
    from QUBIT_wig_neg import (wigner_neg_compressed, wigner_neg_compressed_3q)

    from QUBIT_Pauli_sampling import (get_prob_Pauli_2q,get_prob_Pauli_3q, sample_circuit_2q, sample_circuit_3q, opt_Pauli_2q, opt_Pauli_3q, opt_Pauli_2q_global)


    circuit, Tcount = random_connected_circuit(6, 71, Tgate_prob=0.19,
                                       given_state=None, given_measurement=1)

    print('T count=',Tcount)
    # circuit = BValg_circuit('1111', 0)
    circuit_compress_2q = compress2q_circuit(circuit)
    circuit_compress_3q = compress3q_circuit(circuit)

    print("===================Wigner sampling method======================")
    circ = QD_circuit(circuit)
    print("------------------2q compression-------------------")
    circ.compress_circuit(m=2)

    pborn1 = solve_qubit_circuit(circ.circuit)
    pborn2 = solve_qubit_circuit(circ.circuit_compressed)
    print("(2q-compression) Probs:", np.allclose(pborn1, pborn2),"(%.4f, %.4f)"%(pborn1, pborn2))
    circ.opt_x(method='Wigner')

    circ = QD_circuit(circuit)
    print("-----------------3q compression--------------------")
    circ.compress_circuit(m=3)
    pborn1 = solve_qubit_circuit(circ.circuit)
    pborn2 = solve_qubit_circuit(circ.circuit_compressed)
    print("(3q-compression) Probs:", np.allclose(pborn1, pborn2),"(%.4f, %.4f)"%(pborn1, pborn2))

    wigner_neg_compressed_3q(circ.circuit_compressed, method='Wigner')
    print("\n")

#     prob_Pauli_output_2q = get_prob_Pauli_2q(circuit_compress_2q)

    print("===================Pauli sampling method======================")
    print("------------------2q compression without optimization-------------------")
    prob_Pauli_output_2q = get_prob_Pauli_2q(circuit_compress_2q)
    print("------------------2q compression with optimization-------------------")
    prob_Pauli_output_opt_2q = opt_Pauli_2q(prob_Pauli_output_2q)
    print("------------------3q compression without optimization-------------------")
    prob_Pauli_output_3q = get_prob_Pauli_3q(circuit_compress_3q)
    print("------------------3q compression with optimization-------------------")
    prob_Pauli_output_opt_3q = opt_Pauli_3q(prob_Pauli_output_3q)

    print("=====================Prob. Estimation=========================")
    sample_size = int(1e6)
    print('Sample Size:\t',sample_size)
    print("------------------2q compression without optimization-------------------")
    sample_out_2q = sample_circuit_2q(prob_Pauli_output_2q, sample_size = sample_size)
    print("------------------2q compression with optimization-------------------")
    sample_out_opt_2q = sample_circuit_2q(prob_Pauli_output_opt_2q, sample_size = sample_size)
    print("------------------3q compression without optimization-------------------")
    sample_out_3q = sample_circuit_3q(prob_Pauli_output_3q, sample_size = sample_size)
    print("------------------3q compression with optimization-------------------")
    sample_out_opt_3q = sample_circuit_3q(prob_Pauli_output_opt_3q, sample_size = sample_size)
    print("===============================================================")

    x_list = np.linspace(1,sample_size,sample_size)
    y_list_2q = np.cumsum(sample_out_2q)/x_list
    y_list_opt_2q = np.cumsum(sample_out_opt_2q)/x_list
    y_list_3q = np.cumsum(sample_out_3q)/x_list
    y_list_opt_3q = np.cumsum(sample_out_opt_3q)/x_list
    plt.plot(x_list,y_list_2q,label='2q')
    plt.plot(x_list,y_list_opt_2q,label='Optimized 2q')
    plt.plot(x_list,y_list_3q,label='3q')
    plt.plot(x_list,y_list_opt_3q,label='Optimized 3q')
    plt.plot(x_list,pborn1*np.ones(len(x_list)),label='Exact')
#     plt.xlim((0,1e6))
    plt.ylim((-0.2,1.2))
    plt.xlabel('# of samples')
    plt.ylabel('Probability')
    plt.legend()
    plt.show()
