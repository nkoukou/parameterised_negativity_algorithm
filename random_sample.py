import numpy as np
import itertools
from gate_seq2symplectic import gate_sequence2symplectic_form_merged, symplectic_inverse
from quasi_dist import W_state_list_1q, W_meas_list_1q
from make_state import makeState1q, makeMeas1q
DIM = 3
#
#def chop_zeros(a):
#    tol = 1e-14
#    b = np.array(a)
#    b.real[np.abs(b.real)<tol]=0
#    b.imag[np.abs(b.imag)<tol]=0
#    return b

x_range = list(range(DIM))
p_range = list(range(DIM))

def xp2index_1q(xp):
    return xp[0]*DIM + xp[1]
def index2xp_1q(index):
    return np.array([index/DIM,index%DIM],dtype=int)
    
ps_array_1q = []
for xp in itertools.product(x_range,repeat=2):
    xp = np.array(xp).flatten()
    ps_array_1q.append(xp2index_1q(xp))

def sample_xp_1q(p_dist,sign_dist):
    index_sample = np.random.choice(ps_array_1q,p=p_dist)
    sign = sign_dist[index_sample]
    xp_sample = index2xp_1q(index_sample)
    return xp_sample, sign

def sample_xp_string(p_string,sign_string):
    qudit_num = len(p_string)
    xp_string = []
    sign_out = 1
    for qudit_index in range(qudit_num):
        p_dist = p_string[qudit_index]
        sign_dist = sign_string[qudit_index]
        xp, sign = sample_xp_1q(p_dist,sign_dist)
        xp_string = np.append(xp_string,xp)
        sign_out = sign_out*sign
    return np.array(xp_string,dtype=int), sign_out

def get_p_dist_1q(rho,Cov,S):
    p_dist = np.real(W_state_list_1q(rho,Cov,S))
    sign_dist = np.sign(p_dist)
    p_dist = np.abs(p_dist)
    p_norm = p_dist.sum()
    p_dist = p_dist/p_norm
    return p_dist,sign_dist,p_norm
    
def get_p_string(state_string,gamma):
    qudit_num = len(state_string)
    
    p_string = []
    sign_string = []
    p_norm_out = 1
    
    S_eye_1q = np.eye(2,dtype=int)
    for qudit_index in range(qudit_num):
        rho = makeState1q(state_string[qudit_index])
        Cov = np.diag(gamma[2*qudit_index:2*qudit_index+2])
        p_dist, sign_dist,p_norm = get_p_dist_1q(rho,Cov,S_eye_1q)
        p_string.append(p_dist)
        sign_string.append(sign_dist)
        p_norm_out = p_norm_out*p_norm
    return p_string, sign_string, p_norm_out
    
def sample_circuit(circuit_string,gamma,sample_size):
    state_string = circuit_string[0]
    qudit_num = len(state_string)
    p_string, sign_string, p_norm_out = get_p_string(state_string,gamma)
    gate_sequence = circuit_string[1:-1]
    Stot, ztot = gate_sequence2symplectic_form_merged(gate_sequence)
    S = symplectic_inverse(Stot)
    meas_string = circuit_string[-1]
    MeasO, Meas_mode = makeMeas1q(meas_string)
    Cov = np.diag(gamma)
    W_meas_list = np.reshape(W_meas_list_1q(MeasO,Meas_mode, Cov,S),(DIM,DIM))
#    W_meas_list = np.reshape(W_meas_list_1q(MeasO,Meas_mode, Cov,Stot),(DIM,DIM))
    
    output_list = []
    for run in range(sample_size):
        xp_string, sign = sample_xp_string(p_string,sign_string)
#        xp_string = np.array(np.dot(Stot,xp_string) + ztot,dtype=int)
        xp_string = np.array(np.dot(Stot,xp_string) - ztot,dtype=int)
#        xp_string = np.array(np.dot(S,xp_string - ztot),dtype=int)
        ll = xp_string[2*Meas_mode:2*Meas_mode+2]%DIM
        output = np.real(p_norm_out*sign*W_meas_list[ll[0],ll[1]])
        output_list.append(output)
    return output_list

def accum_mean(list):
    size = len(list)
    arr = np.array(range(size))+1
    list_accum = np.add.accumulate(list)
    list_accum_mean = list_accum/arr
    return list_accum_mean


'''Sample Code'''
#print(index2xp_1q(2))
#print(xp2index_1q([0,2]))
#circuit_string = ['0STNT1T','11S11H1','1CT111S','1C11TSH','C1TH11S','1T11HCS','1111CTS','TSS111C','11111Z1']
#state_string = circuit_string[0]
#rho = makeState1q('T')
#Cov = np.array([[0,0],[0,0]])
#S = np.eye(2)
#p_dist, sign_dist, p_norm = get_p_dist_1q(rho,Cov,S)
#print(p_dist)
#print(sign_dist)
#print(p_norm)
#
#gamma = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#sample_size = 10000
#sample_list = sample_circuit(circuit_string,gamma,sample_size)
#print(np.mean(sample_list))
#
#import matplotlib.pyplot as plt
#plt.plot(accum_mean(sample_list))
#plt.show()
