import os
import time
import autograd.numpy as np
import matplotlib.pylab as plt

from QUBIT_circuit_generator import(random_connected_circuit, random_circuit,
       compress2q_circuit, compress3q_circuit, string_to_circuit,
       show_connectivity, solve_qubit_circuit, random_connected_circuit_2q3q)
from QUBIT_QD_circuit import(QD_circuit)
from QUBIT_wig_neg import(wigner_neg_compressed, wigner_neg_compressed_3q)

from QUBIT_Pauli_sampling import(get_prob_Pauli_2q, get_prob_Pauli_3q,
                                 sample_circuit_2q, sample_circuit_3q,
                                 opt_Pauli_2q, opt_Pauli_3q,
                                 opt_Pauli_2q_global)
from QUBIT_hidden_shift import(hidden_shift_circuit, circuit_class_to_label,
                               bool_oracle)


def reverse_circuit(circuit):
    rev_state_list = circuit['meas_list']
    rev_meas_list = circuit['state_list']
    rev_index_list = (circuit['index_list'])[::-1]
    conj_gate_list = []
    for gate in circuit['gate_list']:
        conj_gate_list.append(np.conjugate(gate.T))
    rev_gate_list = conj_gate_list[::-1]

    circuit_reversed = {'state_list':rev_state_list,
                        'gate_list': rev_gate_list,
                        'index_list': rev_index_list,
                        'meas_list':rev_meas_list}
    return circuit_reversed

def random_string(string_len):
    hidden_string = ''
    for index in range(string_len):
        hidden_bit = np.random.choice(['0','1'])
        hidden_string += hidden_bit
    return hidden_string

### set Oracle (random Boolean function) ###

### If you want to put hidden string manually ###
# hidden_string = '11001101'
# string_len = len(hidden_string)

### If you want to generate a random oracle ###
string_len = 10
hidden_string = random_string(string_len)


### Set oracle parameters: number of Z, CZ, and CCZ gates ###
Z_count = 3
CZ_count = 5
CCZ_count = 1

oc = bool_oracle(string_len)
oc.set_random_oracle(Z_count=100,CZ_count=50,CCZ_count=6)
print('Oracle index list:', oc.index_list_3q)


### set sample size ###
sample_size = int(1e4)


############ Below here everything will be run automatically #######

### set circuit ###
def meas_string(meas_qubit_index, string_len):
    meas_out = ''
    for index in range(string_len):
        if index == meas_qubit_index:
            meas_out += '1'
        else:
            meas_out +='/'
    return meas_out

cc = hidden_shift_circuit(string_len,hidden_string,oc)
cc.set_meas(meas_string(0,string_len))
circuit = compress2q_circuit(circuit_class_to_label(cc))

cc_3q = hidden_shift_circuit(string_len,hidden_string,oc,**{'Toffoli_Decomposition':'3q'})
cc_3q.set_meas(meas_string(0,string_len))
circuit_3q = compress2q_circuit(circuit_class_to_label(cc_3q))

### Wigner sampling ###
print("===================Wigner sampling method======================")
circ = QD_circuit(circuit)
print("------------------2q compression-------------------")
circ.compress_circuit(m=2)

# pborn1 = solve_qubit_circuit(circ.circuit)
# pborn2 = solve_qubit_circuit(circ.circuit_compressed)
# print("(2q-compression) Probs:", np.allclose(pborn1, pborn2),
#        "(%.4f, %.4f)"%(pborn1, pborn2))
circ.opt_x(method='Wigner')

circ = QD_circuit(circuit)
print("-----------------3q compression--------------------")
circ.compress_circuit(m=3)
# pborn1 = solve_qubit_circuit(circ.circuit)
# pborn2 = solve_qubit_circuit(circ.circuit_compressed)
# print("(3q-compression) Probs:", np.allclose(pborn1, pborn2),
#       "(%.4f, %.4f)"%(pborn1, pborn2))

wigner_neg_compressed_3q(circ.circuit_compressed, method='Wigner')
print("\n")

### Pauli sampling ###
rev_circuit = reverse_circuit(circuit)
circuit_compress_2q = compress2q_circuit(rev_circuit)

rev_circuit_3q = reverse_circuit(circuit_3q)
circuit_compress_3q = compress3q_circuit(rev_circuit_3q)

print("===================Pauli sampling method======================")
print("----------------2q compression without optimization-----------------")
prob_Pauli_output_2q = get_prob_Pauli_2q(circuit_compress_2q)
print("------------------2q compression with optimization------------------")
prob_Pauli_output_opt_2q = opt_Pauli_2q(prob_Pauli_output_2q)
print("----------------3q compression without optimization-----------------")
prob_Pauli_output_3q = get_prob_Pauli_3q(circuit_compress_3q)
print("------------------3q compression with optimization------------------")
prob_Pauli_output_opt_3q = opt_Pauli_3q(prob_Pauli_output_3q)
print("\n")

def get_rev_state_T_string(meas_qubit_index, string_len):
    state_T_list = []
    for index in range(string_len):
        if index == meas_qubit_index:
            state_T_list.append(np.array([0.0,0.0,0.0,-1.0]))
        else:
            state_T_list.append(np.array([1.0,0.0,0.0,0.0]))
    return state_T_list

def rev_sample_circuit_2q_all(prob_Pauli_output_2q,sample_size):
    out_list = []
    for meas_target in range(string_len):
        prob_Pauli_output_2q['state_T_list'] = get_rev_state_T_string(
            meas_target,string_len)
        sample_out_2q = sample_circuit_2q(prob_Pauli_output_2q,
                                          sample_size=sample_size)
        out_list.append(np.mean(sample_out_2q))
    return out_list

def rev_sample_circuit_3q_all(prob_Pauli_output_3q,sample_size):
    out_list = []
    for meas_target in range(string_len):
        prob_Pauli_output_3q['state_T_list'] = get_rev_state_T_string(
            meas_target,string_len)
        sample_out_3q = sample_circuit_3q(prob_Pauli_output_3q,
                                          sample_size=sample_size)
        out_list.append(np.mean(sample_out_3q))
    return out_list


out_list_2q = rev_sample_circuit_2q_all(prob_Pauli_output_2q,sample_size)
print('\n')
out_list_2q_opt = rev_sample_circuit_2q_all(prob_Pauli_output_opt_2q,
                                            sample_size)
print('\n')
out_list_3q = rev_sample_circuit_3q_all(prob_Pauli_output_3q,sample_size)
print('\n')
out_list_3q_opt = rev_sample_circuit_3q_all(prob_Pauli_output_opt_3q,
                                            sample_size)
print('\n')

print('Shifted String:',hidden_string)
X = np.arange(string_len)+1
plt.bar(X-0.3,(np.array(out_list_2q)+1)/2,
        color = 'b', width = 0.2, label='2q_comp.')
plt.bar(X-0.1,(np.array(out_list_2q_opt)+1)/2,
        color = 'r', width = 0.2, label='2q_comp. + local opt.')
plt.bar(X+0.1,(np.array(out_list_3q)+1)/2,
        color = 'g', width = 0.2, label='3q_comp.')
plt.bar(X+0.3,(np.array(out_list_3q_opt)+1)/2,
        color = 'y', width = 0.2, label='3q_comp. + local opt.')

plt.xticks(X,hidden_string)
plt.tick_params(axis='x',color='w',length=0)

plt.axhline(y=0.0, color='k', linestyle='-')
for xshift in range(string_len+1):
    plt.axvline(x=-0.5+xshift, color='k', linestyle='--',linewidth=0.3)

plt.xlabel('Hidden shift string')
plt.ylabel('Estimated probability')
plt.legend(bbox_to_anchor =(0.5, 1.1), loc = 'center', ncol = 2, fontsize=10)
plt.xlim((0.4,string_len+0.5))
plt.ylim((-0.1,1.1))
plt.tight_layout()
plt.savefig('Hidden_shift_Result.eps')
plt.show()

print('Shifted String:',hidden_string)
X = np.arange(string_len)+1
plt.bar(X,(np.array(out_list_2q)+1)/2,
        color = 'b', width = 0.5, label='2q_comp.')

plt.xticks(X,hidden_string)
plt.tick_params(axis='x',color='w',length=0)

plt.axhline(y=0.0, color='k', linestyle='-')
for xshift in range(string_len+1):
    plt.axvline(x=-0.5+xshift, color='k', linestyle='--',linewidth=0.3)

plt.xlabel('Hidden shift string')
plt.ylabel('Estimated probability')
plt.legend(bbox_to_anchor =(0.5, 1.1), loc = 'center', ncol = 2, fontsize=10)
plt.xlim((0.4,string_len+0.5))
plt.ylim((-0.1,1.1))
plt.tight_layout()
plt.savefig('Hidden_shift_2q_comp.eps')
plt.show()

print('Shifted String:',hidden_string)
X = np.arange(string_len)+1
plt.bar(X,(np.array(out_list_2q_opt)+1)/2,
        color = 'r', width = 0.5, label='2q_comp. + local opt.')

plt.xticks(X,hidden_string)
plt.tick_params(axis='x',color='w',length=0)

plt.axhline(y=0.0, color='k', linestyle='-')
for xshift in range(string_len+1):
    plt.axvline(x=-0.5+xshift, color='k', linestyle='--',linewidth=0.3)

plt.xlabel('Hidden shift string')
plt.ylabel('Estimated probability')
plt.legend(bbox_to_anchor =(0.5, 1.1), loc = 'center', ncol = 2, fontsize=10)
plt.xlim((0.4,string_len+0.5))
plt.ylim((-0.1,1.1))
plt.tight_layout()
plt.savefig('Hidden_shift_2q_comp_loc_opt.eps')
plt.show()

print('Shifted String:',hidden_string)
X = np.arange(string_len)+1
plt.bar(X,(np.array(out_list_3q)+1)/2,
        color = 'g', width = 0.5, label='3q_comp.')

plt.xticks(X,hidden_string)
plt.tick_params(axis='x',color='w',length=0)

plt.axhline(y=0.0, color='k', linestyle='-')
for xshift in range(string_len+1):
    plt.axvline(x=-0.5+xshift, color='k', linestyle='--',linewidth=0.3)

plt.xlabel('Hidden shift string')
plt.ylabel('Estimated probability')
plt.legend(bbox_to_anchor =(0.5, 1.1), loc = 'center', ncol = 2, fontsize=10)
plt.xlim((0.4,string_len+0.5))
plt.ylim((-0.1,1.1))
plt.tight_layout()
plt.savefig('Hidden_shift_3q_comp.eps')
plt.show()

print('Shifted String:',hidden_string)
X = np.arange(string_len)+1
plt.bar(X,(np.array(out_list_3q_opt)+1)/2,
        color = 'y', width = 0.5, label='3q_comp. + local opt.')

plt.xticks(X,hidden_string)
plt.tick_params(axis='x',color='w',length=0)

plt.axhline(y=0.0, color='k', linestyle='-')
for xshift in range(string_len+1):
    plt.axvline(x=-0.5+xshift, color='k', linestyle='--',linewidth=0.3)

plt.xlabel('Hidden shift string')
plt.ylabel('Estimated probability')
plt.legend(bbox_to_anchor =(0.5, 1.1), loc = 'center', ncol = 2, fontsize=10)
plt.xlim((0.4,string_len+0.5))
plt.ylim((-0.1,1.1))
plt.tight_layout()
plt.savefig('Hidden_shift_3q_comp_loc_opt.eps')
plt.show()

plt.subplot(221)
X = np.arange(string_len)+1
plt.bar(X,(np.array(out_list_2q)+1)/2,
        color = 'b', width = 0.5, label='2q_comp.')
plt.xticks(X,hidden_string,fontsize=8)
plt.tick_params(axis='x',color='w',length=0)
plt.axhline(y=0.0, color='k', linestyle='-')
for xshift in range(string_len+1):
    plt.axvline(x=-0.5+xshift, color='k', linestyle='--',linewidth=0.3)

plt.xlabel('Hidden shift string',fontsize=9)
plt.ylabel('Estimated probability',fontsize=9)
plt.legend(bbox_to_anchor =(0.5, 1.15), loc = 'center', ncol = 2, fontsize=8)
plt.xlim((0.4,string_len+0.5))
plt.ylim((-0.1,1.1))

plt.subplot(222)
plt.bar(X,(np.array(out_list_2q_opt)+1)/2,
        color = 'r', width = 0.5, label='2q_comp. + local opt.')
plt.xticks(X,hidden_string,fontsize=8)
plt.tick_params(axis='x',color='w',length=0)
plt.axhline(y=0.0, color='k', linestyle='-')
for xshift in range(string_len+1):
    plt.axvline(x=-0.5+xshift, color='k', linestyle='--',linewidth=0.3)

plt.xlabel('Hidden shift string',fontsize=9)
plt.ylabel('Estimated probability',fontsize=9)
plt.legend(bbox_to_anchor =(0.5, 1.15), loc = 'center', ncol = 2, fontsize=8)
plt.xlim((0.4,string_len+0.5))
plt.ylim((-0.1,1.1))

plt.subplot(223)
plt.bar(X,(np.array(out_list_3q)+1)/2,
        color = 'g', width = 0.5, label='3q_comp.')
plt.xticks(X,hidden_string,fontsize=8)
plt.tick_params(axis='x',color='w',length=0)
plt.axhline(y=0.0, color='k', linestyle='-')
for xshift in range(string_len+1):
    plt.axvline(x=-0.5+xshift, color='k', linestyle='--',linewidth=0.3)

plt.xlabel('Hidden shift string',fontsize=9)
plt.ylabel('Estimated probability',fontsize=9)
plt.legend(bbox_to_anchor =(0.5, 1.15), loc = 'center', ncol = 2, fontsize=8)
plt.xlim((0.4,string_len+0.5))
plt.ylim((-0.1,1.1))


plt.subplot(224)
plt.bar(X,(np.array(out_list_3q_opt)+1)/2,
        color = 'y', width = 0.5, label='3q_comp. + local opt.')

plt.xticks(X,hidden_string,fontsize=8)
plt.tick_params(axis='x',color='w',length=0)
plt.axhline(y=0.0, color='k', linestyle='-')
for xshift in range(string_len+1):
    plt.axvline(x=-0.5+xshift, color='k', linestyle='--',linewidth=0.3)

plt.xlabel('Hidden shift string',fontsize=9)
plt.ylabel('Estimated probability',fontsize=9)
plt.legend(bbox_to_anchor =(0.5, 1.15), loc = 'center', ncol = 2, fontsize=8)
plt.xlim((0.4,string_len+0.5))
plt.ylim((-0.1,1.1))

plt.tight_layout()
plt.savefig('Hidden_shift_all.eps')
plt.show()