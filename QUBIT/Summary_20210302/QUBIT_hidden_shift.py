import numpy as np
from QUBIT_circuit_components import(makeState, makeGate, makeMeas)

class circuit(object):
    def __init__(self,qubit_num):
        '''Initial Circuit'''
        '''|0>---D'''
        '''|0>---/'''
        ''' ⋮ ---/'''
        '''|0>---/'''
        self.qubit_num = qubit_num
        self.state_list = self.qubit_num*[makeState('0')]
        self.gate_list = []
        self.index_list = []
        self.meas_list = self.qubit_num*[np.eye(2)]
        self.meas_list[0] = makeState('0')
    
    def set_init_qubit(self,init_qubit_string):
        self.state_list = []
        for qubit_string in init_qubit_string:
            self.state_list.append(makeState(qubit_string))

    def add_gate(self,U,index):
        self.gate_list.append(U)
        self.index_list.append(index)
        
    def add_gate_list(self,U_list,index_list):
        if len(U_list) != len(index_list):
            raise Exception('Gate and index lists do not match!')
        for jj in range(len(U_list)):
            self.gate_list.append(U_list[jj])
            self.index_list.append(index_list[jj])
        
    def set_meas(self,meas_string):
        self.meas_list = []
        for meas in meas_string:
            if meas =='/':
                self.meas_list.append(np.eye(2))
            else:
                self.meas_list.append(makeState(meas))

            
def Hadamard_all(circuit):
    H_gate = makeGate('H')
    for index in range(circuit.qubit_num):
        circuit.add_gate(H_gate,[index])

def X_string(circuit,x_string):
    X_gate = makeGate('X')
    if len(x_string) != circuit.qubit_num:
        raise Exception('length does not match!')
    index = 0
    for s in x_string:
        if s == '1':
            circuit.add_gate(X_gate,[index])
        if s != '0' and s !='1':
            raise Exception('Invalid string')
        index += 1
        
def half_CZ_gate(circuit):
    if circuit.qubit_num%2 == 1:
        raise Exception('qubit number is not even!')
    half_index = circuit.qubit_num//2
    for index in range(half_index):
        circuit.add_gate(CZ_gate,[index,half_index+index])
        
def make_CCZ():
    gate = np.eye(8)
    gate[7,7] = -1
    return gate

T_gate = makeGate('T')
t_gate = makeGate('t')
Z_gate = np.array([[1,0],[0,-1]])
CZ_gate = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]])
CN_gate = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
CCZ_gate = make_CCZ()

class bool_oracle(object):
    def __init__(self,qubit_num):
        self.qubit_num = qubit_num
        self.gate_list_2q = [Z_gate]
        self.index_list_2q = [[0]]
        self.gate_list_3q = [Z_gate]
        self.index_list_3q = [[0]]
        
    def add_random_Z(self,index = None):
        if index is None:
            index = np.ndarray.tolist(np.random.choice(self.qubit_num//2,1,replace=False))
        self.gate_list_2q.append(Z_gate)
        self.index_list_2q.append(index)
        self.gate_list_3q.append(Z_gate)
        self.index_list_3q.append(index)
        
    def add_random_CZ(self,index = None):
        if index is None:
            index = np.ndarray.tolist(np.random.choice(self.qubit_num//2,2,replace=False))
        self.gate_list_2q.append(CZ_gate)
        self.index_list_2q.append(index)
        self.gate_list_3q.append(CZ_gate)
        self.index_list_3q.append(index)
    
    def add_CCZ_2q(self,index):
        self.gate_list_2q.append(CN_gate)
        self.index_list_2q.append([index[1],index[2]])
        self.gate_list_2q.append(t_gate)
        self.index_list_2q.append([index[2]])
        self.gate_list_2q.append(CN_gate)
        self.index_list_2q.append([index[0],index[2]])
        self.gate_list_2q.append(T_gate)
        self.index_list_2q.append([index[2]])
        self.gate_list_2q.append(CN_gate)
        self.index_list_2q.append([index[1],index[2]])
        self.gate_list_2q.append(t_gate)
        self.index_list_2q.append([index[2]])
        self.gate_list_2q.append(CN_gate)
        self.index_list_2q.append([index[0],index[2]])
        self.gate_list_2q.append(T_gate)
        self.index_list_2q.append([index[1]])
        self.gate_list_2q.append(T_gate)
        self.index_list_2q.append([index[2]])
        self.gate_list_2q.append(CN_gate)
        self.index_list_2q.append([index[0],index[1]])
        self.gate_list_2q.append(T_gate)
        self.index_list_2q.append([index[0]])
        self.gate_list_2q.append(t_gate)
        self.index_list_2q.append([index[1]])
        self.gate_list_2q.append(CN_gate)
        self.index_list_2q.append([index[0],index[1]])
        
    def add_random_CCZ(self,index = None):
        if index is None:
            index = np.ndarray.tolist(np.random.choice(self.qubit_num//2,3,replace=False))
        self.gate_list_3q.append(CCZ_gate)
        self.index_list_3q.append(index)
        self.add_CCZ_2q(index)

    def set_random_oracle(self,Z_count=1,CZ_count=1,CCZ_count=1):
        total_count = Z_count + CZ_count + CCZ_count
        Z_count_temp = Z_count
        CZ_count_temp = CZ_count
        CCZ_count_temp = CCZ_count
        for index in range(total_count):
            total_count_temp = Z_count_temp + CZ_count_temp + CCZ_count_temp
            p_Z = Z_count_temp/total_count_temp
            p_CZ = CZ_count_temp/total_count_temp
            p_CCZ = CCZ_count_temp/total_count_temp
            rr = np.random.choice(3,p=[p_Z,p_CZ,p_CCZ])
            if rr == 0:
                self.add_random_Z()
                Z_count_temp -= 1
            if rr == 1:
                self.add_random_CZ()
                CZ_count_temp -= 1
            if rr == 2:
                self.add_random_CCZ()
                CCZ_count_temp -= 1
                
#     def set_random_oracle_3q(self,Z_count=1,CZ_count=1,CCZ_count=1):
#         total_count = Z_count + CZ_count + CCZ_count
#         Z_count_temp = Z_count
#         CZ_count_temp = CZ_count
#         CCZ_count_temp = CCZ_count
#         for index in range(total_count):
#             total_count_temp = Z_count_temp + CZ_count_temp + CCZ_count_temp
#             p_Z = Z_count_temp/total_count_temp
#             p_CZ = CZ_count_temp/total_count_temp
#             p_CCZ = CCZ_count_temp/total_count_temp
#             rr = np.random.choice(3,p=[p_Z,p_CZ,p_CCZ])
#             if rr == 0:
#                 self.add_random_Z()
#                 Z_count_temp -= 1
#             if rr == 1:
#                 self.add_random_CZ()
#                 CZ_count_temp -= 1
#             if rr == 2:
#                 self.add_random_CCZ()
#                 CCZ_count_temp -= 1
                
def apply_hidden_shift_oracle(circuit, oracle, offset = 0,**kwargs):
    '''offset 0 -> Upper half'''
    '''offset 1 -> Lower half'''
    options = {'Toffoli_Decomposition':'2q'}
    options.update(kwargs)
    
    if circuit.qubit_num%2 == 1:
        raise Exception('qubit number is not even!')
        
    if circuit.qubit_num != oracle.qubit_num:
        raise Exception('circuit and oracle qubit numbers do not match')
    
    
    
    if options['Toffoli_Decomposition']=='2q':
        oracle_gate_list = oracle.gate_list_2q
        oracle_index_list = oracle.index_list_2q
    elif options['Toffoli_Decomposition']=='3q':
        oracle_gate_list = oracle.gate_list_3q
        oracle_index_list = oracle.index_list_3q
        
    half_index = circuit.qubit_num//2
    index_offset = int(offset*half_index)
    gate_list = oracle_gate_list
    index_list = []
    for jj in range(len(oracle_index_list)):
        index_list.append(np.ndarray.tolist(np.array(oracle_index_list[jj],dtype='int') + index_offset))
    circuit.add_gate_list(gate_list,index_list)
    

def hidden_shift_circuit(qubit_num = 6, s_string = None, oracle = None, **kwargs):
    if s_string is None:
        s_string = qubit_num*'1'
    elif len(s_string) != qubit_num:
        raise Exception('qubit number and string size do not match!')
    print('Shifted string:', s_string)
    
    if oracle is None:
        oracle = bool_oracle(qubit_num)

    cc = circuit(qubit_num)
    Hadamard_all(cc)
    X_string(cc,s_string)
    apply_hidden_shift_oracle(cc,oracle,0,**kwargs)
    half_CZ_gate(cc)
    X_string(cc,s_string)
    Hadamard_all(cc)
    apply_hidden_shift_oracle(cc,oracle,1,**kwargs)
    half_CZ_gate(cc)
    Hadamard_all(cc)
    
    return cc


def circuit_class_to_label(circuit_class):
    circuit_label = {'state_list': circuit_class.state_list, 'gate_list': circuit_class.gate_list,'index_list': circuit_class.index_list, 'meas_list': circuit_class.meas_list}
    return circuit_label