import autograd.numpy as np
import itertools
from dis_op import D1q
#from discrete_fourier import FT1q
#from smoothing_function import h
#from gate_seq2symplectic import gate_sequence2symplectic_form_merged, symplectic_inverse
#from quasi_dist import W_state_list_1q, W_meas_list_1q
from make_state import makeState1q, makeMeas1q
from circuit_components import makeGate1q, makeCsum
#from neg_circuit import opt_neg_tot, show_neg_result
#from random_sample import sample_circuit, accum_mean
#from random_circuit_generator import random_circuit_string
#from calc_born_prob import simulate_circuit

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from autograd import grad
from scipy.optimize import Bounds, minimize
from scipy.optimize import basinhopping

import time
import os



DIM = 3
x_range = list(range(DIM))
p_range = list(range(DIM))

rho1 = makeState1q('T')
rho2 = makeState1q('0')

#TGate = np.array([[1,0,0],[0,np.exp(1j*np.pi/8),0],[0,0,np.exp(1j*np.pi/4)]])
TGate = np.array([[1,0,0],[0,np.exp(2*np.pi*1.j/9),0],[0,0,np.exp(-2*np.pi*1.j/9)]])
IGate = np.eye(DIM)
ZGate = np.array([[-0.5 - 0.866025*1.j, 0, 0], [0, 1., 0], [0, 0, -0.5 + 0.866025*1.j]])
CSUM = np.array(makeCsum('CT'))

D1q_list = []
for w in itertools.product(x_range,repeat=2):
    w = np.array(w,dtype=int)
    D1q_list.append(D1q(w))
D1q_list = np.array(np.reshape(D1q_list,(DIM,DIM,DIM,DIM)),dtype = "complex_")

def get_trace_D_Gamma(Gamma):
    out_list = []
    for w in itertools.product(x_range,repeat=2):
        p,q = w[0],w[1]
        out_list.append(np.trace(np.dot(D1q_list[p,q],Gamma)))
    return np.reshape(np.array(out_list,dtype = "complex_"),(DIM,DIM))

def get_F1q0(Gamma):
    D_Gamma_list = get_trace_D_Gamma(Gamma)
    F1q0 = np.zeros((DIM,DIM),dtype = "complex_")
    for w in itertools.product(x_range,repeat=2):
        p,q = w[0],w[1]
        F1q0 = F1q0 + D1q_list[-p,-q]/D_Gamma_list[p,q]
    return np.array(F1q0/DIM,dtype = "complex_")

def get_F1q_list(Gamma):
    F1q0 = get_F1q0(Gamma)
    F_list = []
    for ll in itertools.product(x_range,repeat=2):
        p,q = ll[0],ll[1]
        F_list.append(np.dot(np.dot(D1q_list[p,q],F1q0),D1q_list[-p,-q]))
    return np.reshape(np.array(F_list,dtype = "complex_"),(DIM,DIM,DIM,DIM))

def get_G1q_list(Gamma):
    G_list = []
    for ll in itertools.product(x_range,repeat=2):
        p,q = ll[0],ll[1]
        G_list.append(np.dot(np.dot(D1q_list[p,q],Gamma),D1q_list[-p,-q]))
    return np.reshape(np.array(G_list,dtype = "complex_"),(DIM,DIM,DIM,DIM))

def W_state_1q(rho,Gamma):
    w_list = []
    F1q = get_F1q_list(Gamma)
    for ll in itertools.product(x_range,repeat=2):
        p,q = ll[0],ll[1]
        w_list.append(np.trace(np.dot(rho,F1q[p,q])))
    return np.reshape(np.real(w_list),(DIM,DIM))/DIM

def neg_state_1q(rho,Gamma):
    return np.abs(W_state_1q(rho,Gamma)).sum()

def W_meas_1q(E,Gamma):
    w_list = []
    G1q = get_G1q_list(Gamma)
    for ll in itertools.product(x_range,repeat=2):
        p,q = ll[0],ll[1]
        w_list.append(np.trace(np.dot(E,G1q[p,q])))
    return np.reshape(np.real(w_list),(DIM,DIM))

def neg_meas_1q(E,Gamma):
    return np.max(np.abs(W_meas_1q(E,Gamma)))

def W_gate_1q(U,Gamma_in,Gamma_out):
    w_list = []
    G1q_in = get_G1q_list(Gamma_in)
    F1q_out = get_F1q_list(Gamma_out)
    for ll_in in itertools.product(x_range,repeat=2):
        p_in,q_in = ll_in[0],ll_in[1]
        rho_ev = np.dot(np.dot(U,G1q_in[p_in,q_in]),np.conjugate(U.T))
        for ll_out in itertools.product(x_range,repeat=2):
            p_out, q_out = ll_out[0], ll_out[1]
            w_list.append(np.trace(np.dot(rho_ev,F1q_out[p_out,q_out])))
    return np.reshape(np.real(w_list),(DIM,DIM,DIM,DIM))/DIM

def neg_gate_1q(U1q,Gamma_in,Gamma_out):
    neg_list = []
    G1q_in = get_G1q_list(Gamma_in)
    F1q_out = get_F1q_list(Gamma_out)
    for ll_in in itertools.product(x_range,repeat=2):
        p_in,q_in = ll_in[0],ll_in[1]
        rho_ev = np.dot(np.dot(U1q,G1q_in[p_in,q_in]),np.conjugate(U1q.T))
        neg = 0
        for ll_out in itertools.product(x_range,repeat=2):
            p_out, q_out = ll_out[0], ll_out[1]
            neg = neg + np.abs(np.trace(np.dot(rho_ev,F1q_out[p_out,q_out])))
        neg_list.append(neg)
    return np.max(neg_list)/DIM

def W_gate_2q(U2q,Gamma1_in,Gamma2_in,Gamma1_out,Gamma2_out):
    w_list = []
    G1_in = get_G1q_list(Gamma1_in)
    G2_in = get_G1q_list(Gamma2_in)
    F1_out = get_F1q_list(Gamma1_out)
    F2_out = get_F1q_list(Gamma2_out)
    for ll_in in itertools.product(x_range,repeat=4):
        p1_in,q1_in,p2_in,q2_in = ll_in[0],ll_in[1],ll_in[2],ll_in[3]
        G_in = np.kron(G1_in[p1_in,q1_in],G2_in[p2_in,q2_in])
        rho_ev = np.dot(np.dot(U2q,G_in),np.conjugate(U2q.T))
        for ll_out in itertools.product(x_range,repeat=4):
            p1_out,q1_out,p2_out,q2_out = ll_out[0],ll_out[1],ll_out[2],ll_out[3]
            F_out = np.kron(F1_out[p1_out,q1_out],F2_out[p2_out,q2_out])
            w_list.append(np.trace(np.dot(rho_ev,F_out)))
    return np.real(np.reshape(w_list,(DIM,DIM,DIM,DIM,DIM,DIM,DIM,DIM)))/DIM/DIM

def neg_gate_CSUM(GammaC_in,GammaT_in,GammaC_out,GammaT_out):
    G0_in = np.kron(GammaC_in,GammaT_in)
    FC_out = get_F1q_list(GammaC_out)
    FT_out = get_F1q_list(GammaT_out)
    neg = 0
    for ll in itertools.product(x_range,repeat=4):
        p1,q1,p2,q2 = ll[0],ll[1],ll[2],ll[3]
        rho_ev = np.dot(np.dot(CSUM,G0_in),np.conjugate(CSUM.T))
        F_out = np.kron(FC_out[p1,q1],FT_out[p2,q2])
        neg = neg + np.abs(np.trace(np.dot(rho_ev,F_out)))
    return neg/DIM/DIM

def neg_gate_Cliff_2q(U2q,GammaC_in,GammaT_in,GammaC_out,GammaT_out):
    G0_in = np.kron(GammaC_in,GammaT_in)
    FC_out = get_F1q_list(GammaC_out)
    FT_out = get_F1q_list(GammaT_out)
    neg = 0
    for ll in itertools.product(x_range,repeat=4):
        p1,q1,p2,q2 = ll[0],ll[1],ll[2],ll[3]
        rho_ev = np.dot(np.dot(U2q,G0_in),np.conjugate(U2q.T))
        F_out = np.kron(FC_out[p1,q1],FT_out[p2,q2])
        neg = neg + np.abs(np.trace(np.dot(rho_ev,F_out)))
    return neg/DIM/DIM

def neg_gate_2q(U2q,Gamma1_in,Gamma2_in,Gamma1_out,Gamma2_out):
    neg_list = []
    G1_in = get_G1q_list(Gamma1_in)
    G2_in = get_G1q_list(Gamma2_in)
    F1_out = get_F1q_list(Gamma1_out)
    F2_out = get_F1q_list(Gamma2_out)
    for ll_in in itertools.product(x_range,repeat=4):
        p1_in,q1_in,p2_in,q2_in = ll_in[0],ll_in[1],ll_in[2],ll_in[3]
        G_in = np.kron(G1_in[p1_in,q1_in],G2_in[p2_in,q2_in])
        rho_ev = np.dot(np.dot(U2q,G_in),np.conjugate(U2q.T))
        neg = 0
        for ll_out in itertools.product(x_range,repeat=4):
            p1_out,q1_out,p2_out,q2_out = ll_out[0],ll_out[1],ll_out[2],ll_out[3]
            F_out = np.kron(F1_out[p1_out,q1_out],F2_out[p2_out,q2_out])
            neg = neg + np.abs(np.trace(np.dot(rho_ev,F_out)))
        neg_list.append(neg)
    return np.max(neg_list)/DIM/DIM

def x2Gamma(x):
    return np.array([[x[0],x[1] + 1.j*x[2], x[3] + 1.j*x[4] ],[x[1] - 1.j*x[2], x[5], x[6]+1.j*x[7]],[ x[3] - 1.j*x[4],  x[6] - 1.j*x[7], 1-x[0]-x[5]]],dtype = "complex_")

def show_circuit(circuit_string):
    state_string = circuit_string[0]
    gate_sequence = circuit_string[1:-1]
    meas_string = circuit_string[-1]
    print(*state_string,sep='   ')
    for gate_string in gate_sequence:
        print(*gate_string,sep='   ')
    print(*meas_string,sep='   ')
    return 0

def optimize_neg(state_string, gate_sequence, meas_string, opt_method='B', path = 'test_directory'):
    ''' state_string: 'T0TT0T' '''
    ''' gate_sequence: [[[state_index],U1q],[[state_index_c,state_index_t],U2q], ... ] '''
    ''' meas_string:  '000000' '''
    qudit_num = len(state_string)
    x_length = qudit_num

    current_state_index = []
    Init_state_index = []
    x_index = 0
    for state_str in state_string:
        Init_state_index.append([x_index,makeState1q(state_str)])
        current_state_index.append(x_index)
        x_index = x_index + 1

    gate_1q_index = []
    gate_2q_index = []
    for gate_str in gate_sequence:
        if len(gate_str[0])==1:
            t_index = (gate_str[0])[0]
            gate_1q_index.append([[current_state_index[t_index],x_index],gate_str[1]])
            current_state_index[t_index] = x_index
            x_index = x_index + 1
        elif len(gate_str[0])==2:
            c_index = (gate_str[0])[0]
            t_index = (gate_str[0])[1]
            gate_2q_index.append( [[current_state_index[c_index],current_state_index[t_index],x_index,x_index+1],gate_str[1]])
            current_state_index[c_index] = x_index
            current_state_index[t_index] = x_index+1
            x_index = x_index + 2

    meas_index = []
    for meas_str in meas_string:
        if meas_str != '1':
            meas_index.append([x_index,makeState1q(meas_str)])
            x_index = x_index + 1

    x_len = x_index

    def cost_function(x):
        Gamma_list = []
        for x_index in range(x_len):
            Gamma_list.append(x2Gamma(x[8*x_index:8*(x_index+1)]))
        neg = 1.
        for state_index in Init_state_index:
            rho = state_index[1]
            Gamma = Gamma_list[state_index[0]]
            neg = neg*neg_state_1q(rho,Gamma)
        for gate_index in gate_1q_index:
            U1q = gate_index[1]
            Gamma_in = Gamma_list[(gate_index[0])[0]]
            Gamma_out = Gamma_list[(gate_index[0])[1]]
            neg = neg*neg_gate_1q(U1q,Gamma_in,Gamma_out)
        for gate_index in gate_2q_index:
            U2q = gate_index[1]
            GammaC_in = Gamma_list[(gate_index[0])[0]]
            GammaT_in = Gamma_list[(gate_index[0])[1]]
            GammaC_out = Gamma_list[(gate_index[0])[2]]
            GammaT_out = Gamma_list[(gate_index[0])[3]]
            neg = neg*neg_gate_Cliff_2q(U2q,GammaC_in,GammaT_in,GammaC_out,GammaT_out)
        for m_index in meas_index:
            E = m_index[1]
            Gamma = Gamma_list[m_index[0]]
            neg = neg*neg_meas_1q(E,Gamma)
        return np.log(neg)

#    print(Init_state_index)
#    print(gate_1q_index)
#    print(gate_2q_index)
#    print(meas_index)

    x0 = []
    for x_index in range(x_len):
        x0 = np.append(x0,[1,0,0,0,0,0,1,0])
    t0 = time.time()
    Wig_out = cost_function(x0)
    dt = time.time() - t0
    print('Wigner Log negativity:\t\t', Wig_out, '\t(Computation time:',dt,')')

#    x0 = 2*np.random.rand(8*x_len)-1

    '''Optimization with autograd'''
    if opt_method=='B':
        grad_cost_function = grad(cost_function)
        def func(x):
            return cost_function(x), grad_cost_function(x)
        t0=time.time()
        optimize_result = basinhopping(func, x0, minimizer_kwargs={"method":"L-BFGS-B", "jac":True}, disp=False, niter=1)
        dt = time.time()-t0
#    '''Optimization without autograd (Powell)'''
#    elif opt_method=='NG':
#        start_time = time.time()
#        optimize_result = minimize(cost_function, x0, method='Powell')
#        end_time = time.time()
#    '''Optimization without autograd (G)'''
#    elif opt_method=='G':
#        start_time = time.time()
#        optimize_result = opt.minimize(cost_function, x0, method='L-BFGS-B', jac=grad_cost_function, bounds=bnds, options={'disp': show_log})
#        end_time = time.time()

    '''Show results'''
    optimized_x = optimize_result.x
    optimized_value = cost_function(optimized_x)
#     print(optimized_x)
    print('Optimized Log negativity:\t',optimized_value,'\t(Computation time:', dt,')')

    '''Save results'''
    directory = os.path.join('data',path)
    if not os.path.isdir(directory):    os.mkdir(directory)
    np.save(os.path.join('data',path, 'state_string.npy'), state_string)
    np.save(os.path.join('data',path, 'gate_sequence.npy'), gate_sequence)
    np.save(os.path.join('data',path, 'meas_string.npy'), meas_string)
    np.save(os.path.join('data',path, 'optimized_x.npy'), optimized_x)
    np.save(os.path.join('data',path, 'optimized_neg.npy'), optimized_value)
    np.save(os.path.join('data',path, 'Wigner_neg.npy'), Wig_out)

    return optimized_x, optimized_value

def random_gate_sequence(qudit_num, gate_num):
    gate_sequence = []
    for index in range(gate_num):
        if np.random.randint(2) == 0:
            gate_sequence.append([[np.random.randint(qudit_num)],TGate])
        elif np.random.randint(2) == 1:
            [index1, index2] = np.random.choice(range(qudit_num), size=2, replace=False)
            gate_sequence.append([[index1,index2],CSUM])
    return gate_sequence

# qudit_num = 7
# gate_num = 50
# state_string = 'TTTTTTT'
# gate_sequence = random_gate_sequence(qudit_num,gate_num)
#print(*gate_sequence,sep='\n')
# meas_string = '0111111'
# optimize_neg(state_string,gate_sequence,meas_string,path='test01')

#optimize_neg('TTTT',[[[0],TGate],[[0,1],CSUM],[[3],TGate],[[0,3],CSUM],[[0],TGate],[[1,2],CSUM],[[1],TGate],[[2,0],CSUM]],'0010')


'''Sample codes'''
#rho = makeState1q('T')
#U = TGatej
#print(rho)
#print(np.dot(np.dot(U,makeState1q('T')),np.conjugate(U.T)))
#Gamma = np.array([[1,0,0],[0,0,1],[0,1,-0.5]])
#eta = 1.1
#Gamma2 = (1-eta)*Gamma + eta*np.array([[1,0,0],[0,0,1],[0,1,0]])
#print(neg_state_1q(rho,Gamma2))
#print(neg_gate_1q(U,Gamma,Gamma2))

#def cost_function(x):
#    Gamma2 = x2Gamma(x)
#    return neg_gate_1q(U,Gamma,Gamma2)
#x0 = 10*np.random.rand(8)-5
#optimize_result = minimize(cost_function, x0, method='Powell')
#optimized_x = optimize_result.x
#optimized_value = cost_function(optimized_x)
#print(optimized_x)
#print(optimized_value)
#print(x2Gamma(optimized_x))
#
#rho = makeState1q('T')
#E = makeState1q('0')
#U = TGate
#Gamma = np.array([[1,0,0],[0,0,1],[0,1,-0.5]])
#print(W_state_1q(rho,Gamma))
#print(neg_state_1q(rho,Gamma))
#print(W_meas_1q(E,Gamma))
#print(neg_meas_1q(E,Gamma))
#print(W_gate_1q(U,Gamma,Gamma))
#print(neg_gate_1q(U,Gamma,Gamma))
#CSUM = np.array(makeCsum('CT'))
#print(W_gate_2q(CSUM,Gamma,Gamma,Gamma,Gamma))
#t0=time.time()
#print(neg_gate_2q(CSUM,Gamma,Gamma,Gamma,Gamma))
#print(time.time()-t0)
#t0=time.time()
#print(neg_gate_CSUM(Gamma,Gamma,Gamma,Gamma))
#print(time.time()-t0)

'''CSUM example'''
#rho1_string = 'T'
#rho2_string = 'T'
#rho1 = makeState1q(rho1_string)
#rho2 = makeState1q(rho2_string)
#print('initial state1: ',rho1_string)
#print('initial state2: ',rho2_string)
#E = makeState1q('0')
#
#def neg_circuit_CSUM(rho1,rho2,E,GammaC_in,GammaT_in,GammaC_out,GammaT_out):
#    return np.real(neg_state_1q(rho1,GammaC_in)*neg_state_1q(rho2,GammaT_in) * neg_gate_CSUM(GammaC_in,GammaT_in,GammaC_out,GammaT_out) * neg_meas_1q(E,GammaC_out) * neg_meas_1q(E,GammaT_out))
#
#def cost_function(x):
#    Gamma1_in = x2Gamma(x[0:8])
#    Gamma2_in = x2Gamma(x[8:16])
#    Gamma1_out = x2Gamma(x[16:24])
#    Gamma2_out = x2Gamma(x[24:32])
#    return np.real(neg_circuit_CSUM(rho1,rho2,E,Gamma1_in,Gamma2_in,Gamma1_out,Gamma2_out))
#
#x0 = [1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0]
#t0=time.time()
#cost_function(x0)
#dt = time.time()-t0
#print('Wigner Negativity: ',cost_function(x0))
#print('Comp_time:',dt)
#
#
#'''Optimization with autograd'''
#grad_cost_function = grad(cost_function)
#def func(x):
#    return cost_function(x), grad_cost_function(x)
#x0 = 2*np.random.rand(32)-1
#t0=time.time()
#optimize_result = basinhopping(func, x0, minimizer_kwargs={"method":"L-BFGS-B", "jac":True}, disp=False, niter=1)
#dt = time.time()-t0
#
#'''Optimization without autograd (B)'''
###def func(x): return cost_function(x)
###x0 = 2*np.random.rand(32)-1
###t0=time.time()
###optimize_result = basinhopping(func, x0, disp=False, niter=1)
###dt = time.time()-t0
#'''Optimization without autograd (Powell)'''
###optimize_result = minimize(cost_function, x0, method='Powell')
#
#'''Optimization result'''
#optimized_x = optimize_result.x
#optimized_value = optimize_result.fun
#Gamma1_in_opt = x2Gamma(optimized_x[0:8])
#Gamma2_in_opt = x2Gamma(optimized_x[8:16])
#Gamma1_out_opt = x2Gamma(optimized_x[16:24])
#Gamma2_out_opt = x2Gamma(optimized_x[24:32])
#
#print('Optimized negativity: ', optimized_value)
#print('Comp_time:',dt)
#print('Optimized Gamma1 In: \n',Gamma1_in_opt)
#print('Optimized Gamma2 In: \n',Gamma2_in_opt)
#print('Optimized Gamma1 Out: \n',Gamma1_out_opt)
#print('Optimized Gamma2 Out: \n',Gamma2_out_opt)
#
#print('State1: \n',W_state_1q(rho1,Gamma1_in_opt))
#print('State2: \n',W_state_1q(rho2,Gamma2_in_opt))
##print('Gate: \n',W_gate_2q(CSUM,Gamma1_in_opt,Gamma2_in_opt,Gamma1_out_opt,Gamma2_out_opt))
#print('Meas1: \n',W_meas_1q(E,Gamma1_out_opt))
#print('Meas2: \n',W_meas_1q(E,Gamma2_out_opt))


'''Single qudit example'''
#rho_string = 'T'
#rho = makeState1q(rho_string)
#U = IGate
#E = makeState1q('0')
#
#def neg_circuit_1q(rho,U,E,Gamma_in,Gamma_out):
#    return np.real(neg_state_1q(rho,Gamma_in) * neg_gate_1q(U,Gamma_in,Gamma_out) * neg_meas_1q(E,Gamma_out))
#
#def cost_function(x):
#    Gamma_in = x2Gamma(x[0:8])
#    Gamma_out = x2Gamma(x[8:16])
#    return neg_circuit_1q(rho,U,E,Gamma_in,Gamma_out)
#x0 = [1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0]
#t0 = time.time()
#cost_function(x0)
#print(time.time()-t0)
#print('Wigner Negativity: ',cost_function(x0))
#
#
#x0 = 2*np.random.rand(16)-1
#grad_cost_function = grad(cost_function)
#def func(x):
#    return cost_function(x), grad_cost_function(x)
#
#t0=time.time()
#optimize_result = basinhopping(func, x0, minimizer_kwargs={"method":"L-BFGS-B", "jac":True}, disp=False, niter=1)
##optimize_result = minimize(cost_function, x0, method='Powell')
##optimize_result = minimize(cost_function, x0, method='L-BFGS-B', jac=grad_cost_function)
#dt = time.time()-t0
#
#optimized_x = optimize_result.x
#optimized_value = optimize_result.fun
#Gamma_in_opt = x2Gamma(optimized_x[0:8])
#Gamma_out_opt = x2Gamma(optimized_x[8:16])
#
#print('Optimized negativity: ', cost_function(optimized_x))
#print('Comp_time:',dt)
#print('Optimized Gamma In: \n',Gamma_in_opt)
#print('Optimized Gamma Out: \n',Gamma_out_opt)
#
#print('State: \n',W_state_1q(rho,Gamma_in_opt))
#print('Gate: \n',W_gate_1q(IGate,Gamma_in_opt,Gamma_out_opt))
#print('Meas: \n',W_meas_1q(E,Gamma_out_opt))


'''Sample circuit'''
#rho1_string = 'T'
#rho2_string = 'T'
#rho3_string = 'T'
#rho1 = makeState1q(rho1_string)
#rho2 = makeState1q(rho2_string)
#rho3 = makeState1q(rho3_string)
#print('initial state1: ',rho1_string)
#print('initial state2: ',rho2_string)
#print('initial state3: ',rho3_string)
#E = makeState1q('0')
#
#def neg_circuit(rho1,rho2,rho3,E,Gamma01,Gamma02,Gamma03,Gamma11,Gamma12,Gamma21,Gamma22,Gamma31,Gamma32,Gamma33):
#    outcome = 1.
#    outcome = outcome*neg_state_1q(rho1,Gamma01)*neg_state_1q(rho2,Gamma02)*neg_state_1q(rho3,Gamma03)
#    outcome = outcome*neg_gate_CSUM(Gamma01,Gamma02,Gamma11,Gamma12)*neg_gate_CSUM(Gamma12,Gamma03,Gamma22,Gamma33)*neg_gate_1q(TGate,Gamma11,Gamma21)*neg_gate_CSUM(Gamma22,Gamma21,Gamma32,Gamma31)
#    outcome = outcome*neg_meas_1q(E,Gamma31)
#    return outcome
#
#def cost_function(x):
#    Gamma01 = x2Gamma(x[0:8])
#    Gamma02 = x2Gamma(x[8:16])
#    Gamma03 = x2Gamma(x[16:24])
#    Gamma11 = x2Gamma(x[24:32])
#    Gamma12 = x2Gamma(x[32:40])
#    Gamma21 = x2Gamma(x[40:48])
#    Gamma22 = x2Gamma(x[48:56])
#    Gamma31 = x2Gamma(x[56:64])
#    Gamma32 = x2Gamma(x[64:72])
#    Gamma33 = x2Gamma(x[72:80])
#    return neg_circuit(rho1,rho2,rho3,E,Gamma01,Gamma02,Gamma03,Gamma11,Gamma12,Gamma21,Gamma22,Gamma31,Gamma32,Gamma33)
#
#x0 = [1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0]
#t0=time.time()
#cost_function(x0)
#print('Wigner Negativity:\t', cost_function(x0))
#print('Computation time:\t', time.time()-t0)

'''Optimization with autograd'''
#grad_cost_function = grad(cost_function)
#def func(x):
#    return cost_function(x), grad_cost_function(x)
#x0 = [1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0]
#t0=time.time()
#optimize_result = basinhopping(func, x0, minimizer_kwargs={"method":"L-BFGS-B", "jac":True}, disp=False, niter=1)
#dt = time.time()-t0

'''Optimization without autograd (Powell)'''
#x0 = [1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0]
#t0=time.time()
#optimize_result = minimize(cost_function, x0, method='Powell')
#dt = time.time()-t0

'''Optimization without autograd (G)'''
#grad_cost_function = grad(cost_function)
#def func(x):
#    return cost_function(x), grad_cost_function(x)
#x0 = [1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0]
#x0 = 2*np.random.rand(80)-1
#t0=time.time()
#optimize_result = minimize(cost_function, x0, method='L-BFGS-B', jac=grad_cost_function)
#dt = time.time()-t0
'''Show results'''
#optimized_x = optimize_result.x
#optimized_value = cost_function(optimized_x)
#
#print('Optimized negativity: ',cost_function(optimized_x))
#print('Computation time: ',dt)
#print('Optimized x: ',optimized_x)

