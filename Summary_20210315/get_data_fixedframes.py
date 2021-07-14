import time
import os
import itertools as it

import autograd.numpy as np
from autograd import(grad)
from scipy.optimize import(Bounds, minimize)
from scipy.optimize import(basinhopping)
import matplotlib.pylab as plt

from circuit_components import(makeGate)
from circuit_generator import(random_connected_circuit)
from phase_space import(x2Gamma, neg_gate_1q_max, neg_gate_2q_max,
                        neg_meas_1q, neg_state_1q)
from frame_opt import(wigner_neg_compressed)

direc = 'test'

def fixed_opt_state(state, set_of_frames):
    neg = neg_state_1q(state, x2Gamma(set_of_frames[0]))
    return neg, 0

def fixed_opt_gate2q_alt(U2q, x_in1, x_in2, set_of_frames):
    neg_list = [neg_gate_2q_max(U2q, x2Gamma(x_in1), x2Gamma(x_in2),
                                x2Gamma(set_of_frames[i]),
                                x2Gamma(set_of_frames[i]))
                for i,j in it.product(range(len(set_of_frames)), repeat=2)]
    neg = min(neg_list)
    index_total = neg_list.index(neg)
    x_out1_index = index_total//len(set_of_frames)
    x_out2_index = index_total%len(set_of_frames)
    return neg, x_out1_index, x_out2_index

def fixed_opt_gate2q(U2q, x_in1, x_in2, set_of_frames):
    F = len(set_of_frames)
    neg = np.zeros((F, F))
    for i in range(F):
        for j in range(F):
            neg[i,j] = neg_gate_2q_max(U2q,
                                x2Gamma(x_in1),
                                x2Gamma(x_in2),
                                x2Gamma(set_of_frames[i]),
                                x2Gamma(set_of_frames[j]))
    negmin = neg.min()
    idx = neg.argmin()
    x_out1_idx = idx//F
    x_out2_idx = idx%F
    return negmin, x_out1_idx, x_out2_idx


def fixed_opt_neg(circuit, set_of_frames):
    state_list = circuit['state_list']
    gate_list = circuit['gate_list']
    index_list = circuit['index_list']
    meas_list = circuit['meas_list']

    x_rho_opt_list = []
    neg_rho_opt_list = []
    neg_tot = 1.
    for rho in state_list:
        neg, x_opt_index = fixed_opt_state(rho, set_of_frames)
        x_rho_opt_list.append(set_of_frames[x_opt_index])
        neg_rho_opt_list.append(neg)
        neg_tot *= neg

    x_running = x_rho_opt_list.copy()
    x_gate_out_opt_list = []
    neg_gate_opt_list = []
    for gate_index in range(len(gate_list)):
        U2q = gate_list[gate_index]
        [qudit_index1, qudit_index2] = index_list[gate_index]
        x_in1 = x_running[qudit_index1]
        x_in2 = x_running[qudit_index2]
        neg, x_out_opt1_index, x_out_opt2_index = fixed_opt_gate2q(U2q, x_in1,
                                                         x_in2, set_of_frames)
        x_gate_out_opt_list.append([set_of_frames[x_out_opt1_index],
                                    set_of_frames[x_out_opt2_index]].copy())
        neg_gate_opt_list.append(neg)
        neg_tot *= neg
        x_running[qudit_index1] = set_of_frames[x_out_opt1_index]
        x_running[qudit_index2] = set_of_frames[x_out_opt2_index]

    x_meas_opt_list = x_running
    qudit_index = 0
    neg_meas_opt_list = []
    for meas in meas_list:
        if str(meas) == '/': continue
        x_out = x_meas_opt_list[qudit_index]
        qudit_index += 1
        neg = neg_meas_1q(meas, x2Gamma(x_out))
        neg_meas_opt_list.append(neg)
        neg_tot *= neg

    # print('--------------------- FIXED OPTIMIZATION ---------------------')
    # print('Fixed Opt Log Neg:', np.log(neg_tot))
    # print('--------------------------------------------------------------')

    x_opt_list_tot = np.append(np.array(x_rho_opt_list).flatten(),
                               np.array(x_gate_out_opt_list).flatten())
    return x_opt_list_tot, np.log(neg_tot)

def find_optimal_frame(n_qudit=20, n_CNOT=40):
    frame0 = np.array([0., 0.766, 0.6428, 0., 0., 0., 0., 0.])
    # np.array([1,0,0,0,0,0,1,0])
    def cost_function(frame):
        neg_list, x_list = optimise_frame(frame, rng=(50,61))
        neg = np.sum(neg_list[1] - neg_list[0])
        return neg
    # grad_cost_function = grad(cost_function)
    # def func(x):
    #     return cost_function(x), grad_cost_function(x)
    # res = basinhopping(func, frame0,
    #         minimizer_kwargs={"method":"L-BFGS-B","jac":True},niter=10)
    res = minimize(cost_function, frame0)
    return res


def optimise_frame(frame, n_qudit=20, n_CNOT=40, rng=None):
    frames = np.stack((np.array([1,0,0,0,0,0,1,0]),frame))
    if rng is None: rng=(0,2*n_CNOT+1)

    neg_list = np.zeros((2, 2*n_CNOT+1))
    x_list = [[],[]]
    for n_Tgate in range(rng[0], rng[1]):
        if n_Tgate%5==0: print('T: %d/%d'%(n_Tgate,2*n_CNOT))

        circuit, Tcount = random_connected_circuit(qudit_num=n_qudit,
                        circuit_length=n_CNOT, Tgate_prob=n_Tgate,
                        given_state=None, given_measurement=2, method='c')

        WI_x, WI_neg = wigner_neg_compressed(circuit)
        F2_x, F2_neg = fixed_opt_neg(circuit, frames)

        neg_list[0, n_Tgate]  = WI_neg
        neg_list[1, n_Tgate]  = F2_neg
        x_list[0].append(WI_x)
        x_list[1].append(F2_x)

    return neg_list, np.array(x_list)

def plot_neg(n_CNOT, neg_list):
    tcount = np.arange(2*n_CNOT+1)
    fig, ax = plt.subplots(1,1)
    ax.plot(tcount, neg_list[0], ls='-', marker='', c='k')
    ax.plot(tcount, neg_list[1], ls='', marker='o', c='r')


def optimise_single_frame():
    U = makeGate('T')
    def cost_function(frame):
        frames = [np.array([1,0,0,0,0,0,1,0]), frame]
        neg = 1.
        for f0 in frames:
            for f1 in frames:
                neg *= neg_gate_1q_max(U, x2Gamma(f0), x2Gamma(f1))
        return neg
    grad_cost_function = grad(cost_function)
    def func(x):
        return cost_function(x), grad_cost_function(x)
    frame0 = np.array([0., 0.766, 0.6428, 0., 0., 0., 0., 0.])
    # frame0 = np.array([1,1,0,0,0,0,1,0])
    res = basinhopping(func, frame0,
                       minimizer_kwargs={"method":"L-BFGS-B","jac":True},
                       disp=False, niter=1)
    # res = minimize(cost_function, frame, args=(U))
    return res
t = optimise_single_frame()
print(t)

f0 = np.array([1,0,0,0,0,0,1,0])
U = makeGate('T')
t1= neg_gate_1q_max(U, x2Gamma(f0), x2Gamma(f0))
t2= neg_gate_1q_max(U, x2Gamma(f0), x2Gamma(t.x))
t3= neg_gate_1q_max(U, x2Gamma(t.x), x2Gamma(f0))
t4= neg_gate_1q_max(U, x2Gamma(t.x), x2Gamma(t.x))
print(t1, t2, t3, t4, t1*t2*t3*t4)

# n_qudit=20
# n_CNOT=40
# x1 = [0., 0.766, 0.6428, 0., 0., 0., 0., 0.]
# x1 = np.array([ 3.5879,  0.6287,  1.636 , -1.4022,  2.0344,  2.5879,  0.3751,
# #         0.6287])
# fname = 'Q'+str(n_qudit)+'_CNOT'+str(n_CNOT)+'.npy'
# neg_list, x_list = optimise_frame(frame=x1, n_qudit=n_qudit, n_CNOT=n_CNOT)
# np.save(os.path.join(direc, 'neg_'+fname), neg_list)
# np.save(os.path.join(direc, 'x_'+fname), x_list)
# plot_neg(n_CNOT, neg_list)

# res = find_optimal_frame(n_qudit=20, n_CNOT=40)

















