import numpy as np
import itertools
from dis_op import D1q
from smoothing_function import h
from discrete_fourier import FT1q


DIM = 3

'''Default expression of a phase space point is (x1, p1, x2, p2, ..., xN, pN)'''
x_range = list(range(DIM))
p_range = list(range(DIM))

def W_state_point_1q(rho,ll,Cov,S):
    W_out = 0
    for w in itertools.product(x_range,repeat=2):
        w = np.array(w).flatten()
        W_out = W_out + h(Cov,S,w)*np.trace(np.dot(rho,D1q(-w)))*FT1q(ll,w)
    return W_out/DIM/DIM

def W_state_list_1q(rho,Cov,S):
    W_list = []
    for ll in itertools.product(x_range,repeat=2):
        ll = np.array(ll).flatten()
        W_list.append(W_state_point_1q(rho,ll,Cov,S))
    return W_list

def W_meas_point_1q(MeasO, Meas_mode, ll,Cov,S):
    N = len(Cov)
    W_out = 0
    for w in itertools.product(x_range,repeat=2):
        v = np.zeros(N,dtype=int)
        w = np.array(w,dtype=int).flatten()
        v[2*Meas_mode] = w[0]
        v[2*Meas_mode+1] = w[1]
        W_out = W_out + 1./h(Cov,S,-v)*np.trace(np.dot(MeasO,D1q(-w)))*FT1q(ll,w)
    return W_out/DIM

def W_meas_list_1q(MeasO, Meas_mode, Cov,S):
    W_list = []
    for ll in itertools.product(x_range,repeat=2):
        ll = np.array(ll).flatten()
        W_list.append(W_meas_point_1q(MeasO,Meas_mode, ll,Cov,S))
    return W_list
