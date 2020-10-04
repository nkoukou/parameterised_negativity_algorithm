import numpy as np
import numpy.random as nr
import matplotlib.pylab as plt
import matplotlib.colors as colors
import circuit_components as cc
import itertools as it
from copy_phase_space import CopyPhaseSpace, evolve
#from copy_phase_space import evolve

d = 3
ps = CopyPhaseSpace(d,1)

def chop_zero(x):
    if np.isclose(x,0): return 0
    else: return x

def W_neg_state(state,s):
    A0s = ps.A0_1q(s)
    coords = it.product(*([range(d)]*2))
    wig_neg = 0.
    for xp in coords:
        wig_neg = wig_neg + abs(1./d * np.real( np.trace(
                    np.dot(state, evolve(A0s, ps.D(xp)) )) ))
#    return chop_zero(np.log(wig_neg))
    return np.log(wig_neg)

def W_neg_ch1q(gate1q, s_in, s_out):
    A0sin = ps.A0_1q(-s_in)
    A0sout = ps.A0_1q(s_out)
    coords = it.product(*([range(d)]*2))
    wig_neg_list = []
    for xp in coords:
        coords2 = it.product(*([range(d)]*2))
        wig_neg = 0.
        Aev = evolve(evolve(A0sin, ps.D(xp)), gate1q)
        for px in coords2:
            wig_neg = wig_neg + abs(1./d * np.real( np.trace(
                        np.dot(evolve(A0sout, ps.D(px)), Aev)) ))
        wig_neg_list.append(wig_neg)
#    return chop_zero(np.log(max(wig_neg_list)))
    return np.log(max(wig_neg_list))

def W_neg_csum(s_C_in,s_T_in,s_C_out,s_T_out):
    A0sin = np.kron(ps.A0_1q(-1.*s_C_in),ps.A0_1q(-1.*s_T_in))
    A0sout = np.kron(ps.A0_1q(s_C_out),ps.A0_1q(s_T_out))
    CSUM = np.array(
      [[1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 1., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 1., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 1.],
       [0., 0., 0., 0., 0., 0., 1., 0., 0.]])

    wig_neg_list = []
    coords_C_in = it.product(*([range(d)]*2))
    for xpC in coords_C_in:
        coords_T_in = it.product(*([range(d)]*2))
        for xpT in coords_T_in:
            wig_neg = 0.
            Din = np.kron(ps.D(xpC),ps.D(xpT))
            Aev = evolve(evolve(A0sin,Din), CSUM)
            coords_C_out = it.product(*([range(d)]*2))
            for pxC in coords_C_out:
                coords_T_out = it.product(*([range(d)]*2))
                for pxT in coords_T_out:
                    Dout = np.kron(ps.D(pxC),ps.D(pxT))
                    wig_neg = wig_neg + abs(1./d/d * np.real( np.trace(np.dot(evolve(A0sout, Dout), Aev)) ))
            wig_neg_list.append(wig_neg)
#    return chop_zero(np.log(max(wig_neg_list)))
    return np.log(max(wig_neg_list))

def W_neg_meas(meas,s):
    A0s = ps.A0_1q(-s)
    wig = []
    coords = it.product(*([range(d)]*2))
    for xp in coords:
        wig.append(abs(np.real( np.trace(np.dot(meas, evolve(A0s, ps.D(xp)) )) )))
#    return chop_zero(np.log(max(wig)))
    return np.log(max(wig))

def calc_neg_state(state_string, s_list):
    neg_state = 0.
    for state_index in range(len(state_string)):
        neg_state = neg_state + W_neg_state(cc.makeState1q(state_string[state_index]),s_list[state_index])
#    return chop_zero(neg_state)
    return neg_state

def calc_neg_channel(channel_string,s_list):
    neg_channel = 0.
    number_qutrit = int(len(s_list)/2)
    for channel_index in range(len(channel_string)):
        if channel_string[channel_index] == 'C':
            [Cindex, Tindex] = [channel_index, channel_string.find('T')]
            neg_channel = neg_channel + W_neg_csum(s_list[Cindex], s_list[Tindex], s_list[Cindex + number_qutrit], s_list[Tindex + number_qutrit])
        elif channel_string[channel_index] != 'T':
            gate1q = cc.makeGate1q(channel_string[channel_index])
            neg_channel = neg_channel + W_neg_ch1q(gate1q, s_list[channel_index],s_list[channel_index + number_qutrit])
#    return chop_zero(neg_channel)
    return neg_channel

def calc_neg_meas(meas_string,s_list):
    neg_meas = 0.
    for meas_index in range(len(meas_string)):
        idx = '0' if meas_string[meas_index]=='x' else meas_string[meas_index]
        neg_meas = neg_meas + W_neg_meas(cc.makeState1q(idx),s_list[meas_index])
#    return chop_zero(neg_meas)
    return neg_meas

def calc_neg_circuit_list(sequence,s_list):
    neg_circuit_list = []
    circuit_depth = len(sequence) - 2
    number_qutrit = len(sequence[0])

    neg_circuit_list.append(calc_neg_state(sequence[0],s_list[0:number_qutrit]))
    for l_index in range(circuit_depth):
        neg_circuit_list.append( calc_neg_channel(sequence[l_index+1],s_list[(l_index)*number_qutrit:(l_index+2) * number_qutrit]))
    neg_circuit_list.append(calc_neg_meas(sequence[-1],s_list[circuit_depth * number_qutrit:(circuit_depth + 1) * number_qutrit]))

    return neg_circuit_list

def calc_neg_circuit(sequence,s_list):
    circuit_depth = len(sequence) - 2
    number_qutrit = len(sequence[0])

    neg_circuit = calc_neg_state(sequence[0],s_list[0:number_qutrit])
    for l_index in range(circuit_depth):
        neg_circuit = neg_circuit + calc_neg_channel(sequence[l_index+1],s_list[(l_index)*number_qutrit:(l_index+2) * number_qutrit])
    neg_circuit = neg_circuit + calc_neg_meas(sequence[-1],s_list[circuit_depth * number_qutrit:(circuit_depth+1) * number_qutrit])

#    if np.isclose(neg_circuit,0): neg_circuit = 0
#    return chop_zero(neg_circuit)
    return neg_circuit

