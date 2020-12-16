import numpy as np
import os
import matplotlib.pyplot as plt

from test2 import random_gate_sequence, optimize_neg

#import warnings
#warnings.filterwarnings('ignore')


def random_state_string(qudit_num):
    state_string = ''
    for qudit_index in range(qudit_num):
        state_string = state_string + np.random.choice(['0','1','2','+','S','N','T'])
    return state_string

'''For 5 qutrit'''
qudit_num = 5
gate_num = 10
meas_string = '01111'
for run in range(100):
    state_string = random_state_string(qudit_num)
    filename = 'Q'+str(qudit_num)+'_G'+str(gate_num)+'_seq'+format(run, '03d')
    gate_sequence = random_gate_sequence(qudit_num,gate_num)
    optimize_neg(state_string,gate_sequence,meas_string,path=filename)
