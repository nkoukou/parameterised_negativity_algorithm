import numpy as np
from state_functions import(evolve)
from circuit_components import(makeGate, makeState)
from random_circuit_generator import(random_circuit, show_circuit)
from opt_neg import(optimize_neg)
from prob_estimation import(sample)

# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# circuit = ['0', [[[0], 'H']], '0']

# circuit = ['0TT', [[[0], 'T'   ],
#                    [[0,1], 'C+'],
#                    [[1], 'T'   ],
#                    [[0,1], 'C+'],
#                    [[0], 'T'   ],
#                    [[1,2], 'C+'],
#                    [[1],   'T' ],
#                    [[2,0], 'C+']
#                    ],'010']

# circuit = random_circuit(qudit_num=3,
#                           C1qGate_num=6, TGate_num=2, CSUMGate_num=3,
#                           given_state=None,
#                           given_measurement=2)
# show_circuit(circuit)
# optimize_neg(circuit)

# p= sample(circuit, 0)
# q= sample(circuit, np.load('data/test_directory/optimized_x.npy'))


def test_sampling(N=1e1):
    N = int(N)
    path = 'data/test_directory/optimized_x.npy'

    # Test simple Clifford
#    circuit = ['0', [[[0], 'H']], '0']
#    out_wig, out_wig_list = sample(circuit, x=0, niters=N)
#
#    p_born = 1/3
#
#    print('----------------------------------------------------------------')
#    show_circuit(circuit)
#    print('Wigner sample p    =', out_wig)
#    print('Exact Born p       =', p_born)
#    print('----------------------------------------------------------------')

    # Test T state & 2-qutrit measurement
#    circuit = ['0T', [[[1,0], 'C+']], '++']
#    out_wig, out_wig_list = sample(circuit, x=0, niters=N)
#    optimize_neg(circuit)
#    out_opt, out_opt_list = sample(circuit, x=np.load(path), niters=N)
#
#    state = evolve(np.kron(makeState('0'), makeState('T')), makeGate('+C'))
#    p_born = np.trace(np.dot(state, makeState('++'))).real
#
#    wig_plot, = plt.plot(out_wig_list,'r',label = 'Wigner-sampling')
#    opt_plot, = plt.plot(out_opt_list,'b',label = 'Opt-sampling')
#    born_plot, = plt.plot(np.ones(len(out_wig_list))*p_born,'k-.',label = 'Born rule probability')
#    plt.xlabel("Iterations")
#    plt.ylabel("Probability")
#    plt.legend(handles=[wig_plot,opt_plot,born_plot])
#    plt.show()
#
#    print('----------------------------------------------------------------')
#    show_circuit(circuit)
#    print('Wigner sample p    =', out_wig)
#    print('Optimised sample p =', out_opt)
#    print('Exact Born p       =', p_born)
#    print('----------------------------------------------------------------')

    # Test complicated circuit
    circuit = ['1T0', [[[0], 'S'], [[1], 'H'], [[1], 'T'], [[2], 'H'],
                       [[1, 0], 'C+'], [[1], 'T'], [[1, 0], 'C+'],
                       [[2], 'H'], [[2], 'H'], [[2, 1], 'C+'], [[2], 'S']],
               '10/']
    out_wig, out_wig_list = sample(circuit, x=0, niters=N)
    optimize_neg(circuit)
    out_opt, out_opt_list = sample(circuit, x=np.load(path), niters=N)
    
    state = makeState('1T0')
    state = evolve(state, makeGate('SHH'))
    state = evolve(state, makeGate('1TH'))
    state = evolve(state, makeGate('+C1'))
    state = evolve(state, makeGate('11H'))
    state = evolve(state, makeGate('1T1'))
    state = evolve(state, makeGate('+C1'))
    state = evolve(state, makeGate('1+C'))
    state = evolve(state, makeGate('11S'))
    p_born = np.trace(np.dot(state, makeState('10m')*3)).real
    
    wig_plot, = plt.plot(out_wig_list,'r',label = 'Wigner-sampling')
    opt_plot, = plt.plot(out_opt_list,'b',label = 'Opt-sampling')
    born_plot, = plt.plot(np.ones(len(out_wig_list))*p_born,'k-.',label = 'Born rule probability')
    plt.xlabel("Iterations")
    plt.ylabel("Probability")
    plt.legend(handles=[wig_plot,opt_plot,born_plot])
    plt.show()

    print('----------------------------------------------------------------')
    show_circuit(circuit)
    print('Wigner sample p    =', out_wig)
    print('Optimised sample p =', out_opt)
    print('Exact Born p       =', p_born)
    print('----------------------------------------------------------------')

    return 0
    # Test T gate
    # Test CSUM and T

def test_sampling2(qudit_num, C1qGate_num, TGate_num, CSUMGate_num,N=1e2):
    N = int(N)
    path = 'data/test_directory/optimized_x.npy'
    
    circuit = random_circuit(qudit_num, C1qGate_num, TGate_num, CSUMGate_num)

    out_wig, out_wig_list = sample(circuit, x=0, niters=N)
    optimize_neg(circuit)
    out_opt, out_opt_list = sample(circuit, x=np.load(path), niters=N)

    wig_plot, = plt.plot(out_wig_list,'r',label = 'Wigner-sampling')
    opt_plot, = plt.plot(out_opt_list,'b',label = 'Opt-sampling')
#    born_plot, = plt.plot(np.ones(len(out_wig_list))*p_born,'k-.',label = 'Born rule probability')
    plt.xlabel("Iterations")
    plt.ylabel("Probability")
#    plt.legend(handles=[wig_plot,opt_plot,born_plot])
    plt.show()

    state = evolve(np.kron(makeState('0'), makeState('T')), makeGate('+C'))
    p_born = np.trace(np.dot(state, makeState('++'))).real

    print('----------------------------------------------------------------')
    show_circuit(circuit)
    print('Wigner sample p    =', out_wig)
    print('Optimised sample p =', out_opt)
#    print('Exact Born p       =', p_born)
    print('----------------------------------------------------------------')

    return 0

test_sampling(1e5)
#qudit_num = 5
#C1qGate_num = 25
#TGate_num = 5
#CSUMGate_num = 12
#test_sampling2(qudit_num, C1qGate_num, TGate_num, CSUMGate_num,1e4)
