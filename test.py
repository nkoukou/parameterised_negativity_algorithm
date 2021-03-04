import numpy as np
from state_functions import(evolve)
from circuit_components import(makeGate, makeState)
from random_circuit_generator import(random_circuit, compress2q_circuit,
                                     show_circuit)
from opt_neg import(optimize_neg, optimize_neg_compressed)
from prob_estimation import(sample, compare_Wigner_optimised)

# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


circuit = random_circuit(qudit_num=10,
                         C1qGate_num=100, TGate_num=20, CSUMGate_num=15,
                         given_state=0, given_measurement=2)
# circuit = ['011', [
#             [[2], 'H'],
#             [[0], 'S'],
#             [[1], 'T'],
#             [[0], 'T'],
#             [[2, 0], 'C+'],
#             [[1], 'H'],
#             [[1], 'S'],
#             [[0], 'T'],
#             [[0, 2], 'C+'],
#             [[0], 'S'],
#             [[1], 'T'],
#             [[0], 'H'],
#             [[1], 'S'],
#             [[1, 2], 'C+'],
#             [[1], 'H']
#             ], '1T/']
# show_circuit(circuit)
# circuit_compressed = compress2q_circuit(circuit)
# test = compress2q_circuit(circuit_compressed)
# # optimize_neg(circuit)
# print('¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬¬')
# optimize_neg_compressed(circuit_compressed)

# # p= sample(circuit, 0)
# # q= sample(circuit, np.load('data/test_directory/optimized_x.npy'))

def string_to_circuit(circuit_string):
    circuit_compressed = compress2q_circuit(circuit_string)
    state_string_list = circuit_compressed[0]
    gate_compressed_list = circuit_compressed[1]
    meas_string_list = circuit_compressed[2]

    rho_list = []
    for state_string in state_string_list:
        rho_list.append(makeState(state_string))

    gate_U2q_list = []
    gate_qudit_index_list = []
    for gate_compressed in gate_compressed_list:
        gate_qudit_index_list.append(gate_compressed[0])
        gate_U2q_list.append(gate_compressed[1])

    meas_list = []
    for meas_string in meas_string_list:
        if meas_string=='/':
            continue
        E = makeState(meas_string)
        meas_list.append(E)

    circuit = {'state_list': rho_list, 'gate_list': gate_U2q_list,
               'qudit_index_list': gate_qudit_index_list,
               'meas_list': meas_list}

    return circuit

# def test_solver():
#     check1 = solve_circuit_symbolic(circuit)

#     state, gates, meas = circuit_compressed

#     return check1, state, gates, meas


# def test_sampling(N=1e1):
#     N = int(N)
#     path = 'data/test_directory/optimized_x.npy'

#     # # Test simple Clifford
#     # circuit = ['0', [[[0], 'H']], '0']
#     # out_wig, out_wig_list = sample(circuit, x=0, niters=N)

#     # p_born = 1/3

#     # print('--------------------------------------------------------------')
#     # show_circuit(circuit)
#     # print('Wigner sample p    =', out_wig)
#     # print('Exact Born p       =', p_born)
#     # print('--------------------------------------------------------------')

#     # Test T state & 2-qutrit measurement
#     # circuit = ['0T', [[[1,0], 'C+']], '++']
#     # out_wig, out_wig_list = sample(circuit, x=0, niters=N)
#     # optimize_neg(circuit)
#     # out_opt, out_opt_list = sample(circuit, x=np.load(path), niters=N)

#     # state = evolve(np.kron(makeState('0'), makeState('T')), makeGate('+C'))
#     # p_born = np.trace(np.dot(state, makeState('++'))).real

#     # wig_plot, = plt.plot(out_wig_list,'r',label = 'Wigner-sampling')
#     # opt_plot, = plt.plot(out_opt_list,'b',label = 'Opt-sampling')
#     # born_plot, = plt.plot(np.ones(len(out_wig_list))*p_born,'k-.',
#     #                        label = 'Born rule probability')
#     # plt.xlabel("Iterations")
#     # plt.ylabel("Probability")
#     # plt.legend(handles=[wig_plot,opt_plot,born_plot])
#     # plt.show()

#     # print('--------------------------------------------------------------')
#     # show_circuit(circuit)
#     # print('Wigner sample p    =', out_wig)
#     # print('Optimised sample p =', out_opt)
#     # print('Exact Born p       =', p_born)
#     # print('--------------------------------------------------------------')

#     # Test complicated circuit
#     circuit = ['1T0', [[[0], 'S'], [[1], 'H'], [[1], 'T'], [[2], 'H'],
#                        [[1, 0], 'C+'], [[1], 'T'], [[1, 0], 'C+'],
#                        [[2], 'H'], [[2], 'H'], [[2, 1], 'C+'], [[2], 'S']],
#                '10/']
#     out_wig, out_wig_list = sample(circuit, x=0, niters=N)
#     optimize_neg(circuit)
#     out_opt, out_opt_list = sample(circuit, x=np.load(path), niters=N)

#     state = makeState('1T0')
#     state = evolve(state, makeGate('SHH'))
#     state = evolve(state, makeGate('1TH'))
#     state = evolve(state, makeGate('+C1'))
#     state = evolve(state, makeGate('11H'))
#     state = evolve(state, makeGate('1T1'))
#     state = evolve(state, makeGate('+C1'))
#     state = evolve(state, makeGate('1+C'))
#     state = evolve(state, makeGate('11S'))
#     p_born = np.trace(np.dot(state, makeState('10m')*3)).real

#     wig_plot, = plt.plot(out_wig_list,'r',label = 'Wigner-sampling')
#     opt_plot, = plt.plot(out_opt_list,'b',label = 'Opt-sampling')
#     born_plot, = plt.plot(np.ones(len(out_wig_list))*p_born,'k-.',
#                           label = 'Born rule probability')
#     plt.xlabel("Iterations")
#     plt.ylabel("Probability")
#     plt.legend(handles=[wig_plot,opt_plot,born_plot])
#     plt.show()

#     print('----------------------------------------------------------------')
#     show_circuit(circuit)
#     print('Wigner sample p    =', out_wig)
#     print('Optimised sample p =', out_opt)
#     print('Exact Born p       =', p_born)
#     print('----------------------------------------------------------------')

#     return 0

# def test_sampling2(qudit_num, C1qGate_num, TGate_num, CSUMGate_num,N=1e2):
#     N = int(N)
#     path = 'data/test_directory/optimized_x.npy'

#     circuit = random_circuit(qudit_num, C1qGate_num, TGate_num, CSUMGate_num)

#     out_wig, out_wig_list = sample(circuit, x=0, niters=N)
#     optimize_neg(circuit)
#     out_opt, out_opt_list = sample(circuit, x=np.load(path), niters=N)

#     wig_plot, = plt.plot(out_wig_list,'r',label = 'Wigner-sampling')
#     opt_plot, = plt.plot(out_opt_list,'b',label = 'Opt-sampling')
#     # born_plot, = plt.plot(np.ones(len(out_wig_list))*p_born,'k-.',
#     #                       label = 'Born rule probability')
#     plt.xlabel("Iterations")
#     plt.ylabel("Probability")
#     # plt.legend(handles=[wig_plot, opt_plot, born_plot])
#     plt.show()

#     state = evolve(np.kron(makeState('0'), makeState('T')), makeGate('+C'))
#     p_born = np.trace(np.dot(state, makeState('++'))).real

#     print('----------------------------------------------------------------')
#     show_circuit(circuit)
#     print('Wigner sample p    =', out_wig)
#     print('Optimised sample p =', out_opt)
# #    print('Exact Born p       =', p_born)
#     print('----------------------------------------------------------------')

#     return 0

# test_sampling(1e5)
#qudit_num = 5
#C1qGate_num = 25
#TGate_num = 5
#CSUMGate_num = 12
#test_sampling2(qudit_num, C1qGate_num, TGate_num, CSUMGate_num,1e4)
