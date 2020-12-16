import autograd.numpy as np
from circuit_components import(makeGate)
from opt_neg import(optimize_neg)

# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt

def show_circuit(circuit_string):
    state_string = circuit_string[0]
    gate_sequence = circuit_string[1:-1]
    meas_string = circuit_string[-1]
    print(*state_string,sep='   ')
    for gate_string in gate_sequence:
        print(*gate_string,sep='   ')
    print(*meas_string,sep='   ')
    return 0

TGate = makeGate('T')
CSUM = makeGate('C+')
def random_gate_sequence(qudit_num, gate_num):
    gate_sequence = []
    for index in range(gate_num):
        if np.random.randint(2) == 0:
            gate_sequence.append([[np.random.randint(qudit_num)],TGate])
        elif np.random.randint(2) == 1:
            [index1, index2] = np.random.choice(range(qudit_num), size=2,
                                                replace=False)
            gate_sequence.append([[index1,index2],CSUM])
    return gate_sequence

qudit_num = 3
gate_num = 15
state_string = '0TT' #'TTTTTTT'
gate_sequence = random_gate_sequence(qudit_num, gate_num)
#print(*gate_sequence,sep='\n')
meas_string = '0111111'
optimize_neg(state_string, gate_sequence, meas_string, path='test00')

optimize_neg('TTTT', [[[0],TGate ],
                      [[0,1],CSUM],
                      [[3],TGate ],
                      [[0,3],CSUM],
                      [[0],TGate ],
                      [[1,2],CSUM],
                      [[1],TGate ],
                      [[2,0],CSUM]
                     ],'0010')


'''Sample codes'''
# DIM = 3
# x_range = list(range(DIM))
# p_range = list(range(DIM))
# rho1 = makeState1q('T')
# rho2 = makeState1q('0')
# IGate = makeGate('1')

#rho = makeState1q('T')
#U = TGate
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
# def neg_circuit_CSUM(rho1,rho2,E,GammaC_in,GammaT_in,GammaC_out,GammaT_out):
#     return np.real(neg_state_1q(rho1,GammaC_in)*neg_state_1q(rho2,GammaT_in
#                    )*neg_gate_CSUM(GammaC_in,GammaT_in,GammaC_out,GammaT_out
#                    )*neg_meas_1q(E,GammaC_out) * neg_meas_1q(E,GammaT_out))
#
#def cost_function(x):
#    Gamma1_in = x2Gamma(x[0:8])
#    Gamma2_in = x2Gamma(x[8:16])
#    Gamma1_out = x2Gamma(x[16:24])
#    Gamma2_out = x2Gamma(x[24:32])
    # return np.real(neg_circuit_CSUM(rho1,rho2,E,Gamma1_in,Gamma2_in,
    #                                 Gamma1_out,Gamma2_out))
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
# optimize_result = basinhopping(func, x0,
#   minimizer_kwargs={"method":"L-BFGS-B", "jac":True}, disp=False, niter=1)
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
## print('Gate: \n',W_gate_2q(CSUM,Gamma1_in_opt,Gamma2_in_opt,Gamma1_out_opt,
##                            Gamma2_out_opt))
#print('Meas1: \n',W_meas_1q(E,Gamma1_out_opt))
#print('Meas2: \n',W_meas_1q(E,Gamma2_out_opt))


'''Single qudit example'''
#rho_string = 'T'
#rho = makeState1q(rho_string)
#U = IGate
#E = makeState1q('0')
#
# def neg_circuit_1q(rho,U,E,Gamma_in,Gamma_out):
#     return np.real(neg_state_1q(rho,Gamma_in
#                     ) * neg_gate_1q(U,Gamma_in,Gamma_out
#                     ) * neg_meas_1q(E,Gamma_out))
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
# optimize_result = basinhopping(func, x0,
#    minimizer_kwargs={"method":"L-BFGS-B", "jac":True}, disp=False, niter=1)
##optimize_result = minimize(cost_function, x0, method='Powell')
# optimize_result = minimize(cost_function, x0, method='L-BFGS-B',
#                            jac=grad_cost_function)
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
# def neg_circuit(rho1,rho2,rho3,E,Gamma01,Gamma02,Gamma03,
#                 Gamma11,Gamma12,Gamma21,Gamma22,Gamma31,Gamma32,Gamma33):
#     outcome = 1.
#     outcome = outcome*neg_state_1q(rho1,Gamma01)*neg_state_1q(rho2,Gamma02
#                       )*neg_state_1q(rho3,Gamma03)
#     outcome = outcome*neg_gate_CSUM(Gamma01,Gamma02,Gamma11,Gamma12
#                       )*neg_gate_CSUM(Gamma12,Gamma03,Gamma22,Gamma33
#                       )*neg_gate_1q(TGate,Gamma11,Gamma21
#                       )*neg_gate_CSUM(Gamma22,Gamma21,Gamma32,Gamma31)
#     outcome = outcome*neg_meas_1q(E,Gamma31)
#     return outcome
#
# def cost_function(x):
#     Gamma01 = x2Gamma(x[0:8])
#     Gamma02 = x2Gamma(x[8:16])
#     Gamma03 = x2Gamma(x[16:24])
#     Gamma11 = x2Gamma(x[24:32])
#     Gamma12 = x2Gamma(x[32:40])
#     Gamma21 = x2Gamma(x[40:48])
#     Gamma22 = x2Gamma(x[48:56])
#     Gamma31 = x2Gamma(x[56:64])
#     Gamma32 = x2Gamma(x[64:72])
#     Gamma33 = x2Gamma(x[72:80])
#     return neg_circuit(rho1,rho2,rho3,E,Gamma01,Gamma02,Gamma03,
#               Gamma11,Gamma12,Gamma21,Gamma22,Gamma31,Gamma32,Gamma33)
#
# x0 = [1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,
#       0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,
#       1,0,1,0,0,0,0,0,1,0]
#t0=time.time()
#cost_function(x0)
#print('Wigner Negativity:\t', cost_function(x0))
#print('Computation time:\t', time.time()-t0)

'''Optimization with autograd'''
#grad_cost_function = grad(cost_function)
#def func(x):
#    return cost_function(x), grad_cost_function(x)
# x0 = [1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,
#       0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,
#       1,0,1,0,0,0,0,0,1,0]
#t0=time.time()
# optimize_result = basinhopping(func, x0,
#   minimizer_kwargs={"method":"L-BFGS-B", "jac":True}, disp=False, niter=1)
#dt = time.time()-t0

'''Optimization without autograd (Powell)'''
# x0 = [1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,
#       0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,
#       1,0,1,0,0,0,0,0,1,0]
#t0=time.time()
#optimize_result = minimize(cost_function, x0, method='Powell')
#dt = time.time()-t0

'''Optimization without autograd (G)'''
#grad_cost_function = grad(cost_function)
#def func(x):
#    return cost_function(x), grad_cost_function(x)
# x0 = [1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,
#       0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,
#       1,0,1,0,0,0,0,0,1,0]
#x0 = 2*np.random.rand(80)-1
#t0=time.time()
# optimize_result = minimize(cost_function, x0, method='L-BFGS-B',
#                             jac=grad_cost_function)
#dt = time.time()-t0
'''Show results'''
#optimized_x = optimize_result.x
#optimized_value = cost_function(optimized_x)
#
#print('Optimized negativity: ',cost_function(optimized_x))
#print('Computation time: ',dt)
#print('Optimized x: ',optimized_x)

