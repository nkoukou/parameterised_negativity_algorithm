import numpy as np
import matplotlib.pylab as plt
plt.rcParams['figure.dpi'] = 200

from scipy.linalg import expm
from QUBIT_Pauli_sampling import *
from QUBIT_circuit_components import*

I = np.eye(2)
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1.j],[1.j,0]])
Z = np.array([[1,0],[0,-1]])

CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])

def Rotx(th):
    return np.cos(th/2)*I + 1.j*np.sin(th/2)*X
def Roty(th):
    return np.cos(th/2)*I + 1.j*np.sin(th/2)*Y
def Rotz(th):
    return np.cos(th/2)*I + 1.j*np.sin(th/2)*Z
def RandU(offset=1):
    return np.dot(np.dot(Rotx(offset*np.pi*np.random.rand()),Roty(offset*np.pi*np.random.rand())),Rotz(offset*np.pi*np.random.rand()))

def RandU2q(offset=1):
    return np.dot(np.dot(np.kron(RandU(offset),RandU(offset)),CNOT),np.kron(RandU(offset),RandU(offset)))

def RandU2q():
    H = np.pi*np.random.rand(4,4)
    A = np.pi*np.random.rand(4,4)
    U = expm(1.j*(H+H.T + 1.j*(A- A.T)))
    return U

def RandU2qZ():
    R1 = np.kron(Rotz(np.pi*np.random.rand()),Rotz(np.pi*np.random.rand()))
    R2 = np.kron(Rotz(np.pi*np.random.rand()),Rotz(np.pi*np.random.rand()))
    return np.dot(np.dot(R1,CNOT),R2)

from autograd import(grad)
from scipy.optimize import(Bounds, minimize)
from scipy.optimize import(basinhopping)
import time


def optimizer(cost_function, x0, opt_method='B', niter = 10):
    start_time = time.time()

    if opt_method=='B': # autograd
        grad_cost_function = grad(cost_function)
        def func(x):
            return cost_function(x), grad_cost_function(x)
        optimize_result = basinhopping(func, x0, minimizer_kwargs={"method":"L-BFGS-B","jac":True},niter=niter)

    elif opt_method=='NG': # Powell
        optimize_result = minimize(cost_function, x0, method='Powell')

    elif opt_method=='G': # Without autograd
        optimize_result = minimize(cost_function, x0, method='L-BFGS-B',jac=grad_cost_function)

    else:
        raise Exception('Invalid optimisation method')

    dt = time.time()-start_time

    return optimize_result, dt

U1 = RandU2q()
U2 = RandU2q()

T1 = T_matrix_2q(U1)
T2 = T_matrix_2q(U2)

neg_U1 = max_negativity(T1)
neg_U2 = max_negativity(T2)

print('neg_U1:',neg_U1)
print('neg_U2:',neg_U2)
print('neg_Utot:',neg_U1*neg_U2)
print('\n')

def ct(x):
    return max_negativity(RTR_matrix_2q(T1,[0,0,0,0,0,0],[x[0],x[1],x[2],0,0,0]))

x0 = [0,0,0]
optimize_result, dt = optimizer(ct, x0)
opt_x = optimize_result.x
print('opt_x:',opt_x)
opt_value = ct(opt_x)
# print(opt_value)
# print(ct(opt_x))

neg_opt_U1 = max_negativity(RTR_matrix_2q(T1,[0,0,0,0,0,0],[opt_x[0],opt_x[1],opt_x[2],0,0,0]))
neg_opt_U2 = max_negativity(RTR_matrix_2q(T2,[0,0,0,opt_x[0],opt_x[1],opt_x[2]],[0,0,0,0,0,0]))

print('neg_opt_U1:',neg_opt_U1)
print('neg_opt_U2:',neg_opt_U2)
print('neg_opt_Utot:',neg_opt_U1*neg_opt_U2)
print('\n')

def ct2(x):
    return max_negativity(RTR_matrix_2q(T1,[0,0,0,0,0,0],[x[0],x[1],x[2],0,0,0]))*max_negativity(RTR_matrix_2q(T2,[0,0,0,x[0],x[1],x[2]],[0,0,0,0,0,0]))

x0 = [0,0,0]
optimize_result, dt = optimizer(ct2, x0)
opt_x = optimize_result.x
print('opt_x:',opt_x)
opt_value = ct2(opt_x)
# print(opt_value)
# print(ct(opt_x))

neg_opt_U1 = max_negativity(RTR_matrix_2q(T1,[0,0,0,0,0,0],[opt_x[0],opt_x[1],opt_x[2],0,0,0]))
neg_opt_U2 = max_negativity(RTR_matrix_2q(T2,[0,0,0,opt_x[0],opt_x[1],opt_x[2]],[0,0,0,0,0,0]))

print('neg_opt_U1:',neg_opt_U1)
print('neg_opt_U2:',neg_opt_U2)
print('neg_opt_Utot:',neg_opt_U1*neg_opt_U2)
print('\n')


Utot = np.dot(np.kron(U2,I),np.kron(I,U1))
Ttot = T_matrix_3q(Utot)
neg_Utot = max_negativity(Ttot)
print('neg_Utot',neg_Utot)

def compare_neg():
    U1 = RandU2q()
    U2 = RandU2q()

    T1 = T_matrix_2q(U1)
    T2 = T_matrix_2q(U2)

    neg_U1 = max_negativity(T1)
    neg_U2 = max_negativity(T2)

    def ct(x):
        return max_negativity(RTR_matrix_2q(T1,[0,0,0,0,0,0],[x[0],x[1],x[2],0,0,0]))
    
    x0 = [0,0,0]
    optimize_result, dt = optimizer(ct, x0)
    opt_x = optimize_result.x

    neg_opt_U1 = max_negativity(RTR_matrix_2q(T1,[0,0,0,0,0,0],[opt_x[0],opt_x[1],opt_x[2],0,0,0]))
    neg_opt_U2 = max_negativity(RTR_matrix_2q(T2,[0,0,0,opt_x[0],opt_x[1],opt_x[2]],[0,0,0,0,0,0]))
    
    def ct2(x):
        return max_negativity(RTR_matrix_2q(T1,[0,0,0,0,0,0],[x[0],x[1],x[2],0,0,0]))*max_negativity(RTR_matrix_2q(T2,[0,0,0,x[0],x[1],x[2]],[0,0,0,0,0,0]))

    x0 = [0,0,0]
    optimize_result, dt = optimizer(ct2, x0)
    opt_x2 = optimize_result.x
    
    neg_opt_U1_2 = max_negativity(RTR_matrix_2q(T1,[0,0,0,0,0,0],[opt_x2[0],opt_x2[1],opt_x2[2],0,0,0]))
    neg_opt_U2_2 = max_negativity(RTR_matrix_2q(T2,[0,0,0,opt_x2[0],opt_x2[1],opt_x2[2]],[0,0,0,0,0,0]))

    Utot = np.dot(np.kron(U2,I),np.kron(I,U1))
    Ttot = T_matrix_3q(Utot)
    neg_Utot = max_negativity(Ttot)
    
    return neg_U1*neg_U2, neg_opt_U1*neg_opt_U2, neg_opt_U1_2*neg_opt_U2_2, max_negativity(Ttot)

def compare_negZ():
    U1 = RandU2qZ()
    U2 = RandU2qZ()

    T1 = T_matrix_2q(U1)
    T2 = T_matrix_2q(U2)

    neg_U1 = max_negativity(T1)
    neg_U2 = max_negativity(T2)

    def ct(x):
        return max_negativity(RTR_matrix_2q(T1,[0,0,0,0,0,0],[x[0],x[1],x[2],0,0,0]))

    x0 = [0,0,0]
    optimize_result, dt = optimizer(ct, x0)
    opt_x = optimize_result.x
    
    neg_opt_U1 = max_negativity(RTR_matrix_2q(T1,[0,0,0,0,0,0],[opt_x[0],opt_x[1],opt_x[2],0,0,0]))
    neg_opt_U2 = max_negativity(RTR_matrix_2q(T2,[0,0,0,opt_x[0],opt_x[1],opt_x[2]],[0,0,0,0,0,0]))

    return neg_U1*neg_U2, neg_opt_U1*neg_opt_U2

x_list = []
y1_list = []
y2_list = []
y3_list = []
for index in range(100):
    data = compare_neg()
    x_list.append(data[0])
    y1_list.append(data[1])
    y2_list.append(data[2])
    y3_list.append(data[3])
    
plt.plot(x_list,x_list)
plt.plot(x_list,y1_list,linewidth=0,marker='+',markersize=5,label='y1')
plt.plot(x_list,y2_list,linewidth=0,marker='*',markersize=5,label='y2')
plt.plot(x_list,y3_list,linewidth=0,marker='.',markersize=5,label='y3')
plt.legend()
plt.show()

plt.plot(y2_list,y2_list,linewidth=0,marker='+',markersize=5,label='y1')
plt.plot(y2_list,y3_list,linewidth=0,marker='+',markersize=5,label='y1')
plt.show()

log_diff_list1 = np.log2(np.array(x_list)/np.array(y1_list))
log_diff_list2 = np.log2(np.array(x_list)/np.array(y2_list))
log_diff_list3 = np.log2(np.array(x_list)/np.array(y3_list))

plt.hist(np.log2(np.array(x_list)),20,label='x')
plt.hist(np.log2(np.array(y1_list)),20,label='y1')
plt.hist(np.log2(np.array(y2_list)),20,label='y2')
plt.hist(np.log2(np.array(y3_list)),20,label='y3')
plt.legend()
# plt.hist(log_diff_list)
plt.show()