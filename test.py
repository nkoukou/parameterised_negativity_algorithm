import numpy as np
from circuit_components import(makeGate)
from random_circuit_generator import(random_circuit, show_circuit)
from opt_neg import(optimize_neg)
from prob_estimation import(sample)

# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt


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

# circuit = random_circuit(qudit_num=2,
#                           C1qGate_num=0, TGate_num=1, CSUMGate_num=1,
#                           given_state=None,
#                           given_measurement=2)
# show_circuit(circuit)
# optimize_neg(circuit)



def test_sampling(N=1e1):
    N = int(N)
    tests = []

    circuit = ['0', [[[0], 'H']], '0']
    out = np.zeros(len(circuit[0]))
    for i in range(N):
        out += sample(circuit, x=None)
    out /=N
    tests.append(np.allclose(out, 1/3, atol=1e-2))

    circuit = ['0T', [[[1,0], 'C+']], '01']

    return tests

    # Test T gate
    # Test CSUM and T

