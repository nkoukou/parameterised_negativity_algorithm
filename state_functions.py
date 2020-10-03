import numpy as np
from math import isclose
np.set_printoptions(precision=4, suppress=True)

ksi = np.exp(2*np.pi*1.j/9)

# State functions
def psi2rho(psi):
    ''' Converts a pure state psi (1d numpy array) to a density operator (2d
    numpy array).
    '''
    psi = np.array(psi)
    rho = np.outer(psi, np.conjugate(psi))
    if (isclose(np.real(np.trace(rho)), 1) and
        isclose(np.imag(np.trace(rho)), 0)):
        return rho
    else:
        raise Exception('The trace of the state is {0:.2f}'.format(
                         np.trace(rho)))

### Important magic states
strange = psi2rho( 1/np.sqrt(2) * np.array([0, 1, -1])      )
norrell = psi2rho( 1/np.sqrt(6) * np.array([-1, 2, -1])     )
tmagik  = psi2rho( 1/np.sqrt(2) * np.array([ksi, 1, 0])     )
tmagic  = psi2rho( 1/np.sqrt(3) * np.array([ksi, 1, 1/ksi]) )
###

def maxmixed(n):
    return 1/n * np.eye(n)

def maxcoh(n):
    return 1/n * np.ones((n,n))

def thermal(H, beta):
    '''
    H    - 1d numpy array - Hamiltonian
    beta - float          - inverse temperature
    '''
    pops = np.exp(-beta*H)
    return np.diag(pops/pops.sum())

def copy(rho, sigma):
    '''
    rho - 2d numpy array - density operator
    Creates product state rho \otimes sigma if sigma is a density operator.
    Creates n-copy state rho^{\otimes n} if sigma = n, where n is a float.
    '''
    if type(sigma)==int:
        if sigma==1:
            return rho
        else:
            return np.kron(rho, copy(rho, sigma-1))
    else:
        return np.kron(rho, sigma)

def church(rho, n):
    ''' Extends density operator rho to dimension n.
    '''
    diff = n - rho.shape[0]
    if diff<0:
        raise Exception('Given n must not be less than dimension of rho')
    rho = np.pad(rho, ((0,diff),(0,diff)), 'constant')
    return rho

def mix(states, probs):
    '''
    states - list of density operators
    probs  - list of floats (must sum to 1)
    '''
    if not np.isclose(sum(probs), 1):
        raise Exception('Mixing probs do not sum up to 1.')
    statef = 0.+0.j
    for i in range(len(probs)):
        statef += np.real(probs[i])*states[i]
    return statef

def addNoise(rho, eps):
    ''' Returns mixture of rho with eps amount of the maximally mixed state.
    '''
    return (1-eps)*rho + eps*maxmixed(rho.shape[0])

def isRho(rho, off=1.0e-9):
    ''' Checks if rho is a valid quantum state.
    '''
    rho = np.array(rho)

    tr = np.trace(rho)
    isTrOne = np.isclose(tr, 1)
    if not isTrOne:
        raise Exception('State has trace {0:.2f}'.format(tr))

    isHerm = np.all(np.isclose(rho - rho.T.conj(), 0))
    if not isHerm:
        raise Exception('State is not Hermitian')

    eigs = np.linalg.eig(rho)[0]
    isEigsReal = np.all(np.isclose(np.imag(eigs), np.zeros(eigs.size)))
    isEigsPos = np.all(np.real(eigs)+off>0)
    isPos = isEigsReal and isEigsPos
    if not isPos:
        raise Exception('State is not positive \n Eigvs: {0}'.format(eigs))

    return 'rho is a valid quantum state'