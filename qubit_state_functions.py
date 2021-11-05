import numpy as np
np.set_printoptions(precision=4, suppress=True)

############################ Phase space dimension ############################
DIM   = 2
tau   = np.exp(1.j*np.pi/DIM)
omega = tau*tau
ksi   = np.exp(np.pi*1.j/4)
###############################################################################

def inverse(a, dim):
    ''' Calculates multiplicative inverse of number a (mod d).
    '''
    if a==0:
        return 0
    cand = np.arange(dim)
    inverses = (cand*a)%dim
    return cand[inverses==1][0]

def power(a, p):
    ''' Calculates the matrix power a**p for given 2d-array a and non-negative
        integer p.
    '''
    if p==0:
        return np.eye(a.shape[0])
    elif p==1:
        return a
    else:
        return np.dot(a, power(a, (p-1)))

def chi(q, dim):
    return np.exp(1.j*2.*np.pi*q/dim)

def element(k, b, dim):
    ''' Creates unit matrix element for a 1-copy subsystem.
    '''
    element = np.zeros((dim, dim))
    element[k%dim,b%dim] = 1
    return element

def psi2rho(psi):
    ''' Converts a pure state psi (1d numpy array) to a density operator (2d
    numpy array).
    '''
    psi = np.array(psi)
    rho = np.outer(psi, np.conjugate(psi))
    if (np.isclose(np.real(np.trace(rho)), 1) and
        np.isclose(np.imag(np.trace(rho)), 0)):
        return rho
    else:
        raise Exception('The trace of the state is {0:.2f}'.format(
                         np.trace(rho)))

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

def evolve(X, U, is_state=True):
    ''' Returns UXU^\dagger if is_state is True else U^\daggerXU.
    '''
    (U, V) = (U, U.conj().T) if is_state else (U.conj().T, U)
    return np.dot(U, np.dot(X, V))

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