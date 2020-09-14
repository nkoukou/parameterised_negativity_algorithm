import numpy as np
import itertools as it
import matplotlib.pylab as plt
np.set_printoptions(precision=4, suppress=True)

class CopyPhaseSpace(object):
    '''
    Represents a phase space for a n-copy system of prime dimension dim
    according to "Veitch et.al. (2012) Negative quasi-probability as a
    resource for quantum computation" and similar literature.
    '''
    def __init__(self, dim, n):
        ''' dim: dimension - integer
            xp: phase space coordinates - list or tuple
        '''
        if isinstance(dim, inttypes):
            if (factors(dim).size==1):
                self.dim = dim
            else:
                print(dim)
                raise Exception('Given dimension dim is not prime.')
        else:
            print(dim)
            raise Exception('Given dimension dim is not integer.')
        self.n = n

        self.tau = -np.exp(1j*np.pi/self.dim)
        self.omega = self.tau * self.tau
        self.X = np.roll(np.eye(self.dim), 1, axis=0)
        self.Z = np.diag(self.omega**np.arange(self.dim))
        self.A0 = np.roll(np.eye(self.dim)[::-1], 1, axis=0
                          ) if self.dim>2 else A0d2

    def ele1q(self, k, b):
        ''' Creates unit matrix element for a 1-copy subsystem.
        '''
        element = np.zeros((self.dim, self.dim))
        element[k%self.dim,b%self.dim] = 1
        return element

    def inverse(self, a):
        ''' Calculates multiplicative inverse of number a (mod d).
        '''
        if a==0:
            return 0
        cand = np.arange(self.dim)
        inverses = (cand*a)%self.dim
        return cand[inverses==1][0]

    def errpos(self, xp, c=1):
        if len(xp)!=c*self.n:
          raise Exception('An n-copy system takes n positional paramenters.')
        else:
          return

    def D(self, xp):
        ''' Calculates displacement operator D at point xp.
        '''
        self.errpos(xp, 2)
        D = 1
        for i in range(self.n):
            x, p = xp[2*i], xp[2*i+1]
            T = self.tau**(x*p) * np.dot(power(self.X, x), power(self.Z, p))
            D = np.kron(D, T)
        return D

    def A(self, xp):
        ''' Calculates phase-space point operator A at point xp.
        '''
        self.errpos(xp, 2)
        A = 1
        for i in range(self.n):
            x, p = xp[2*i], xp[2*i+1]
            T = self.tau**(x*p) * np.dot(power(self.X, x), power(self.Z, p))
            a = evolve(self.A0, T)
            A = np.kron(A, a)
        return A

    def D1q(self, xp):
        ''' Calculates displacement operator D at point xp for a 1-copy
            subsystem.
        '''
        x, p = xp[0], xp[1]
        return self.tau**(x*p) * np.dot(power(self.X, x), power(self.Z, p))

    def A1q(self, xp):
        ''' Calculates phase-space point operator at point xp for a 1-copy
            subsystem.
        '''
        return evolve(self.A0, self.D1q(xp))

    def line1q(self, a, k):
        ''' Returns list of phase space points xp that satisfy the line
            equation a*xp = k for a 1-copy subsystem.
        '''
        points = []
        for i in range(self.dim):
            if a[1]==0:
                point = (k,i)
            else:
                ai = self.inverse(a[1])
                point = (i, (-ai*a[0]*i+ai*k)%self.dim)
            points.append(point)
        return points

    def stabilizer(self, aa, kk):
        ''' Returns the stabilizer state with Wigner function the uniform
        distribution over phase space points xp on the line aa[i]*xp = kk[i]
        for i in range(self.n).
        '''
        self.errpos(aa)
        self.errpos(kk)

        stab = 1
        for i in range(self.n):
            a = aa[i]
            k = kk[i]
            stab_line1q = self.line1q(a,k)

            stab1q = 0
            for point in stab_line1q:
                stab1q += 1/self.dim * self.A1q(point)
            stab = np.kron(stab, stab1q)
        return stab

    def clifford(self, FF, zz):
        ''' Returns the Clifford unitary that corresponds to symplectic
        rotation F and translation z on the phase space.
        '''
        self.errpos(FF)
        self.errpos(zz)

        N = self.dim
        cliff = 1
        for i in range(self.n):
            z = tuple(zz[i])
            a,b,c,d = (np.array(FF[i])).flatten()

            tot = 0
            if np.isclose(b,0):
                for k in range(N):
                    exp = a*c*k*k
                    tot += self.tau**exp * self.ele1q((a*k)%N, k)
            else:
                for j in range(N):
                    for k in range(N):
                        exp = (a*k*k - 2*j*k + d*j*j)*self.inverse(b)
                        tot += self.tau**exp * self.ele1q(j, k)
                tot *= 1/np.sqrt(N)
            cliff1q = np.matmul(self.D1q(z), tot)
            cliff = np.kron(cliff, cliff1q)
        return cliff

    def csum(self, c, t):
        ''' Returns a CSUM unitary with the c-th qudit as control and the t-th
        qudit as target.
        '''
        s = 0
        if c<=t:
            A0 = np.eye(self.dim**(c))
            A1 = np.eye(self.dim**(self.n-1-t))
            for i in range(self.dim):
                for j in range(self.dim):
                    temp = np.kron( self.ele1q(i,i),
                                    np.eye(self.dim**(t-c-1)) )
                    s += np.kron( temp, self.ele1q(j+i, j) )
        else:
            A0 = np.eye(self.dim**(t))
            A1 = np.eye(self.dim**(self.n-1-c))
            for i in range(self.dim):
                for j in range(self.dim):
                    temp = np.kron( self.ele1q(j+i,j),
                                    np.eye(self.dim**(c-t-1)) )
                    s += np.kron( temp, self.ele1q(i, i) )

        return np.kron(A0, np.kron(s, A1) )

    def wstate(self, state):
        xcoords = it.product(*([range(self.dim)]*(2*self.n)))
        wig = []
        for x in xcoords:
            w = 1/self.dim**self.n * np.real(
                np.trace(np.dot(state, self.A(x))) )
            if np.isclose(w,0): w = 0
            wig.append(w)
        wig = np.array(wig)
        return wig

    def wgate(self, gate):
        xcoords = it.product(*([range(self.dim)]*(2*self.n)))
        wig = []
        for x in xcoords:
            ycoords = it.product(*([range(self.dim)]*(2*self.n)))
            for y in ycoords:
                Aev = evolve(self.A(x), gate)
                w = 1/self.dim**self.n * np.real(
                    np.trace(np.dot(self.A(y), Aev)) )
                if np.isclose(w,0): w = 0
                wig.append(w)
        wig = np.array(wig).reshape((self.dim**(2*self.n),
                                     self.dim**(2*self.n))).T
        return wig

# Auxiliary classes, variables and functions
inttypes = (int, np.integer)
A0d2 = np.array([[1, (1+1j)/2],[(1-1j)/2, 0]])

def evolve(X, U):
    ''' Returns UXU^\dagger.
    '''
    return np.dot(U, np.dot(X, U.conj().T))

def power(a, p):
    ''' Returns a^p.
    '''
    if p==0:
        return np.eye(a.shape[0])
    elif p==1:
        return a
    else:
        return np.dot(a, power(a, p-1))

def factors(n):
    ''' Returns list of all non-distinct prime factors of n.
    '''
    m = 2
    factors = []
    while m <= n:
        if n%m==0:
            factors.append(m)
            n/= m
        else:
            m+= 1
    return np.array(factors)

plt.rc('axes', labelsize=18)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.rc('legend', fontsize=14)
plt.rc('lines', linewidth=1)
plt.rc('lines', markersize=7)