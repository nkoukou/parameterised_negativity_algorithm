import numpy as np

DIM = 3

# State functions
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

ksi = np.exp(2*np.pi*1.j/9)

def makeState1q(state_string):
    ''' Makes a 1-qudit state matrix from the generating state string:
        '0' - |0><0|
        '1' - |1><1|
        '2' - |2><2|
        '+' - |+><+| (maximally coherent state)
        'm' - 1/d 1  (maximally mixed state)
        'S' - |S><S| (Strange state)
        'N' - |N><N| (Norrell state)
        'T' - |T><T| (T state)
        'M' - |M><M| (M state)
    '''
    if state_string=='0':
        state = psi2rho(np.array([1,0,0]))
    elif state_string=='1':
        state = psi2rho(np.array([0,1,0]))
    elif state_string=='2':
        state = psi2rho(np.array([0,0,1]))
    elif state_string=='+':
        state = psi2rho(np.array([1,1,1])/np.sqrt(3))
    elif state_string=='m':
        state = 1/DIM * np.eye(DIM)
    elif state_string=='S':
        state = psi2rho(1/np.sqrt(2) * np.array([0, 1, -1]))
    elif state_string=='N':
        state = psi2rho(1/np.sqrt(6) * np.array([-1, 2, -1]))
    elif state_string=='T':
        state = psi2rho(1/np.sqrt(3) * np.array([ksi, 1, 1/ksi]))
    else:
        raise Exception('Invalid state string')
    return state

def makeMeas1q(meas_string):
    if meas_string.find('X') != -1:
        MeasO = makeState1q('+')
        Meas_mode = meas_string.find('X')
    if meas_string.find('Z') != -1:
        MeasO = makeState1q('0')
        Meas_mode = meas_string.find('Z')
    if meas_string.find('T') != -1:
        MeasO = makeState1q('T')
        Meas_mode = meas_string.find('T')
    return MeasO, Meas_mode

def makeState(state_string):
    ''' Makes a state matrix from the generating state string,
        e.g. '+', '000SS', '012+TSN'.
    '''
    state = 1
    for s in state_string:
        # print(s)
        temp = makeState1q(s)
        state = np.kron(state, temp)
    return state

def makeMeas(meas_string):
    ''' Makes a measurement projector matrix,
        e.g. 'X11' (projects qudit 0 on state |+>, '11Z1' (projects qudit 2 on
        state |0>).
    '''
    N = len(meas_string)
    MeasO, Meas_mode = makeMeas1q(meas_string)

    idx = np.array([Meas_mode, N - Meas_mode - 1])
    ids = DIM**idx
    measp = np.kron( np.kron(np.eye(ids[0]), MeasO), np.eye(ids[1]) )

    return measp
