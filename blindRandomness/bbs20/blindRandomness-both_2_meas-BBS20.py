"""
    Compute local randomness (min-entropy)
      where both Alce and Bob perform a second measurement,
      and Bob use his second measurement to guess Alice's outcome.
    This program is based on the idea in Fu-Miller's work (Phys. Rev. A 97, 032324)
      and the equivalent sequential measuremnet scenario in
      Bowles-Baccari-Salavrakos's paper (arXiv:1911.11056).
"""
import ncpol2sdpa as ncp
import numpy as np
from joblib import Parallel, delayed
import time
import sys, os

### Add current directory to Python path
sys.path.append('..')
from common_func.SDP_helper import *

CLASS_ZERO_POS = zero_position_map()

# Score (winning probability) constraint of non-local game
def winProb(P, scenario = [[4,4,4,4],[4,4,4,4]], meas_prob = [], random_meas = True, signaling = True):
    """
        Function of winning probability of nonlocal game
        Here we use a modified CHSH non-local game for example
        - P : P([a,b],[x,y])=P(ab|xy) is the probability that get the outputs (a,b) for given inputs (x,y)                                               
        - scenario : Tell the number of inputs and number of outputs for each input in each party
                    may be useful for general scenario
        - meas_prob : The probability of the inpus P(xy)
        - random_meas : Take equal probability for each input combination if this is true
    """
    try:
        configA = scenario[0]
        num_x = len(configA)
        configB = scenario[1]
        num_y = len(configB)
    except IndexError:
        print(f'Wrong input scenario: {scenario}')
    win_prob = 0
    for x in range(num_x):
        for y in range(num_y):
            for a in range(configA[x]):
                for b in range(configB[y]):
                    # One should modify the following line for different winning condtion
                    a1 = int(a/2)
                    b1 = int(b/2)
                    x1 = int(x/2)
                    y1 = int(y/2)
                    if ( a1^b1 == (x1*y1)^1 ) and (y%2 == x1) and (x%2 == y1):
                        # print(f'P({a}{b}|{x}{y})')
                        if not random_meas:
                            try:
                                prob = meas_prob[x1][y1]
                            except IndexError:
                                print(f'Wrong input meas_prob: {meas_prob}')
                        else:
                            prob = 1/4
                        win_prob += P([a,b], [x,y])*prob
    return win_prob

def zeroProbConstr(P, positions, err_tol = 0, on_Pa1b1xy = False):
    """ 
        Set the zero probability constraints for the positions with tolerable error 
    """
    constr = []
    for pos in positions:
        if on_Pa1b1xy:
           constr +=  [err_tol-Pa1b1xy(P, pos)]
        else:
            constr += [err_tol-P(*pos)]
    return constr


def objective(P, inputs = None, target = 0):
    """
       The objective function of guessing probability for given inputs in FM18 scenario with BBS20 method.
       - inputs: Optimize for given inputs.
       - target: 0 for computing P(b2=a1), Bob's guessing probability to a1.
                 1 for computing P(a2=b1), Alice's guessing probability to b1
    """
    A, B = P.parties
    if inputs is None:
        return
    else:
        x, y = inputs
        xy = [x*2+y, y*2+x]
        obj = 0
        if target == 0:
            for a in range(4):
                for b1 in range(2):
                    b = b1*2+int(a/2)
                    obj = obj + P([a,b],xy)
        elif target == 1:
            for b in range(4):
                for a1 in range(2):
                    a = a1*2+int(b/2)
                    obj = obj + P([a,b],xy)
        else:
            print("Wrong input 'target', which should be either 0 or 1")
            return

    return obj

def Pa1b1xy(P, pos):
    a1b1, xy = pos
    a1, b1 = a1b1
    xy = [xy[0]*2+xy[1], xy[1]*2+xy[0]]
    Pa1b1xy = 0
    for a2 in range(2):
        for b2 in range(2):
            ab = [a1*2+a2, b1*2+b2]
            Pa1b1xy += P(ab, xy)
    return Pa1b1xy

def printPa1b1xy(sdp, P):
    for y in range(2):
        for b1 in range(2):
            row = []
            for x in range(2):
                for a1 in range(2):
                    prob = sdp[Pa1b1xy(P, ([a1,b1],[x,y]))]
                    row.append(f'{prob:.8g}')
            print(f'\t'.join(row))
 

LEVEL = 2                       # NPA relaxation level
VERBOSE = 1                     # Relate to how detail of the info will be printed
N_WORKER_LOOP = 2           # Number of tasks running parallelly
N_WORKER_SDP = 6           # Number of cores for each task
SOLVER_CONFIG = ['mosek', {'dparam.presolve_tol_x': 1e-10,
                           'dparam.intpnt_co_tol_rel_gap': 1e-7,
                           'iparam.num_threads': N_WORKER_SDP}]
ACCURATE_DIGIT = 4              # Achievable precision
ZERO_TOL = 1e-10                # Tolerate error for hardy zeros
EPSILON = 1e-5                  # Relax the precise winning prob constraint to a range with epsilon (1e-5)
SAVEDATA = True                # Set the data into file
TIMMING =  True                 # True for timming
OUT_DIR = './data/BBS20'

# Setup of the scenario for Alice and Bob
configA = [4,4,4,4]
configB = [4,4,4,4]
P = ncp.Probability(configA, configB)
A, B = P.parties

# Generate extra monomials AB
extra_monos = []
'''
for a in ncp.flatten(A):
    for b in ncp.flatten(B):
        extra_monos.append(a*b)
'''

substs = P.substitutions
### Setup the sequential measurement constraints
# Orthogonality
#seq_orthogonal_constr = []
for y in range(2):
    for b1 in range(2):
        substs.update({B[y*2][b1*2]*B[y*2+1][(b1^1)*2]:0,
                       B[y*2+1][(b1^1)*2]*B[y*2][b1*2]:0})
        if b1 == 0:
            substs.update({B[y*2][b1*2+1]*B[y*2+1][(b1^1)*2]:0,
                           B[y*2+1][(b1^1)*2]*B[y*2][b1*2+1]:0})
            
for x in range(2):
    for a1 in range(2):
        substs.update({A[x*2][a1*2]*A[x*2+1][(a1^1)*2]:0,
                       A[x*2+1][(a1^1)*2]*A[x*2][a1*2]:0})
        if a1 == 0:
            substs.update({A[x*2][a1*2+1]*A[x*2+1][(a1^1)*2]:0,
                           A[x*2+1][(a1^1)*2]*A[x*2][a1*2+1]:0})
# print('Sequential orthogonal constraints')
# print(seq_orthogonal_constr)

# Causality (Future input cannot affect past output)
seq_constr = [B[0][0]+B[0][1]-B[1][0]-B[1][1],
              B[2][0]+B[2][1]-B[3][0]-B[3][1],
              A[0][0]+A[0][1]-A[1][0]-A[1][1],
              A[2][0]+A[2][1]-A[3][0]-A[3][1],]

# Task function for multiprocessing
def task(P, win_prob, inputs, substs, seq_constr, extra_monos = [], zero_constr = []):
    sdp = ncp.SdpRelaxation(P.get_all_operators(), verbose=VERBOSE-1)

    win_prob_poly = winProb(P)
    win_prob_constr = [win_prob_poly - win_prob + EPSILON,
                       -win_prob_poly + win_prob + EPSILON]
    
    moment_ineqs = zero_constr + win_prob_constr
    if LEVEL == 1:
        moment_ineqs += probConstr(P) 
    
    eqs = seq_constr
    obj = objective(P, inputs)

    sdp.get_relaxation(level=LEVEL, objective = -obj,
                       substitutions = substs, equalities = eqs,
                       momentinequalities = moment_ineqs,
                       extramonomials = extra_monos)
    sdp.solve(*SOLVER_CONFIG)

    if VERBOSE:
        printPa1b1xy(sdp,P)

    if sdp.status == 'optimal':
        return sdp[win_prob_poly], -sdp.primal
    else:
        if VERBOSE:
            print(f'Bad solve for winning_prob {win_prob}', file=sys.stderr)
            print(sdp.status, sdp.primal, sdp.dual, file=sys.stderr)
        return sdp[win_prob_poly], -1

ZERO_CLASS = ['2c','3a'] #['','2a','2b','2b_swap','2c','3a','3b','1']

if TIMMING:
    tic = time.time()

for zero_class in ZERO_CLASS:
    zero_pos = CLASS_ZERO_POS.get(zero_class, [])
    zero_constr = zeroProbConstr(P, zero_pos, ZERO_TOL, on_Pa1b1xy = True)
    #print(zero_constr)

    if VERBOSE:
        Correlation = zero_class if zero_class else 'chsh'
        print(f'Correlation type: {Correlation}')
        if zero_pos:
            print(f'Zero positions: {zero_pos}')
        else:
            print(f'No zero constraints')
    
    _, Q_bound, _ = maxBellFunc([[2,2],[2,2]],
                                {'id': 1/4, 'A0': 1/2, 'B0': 1/2, 'A0*B0': -1/2,
                                 'A0*B1': -1/2, 'A1*B0': -1/2, 'A1*B1': 1/2},
                                zero_positions = zero_pos, zero_tol = ZERO_TOL, constraints = {},
                                level = LEVEL, verbose = VERBOSE-1, solver_config = SOLVER_CONFIG)

    P_WIN_Q = truncate(-Q_bound, ACCURATE_DIGIT)
    P_WIN_C = 0.75
    P_WIN_MID = (P_WIN_Q + P_WIN_C)/2
    P_WINs = [*np.linspace(P_WIN_Q, P_WIN_MID, 6, endpoint=False), *np.linspace(P_WIN_MID, P_WIN_C, 4)]
    NUM_SLICE = len(P_WINs)
    
    INPUTS = [(0,0),(0,1),(1,0),(1,1)]

    for inputs in INPUTS:
        if VERBOSE:
            if inputs is None:
                print('Averaged inputs')
            else:
                print(f'Chosen input xy={inputs[0]}{inputs[1]}')

        results = Parallel(n_jobs=N_WORKER_LOOP, verbose = 0)(
                    delayed(task)(P, p_win, inputs, substs, seq_constr, extra_monos = extra_monos,
                                  zero_constr = zero_constr) for p_win in P_WINs)
        
        if VERBOSE:
            for p_win, p_quess in results:
                print(f'winning_prob: {p_win:.10g}\tguessing_prob: {p_quess:.10g}')

        if SAVEDATA:
            OUT_COM = 'br'
            CLS = 'chsh' if not zero_class else f'class_{zero_class}'
            INP = f'xy_{inputs[0]}{inputs[1]}' if inputs is not None else 'avg_inputs'
            OUT_FILE = f'{OUT_COM}-{CLS}-{INP}-BBS20-4444.csv'
            OUT_PATH = os.join.path(OUT_DIR, OUT_FILE)
            np.savetxt(OUT_PATH, results, fmt='%.5g', delimiter=',')

if TIMMING:
    toc = time.time()
    print(f'Elapsed time: {toc-tic}')
