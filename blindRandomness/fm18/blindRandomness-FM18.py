"""
    Compute local randomness (min-entropy)
    where Bob does the second measurement to guess Alice's outcome.
    This program is based on the idea in Fu-Miller's work (Phys. Rev. A 97, 032324).
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

def BellFunction(operator_list, polynomial):
    """
        Construct Bell function from given name of operator and its coefficient
    """
    Bell_function = 0
    operator_names = [op.__repr__() for op in operator_list]
    for monomial, coeff in polynomial.items():
        if monomial == 'id':
            Bell_function += coeff
        else:
            ops = monomial.split('*')
            new_term = coeff
            for op in ops:
                try:
                    op_index = operator_names.index(op)
                except ValueError:
                    print(f'Operator {op} not in operator_list.')
                    return
                new_term *= operator_list[op_index]
            Bell_function += new_term
    return Bell_function

def maxBellFunc(scenario, bell_function,
                zero_positions = [], zero_tol = 0, constraints = {},
                level = 2, verbose = 1, solver_config = {}, showProb = False):
    """
        Maxmize given Bell function in given scenario
    """
    P = ncp.Probability(*scenario)
    all_operators = P.get_all_operators()
    
    sdp = ncp.SdpRelaxation(all_operators, verbose = verbose)
    
    objective = BellFunction(all_operators, bell_function)
    substs = P.substitutions
    
    constr = {'equalities': [], 'inequalities': [],
               'momentequalities': [], 'momentinequalities': []}

    if constraints:
        for key in list(set(constraints.keys()) & set(constr.keys())):
            polys = constraints[key]
            for poly in polys:
                constr[key].append(BellFunction(all_operators, poly))

    if zero_positions:
        constr['momentinequalities'] += zeroConstr(P, zero_positions, zero_tol)
    
    sdp.get_relaxation(level = level, objective = -objective,
                       substitutions = substs,
                       **constr)
    sdp.solve(*solver_config)

    if verbose:
        print(sdp.status, sdp.primal, sdp.dual)

    if showProb:
        printProb(sdp, P, scenario)
    
    return sdp.status, sdp.primal, sdp.dual


# Score (winning probability) constraint of non-local game
def winProb(P, scenario = [[2,2],[2,2]], meas_prob = [], random_meas = True):
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
                    # One should modify the following line for different winning condition
                    if ( a^b == (x*y)^1 ):
                        if not random_meas:
                            try:
                                prob = meas_prob[x][y]
                            except IndexError:
                                print(f'Wrong input meas_prob: {meas_prob}')
                        else:
                            prob = 1/4
                        win_prob += P([a,b], [x,y])*prob
    return win_prob

def fixProbConstr(P, positions, fixed_val, err_tol = 0):
    """ 
        Set the zero probability constraints for the positions with tolerable error 
    """
    constr = []
    for pos in positions:
        constr += [P(*pos) - fixed_val + err_tol, -P(*pos) + fixed_val + err_tol]
    return constr

def objective(P, inputs = None):
    """
       The objective function of guessing probability for given inputs in FM18 scenario.
       - inputs: Optimize for given inputs. If it isn't specified (or assigned None), 
                 the objective will be made by uniformly averaging over all inputs.
    """
    A, B = P.parties
    if inputs is None:
        obj = 1
        for x in range(2):
            for y in range(2):
                #y_prime = y*2 + x + 2
                y_prime = x + 2
                obj += (- A[x][0] - B[y_prime][0] + 2*A[x][0]*B[y_prime][0] + B[y][0]*B[y_prime][0] + B[y_prime][0]*B[y][0] \
                        - 2*A[x][0]*B[y][0]*B[y_prime][0] - 2*A[x][0]*B[y_prime][0]*B[y][0] \
                        - 2*B[y][0]*B[y_prime][0]*B[y][0] + 4*A[x][0]*B[y][0]*B[y_prime][0]*B[y][0])/4
                
    else:
        x, y = inputs
        '''
        obj = 0
        for a in range(2):
            for b1 in range(2):
                b = b1*2+a
                obj = obj + P([a,b],[x, y*2+x])
        '''
        #y_prime = y*2 + x + 2
        y_prime = x + 2
        obj = 1 - A[x][0] - B[y_prime][0] + 2*A[x][0]*B[y_prime][0] + B[y][0]*B[y_prime][0] + B[y_prime][0]*B[y][0] \
                - 2*A[x][0]*B[y][0]*B[y_prime][0] - 2*A[x][0]*B[y_prime][0]*B[y][0] \
                - 2*B[y][0]*B[y_prime][0]*B[y][0] + 4*A[x][0]*B[y][0]*B[y_prime][0]*B[y][0]

    return obj

def Pab1xy(P, pos):
    ab1, xy = pos
    xy = [xy[0], xy[1]*2+xy[0]]
    return P([ab1[0], ab1[1]*2], xy) + P([ab1[0], ab1[1]*2+1], xy)


LEVEL = 2                       # NPA relaxation level
VERBOSE = 1                     # Relate to how detail of the info will be printed
N_WORKER_LOOP = 1           # Number of tasks running parallelly
N_WORKER_SDP = 6           # Number of cores for each task
SOLVER_CONFIG = ['mosek', {'dparam.presolve_tol_x': 1e-10,
                           'dparam.intpnt_co_tol_rel_gap': 1e-7,
                           'iparam.num_threads': N_WORKER_SDP}]
ACCURATE_DIGIT = 4              # Achievable precision
ZERO_TOL = 1e-10                # Tolerate error for hardy zeros
EPSILON = 1e-5                  # Relax the precise winning prob constraint to a range with epsilon
SAVEDATA = False               # Set the data into file
TIMMING = True                  # True for timming
OUT_DIR = './data/FM18'


# Setup of the scenario for Alice and Bob
configA = [2,2]
configB = [2,2,2,2]
P = ncp.Probability(configA, configB)
A, B = P.parties

substs = P.substitutions

if TIMMING:
    tic = time.time()

# Task function for multiprocessing
def task(P, win_prob, inputs, substs, zero_constr = []):
    sdp = ncp.SdpRelaxation(P.get_all_operators(), verbose=VERBOSE-1)

    win_prob_poly = winProb(P)
    win_prob_constr = [win_prob_poly - win_prob + EPSILON,
                       -win_prob_poly + win_prob + EPSILON]
    
    moment_ineqs = zero_constr + win_prob_constr
    if LEVEL == 1:
        moment_ineqs += probConstr(P) 
    obj = objective(P, inputs)

    sdp.get_relaxation(level=LEVEL, objective = -obj, substitutions = substs,
                       momentinequalities = moment_ineqs)
    sdp.solve(*SOLVER_CONFIG)

    #if VERBOSE:
        #print(sdp.status, sdp.primal, sdp.dual)
        #printPab1xy(sdp,P)

    if sdp.status == 'optimal':
        return sdp[win_prob_poly], -sdp.primal
    else:
        if VERBOSE:
            print(f'Bad solve for winning_prob {win_prob}', file=sys.stderr)
            print(sdp.status, sdp.primal, sdp.dual, file=sys.stderr)
        return sdp[win_prob_poly], -1

ZERO_CLASS = [''] #,'2a','2b','2b_swap','2c','3a','3b','1']

for zero_class in ZERO_CLASS:
    zero_pos = CLASS_ZERO_POS.get(zero_class, [])
    zero_constr = zeroConstr(P, zero_pos, ZERO_TOL)
    #prob_vals = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    #for fixed_prob_val in prob_vals:
    #    fix_prob_constr = fixProbConstr(P, zero_pos, fixed_prob_val, ZERO_TOL)
        
    if VERBOSE:
        if zero_class:
            print(f'Correlation type: {zero_class}')
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
    P_WINs = [P_WIN_Q] #[*np.linspace(P_WIN_Q, P_WIN_MID, 24, endpoint=False), *np.linspace(P_WIN_MID, P_WIN_C, 16)]
    NUM_SLICE = len(P_WINs)

    INPUTS = [(0,0),(0,1),(1,0),(1,1)]

    for inputs in INPUTS:
        if VERBOSE:
            if inputs is None:
                print('Optimize for averaged inputs')
            else:
                print(f'Chosen input xy={inputs[0]}{inputs[1]}')

        results = Parallel(n_jobs=N_WORKER_LOOP, verbose = 0)(
                    delayed(task)(P, p_win, inputs, substs, zero_constr) for p_win in P_WINs)
        
        if VERBOSE:
            for p_win, p_quess in results:
                print(f'winning_prob: {p_win:.10g}\tguessing_prob: {p_quess:.10g}')

        if SAVEDATA:
            COM = 'br'
            CLS = 'chsh' if not zero_class else f'class_{zero_class}'
            INP = f'xy_{inputs[0]}{inputs[1]}' if inputs is not None else 'avg_inputs'
            OUT_FILE = f'{COM}-{CLS}-{INP}-FM18-alt.csv'
            OUT_PATH = os.join.path(OUT_DIR, OUT_FILE)
            np.savetxt(OUT_PATH, results, fmt='%.5g', delimiter=',')

if TIMMING:
    toc = time.time()
    print(f'Elapsed time: {toc-tic}')
