"""
    Compute local randomness (min-entropy)
    where Bob does the second measurement to guess Alice's outcome.
    This program is based on the idea in Fu-Miller's work (Phys. Rev. A 97, 032324).
"""
import ncpol2sdpa as ncp
import mosek
import numpy as np
from joblib import Parallel, delayed
import time
import sys

KS_ZEROS = {'3a': [([0,0],[0,0]),([1,1],[0,1]),([1,1],[1,0])],
            '3b': [([0,0],[0,0]),([1,1],[0,0]),([1,0],[1,1])],
            '3b_swap': [([0,0],[0,0]),([1,1],[0,0]),([0,1],[1,1])],
            '2a': [([0,0],[0,0]),([1,1],[0,0])],
            '2b': [([0,0],[0,0]),([1,1],[1,0])],
            '2b_swap': [([0,0],[0,0]),([1,1],[0,1])],
            '2c': [([0,0],[0,0]),([1,0],[1,1])],
            '2c_swap': [([0,0],[0,0]),([0,1],[1,1])],
            '1' : [([0,0],[0,0])]}

def truncate(number, digit):
    """
        Truncate the floating number up to the given digit after point
        It will ignore any number after the digit
    """
    truncated_number = round(number, digit)
    if round(number, digit+1) < truncated_number:
        truncated_number -= 10 ** (-digit)
    return truncated_number

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
        constr['momentinequalities'] += zeroProbConstr(P, zero_positions, zero_tol)
    
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
def winProb(P, configs = [[2,2],[2,2]], meas_prob = [], random_meas = True):
    """
        Function of winning probability of nonlocal game
        Here we use a modified CHSH non-local game for example
        - P : P([a,b],[x,y])=P(ab|xy) is the probability that get the outputs (a,b) for given inputs (x,y)                                               
        - configs : Tell the number of inputs and number of outputs for each input in each party
                    may be useful for general scenario
        - meas_prob : The probability of the inpus P(xy)
        - random_meas : Take equal probability for each input combination if this is true
    """
    try:
        config_A = configs[0]
        num_x = len(config_A)
        config_B = configs[1]
        num_y = len(config_B)
    except IndexError:
        print(f'Wrong input configs: {configs}')
    win_prob = 0
    for x in range(num_x):
        for y in range(num_y):
            for a in range(config_A[x]):
                for b in range(config_B[y]):
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

def zeroProbConstr(P, positions, err_tol = 0):
    """ 
        Set the zero probability constraints for the positions with tolerable error 
    """
    constr = []
    for pos in positions:
        constr += [err_tol-P(*pos)]
    return constr

def fixProbConstr(P, positions, fixed_val, err_tol = 0):
    """ 
        Set the zero probability constraints for the positions with tolerable error 
    """
    constr = []
    for pos in positions:
        constr += [P(*pos) - fixed_val + err_tol, -P(*pos) + fixed_val + err_tol]
    return constr

def probConstr(P, configs=[[2,2],[2,2,2,2]]):
    try:
        config_A = configs[0]
        num_x = len(config_A)
        config_B = configs[1]
        num_y = len(config_B)
    except IndexError:
        print(f'Wrong input configs: {configs}')
    return [P([a,b],[x,y]) for x in range(num_x) for y in range(num_y) \
            for a in range(config_A[x]) for b in range(config_B[y])]

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

def printProb(sdp, P, configs=[[2,2],[2,2]]):
    try:
        config_A = configs[0]
        num_x = len(config_A)
        config_B = configs[1]
        num_y = len(config_B)
    except IndexError:
        print(f'Wrong input configs: {configs}')
    for y in range(num_y):
        for b in range(config_B[y]):
            s = f'\t'.join([str(sdp[P([a,b],[x,y])]) for x in range(num_x) for a in range(config_A[x])])
            print(s)

LEVEL = 2                       # NPA relaxation level
VERBOSE = 1                     # Relate to how detail of the info will be printed
NUM_PARALLEL_TASK = 1           # Number of tasks running parallelly
NUM_CORE_PER_TASK = 6           # Number of cores for each task
SOLVER_CONFIG = ['mosek', {'dparam.presolve_tol_x': 1e-10,
                           'dparam.intpnt_co_tol_rel_gap': 1e-7,
                           'iparam.num_threads': NUM_CORE_PER_TASK}]
ACCURATE_DIGIT = 4              # Achievable precision
ZERO_ERR = 1e-10                # Tolerate error for hardy zeros
EPSILON = 1e-5                  # Relax the precise winning prob constraint to a range with epsilon
SAVE_DATA = False               # Set the data into file
TIMMING = True                  # True for timming


# Setup of the scenario for Alice and Bob
A_config = [2,2]
B_config = [2,2,2,2]
P = ncp.Probability(A_config, B_config)
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
    zero_pos = KS_ZEROS.get(zero_class, [])
    zero_constr = zeroProbConstr(P, zero_pos, ZERO_ERR)
    #prob_vals = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    #for fixed_prob_val in prob_vals:
    #    fix_prob_constr = fixProbConstr(P, zero_pos, fixed_prob_val, ZERO_ERR)
        
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
                                zero_positions = zero_pos, zero_tol = ZERO_ERR, constraints = {},
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

        results = Parallel(n_jobs=NUM_PARALLEL_TASK, verbose = 0)(
                    delayed(task)(P, p_win, inputs, substs, zero_constr) for p_win in P_WINs)
        
        if VERBOSE:
            for p_win, p_quess in results:
                print(f'winning_prob: {p_win:.10g}\tguessing_prob: {p_quess:.10g}')

        if SAVE_DATA:
            OUT_COMMON = 'local_randomness'
            NON_LOCAL_RELATION = 'CHSH' if not zero_class else f'class_{zero_class}'
            INPUT = f'xy_{inputs[0]}{inputs[1]}' if inputs is not None else 'avg_inputs'
            OUT_FILE = f'{OUT_COMMON}-{NON_LOCAL_RELATION}-{INPUT}-FM18-alt.csv'
            np.savetxt(OUT_FILE, results, fmt='%.10g', delimiter=',')

if TIMMING:
    toc = time.time()
    print(f'Elapsed time: {toc-tic}')
