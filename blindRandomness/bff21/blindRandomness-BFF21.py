"""
    This script is used to compute the von Neumann entropy H(A|XYBE)
      with the Brown-Fawzi-Fawzi method (arXiv:2106.13692v2)
      for blind randomness in the point of view of other player in CHSH non-local game
"""
import ncpol2sdpa as ncp
from sympy.physics.quantum.dagger import Dagger
import numpy as np
from joblib import Parallel, delayed
import time
import sys, os

### Add current directory to Python path
sys.path.append('..')
from common_func.SDP_helper import *

SAVEDATA = True                 # Set the data into file
TIMMING = True                  # True for timming
OUT_DIR = './data/BFF21'   # Folder for the data to save

LEVEL = 2                       # NPA relaxation level
M = 9                           # Num of terms in Gauss-Radau quadrature = 2*M
VERBOSE = 1                     # Relate to how detail of the info will be printed
N_WORKER_QUAD = 4               # Number of workers for parallelly computing quadrature
N_WORKER_SDP = 4                # Number of threads for solving a single SDP
PRIMAL_DUAL_GAP = 1e-6          # Allowable gap between primal and dual
SOLVER_CONFIG = ['mosek', {#'dparam.presolve_tol_x': 1e-10,
                           'dparam.intpnt_co_tol_rel_gap': PRIMAL_DUAL_GAP,
                           'iparam.num_threads': N_WORKER_SDP,
                           'iparam.infeas_report_level': 4}]
# SOLVER_CONFIG = ['sdpa']
ACCURATE_DIGIT = 5               # Achievable precision of the solver
WIN_TOL = 1e-4                   # Relax the precise winning prob constraint to a range with epsilon 2e-5
ZERO_PROB = 1e-9                 # Treat this value as zero for zero probability constraints 1e-10
#GAMMA_A = 0.9                   # Prob. of Alice's input for key generation
#GAMMA_B = 0.9                   # Prob. of Bob's input for key generation

## Positions of zeros for each class
CLASS_ZERO_POS = zero_position_map()

## Inputs that give maximal rate
OPT_INPUT_MAP = {'1': (0,1), '2a': (1,1), '2b': (0,1), '2b_swap': (1,1),
                 '2c': (1,0), '3a': (1,1), '3b': (1,0)}

## Bound/Max winning probability
C_BOUND = 0.75                 # Local bound
CLASS_MAX_WIN = max_win_map()       # Quantum bound for each classes
# GAMMA = 0.25
INP_PROB = [[1/4, 1/4], [1/4, 1/4]]

## Classes of correlations to run
CLASSES = ['chsh', '3b'] #['1','2a','2b','2b_swap','2c','3a','3b']

## Printing precision of numpy arrays
np.set_printoptions(precision=5)

def winProb(P, scenario = [[2,2],[2,2]], inp_probs = np.empty(0)):
    """
        Function of winning probability of nonlocal game
        Here we use a modified CHSH non-local game for example
        - P:        P([a,b],[x,y])=P(ab|xy) is the probability that 
                        get the outputs (a,b) for given inputs (x,y)
        - scenario: Tell the number of inputs and number of outputs for each input
                        in each party may be useful for general scenario
        - inp_probs: The probability of the inpus P(xy)
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
                    if a^b == (x*y)^1:
                        if np.any(inp_probs):
                            try:
                                prob = inp_probs[x][y]
                            except IndexError:
                                print(f'Wrong input inp_probs: {inp_probs}')
                        else:
                            prob = 1/4
                        win_prob += P([a,b], [x,y])*prob
    return win_prob

if VERBOSE:
    if SOLVER_CONFIG[0] == 'mosek':
        print(f'MOSEK primal-dual tol gap: {PRIMAL_DUAL_GAP}')
    print(f'Accurate digit: {ACCURATE_DIGIT}')
    print(f'Number of terms summed in quadrature: {M*2}')
    print(f'WinProb deviation: {WIN_TOL}')

# Setup of the scenario for Alice and Bob
configA = [2,2]
configB = [2,2]
scenario = (configA, configB)
P = ncp.Probability(configA, configB)
A, B = P.parties
substs = P.substitutions

# Generate Eve's operators
Z_ab = ncp.generate_operators('Z_ab', 4, hermitian=False)

# Make Eve's operators commute with Alice and Bob's
for a in P.get_all_operators():
    for z in Z_ab:
        substs.update({z*a: a*z, Dagger(z)*a: a*Dagger(z)})

# Generate ABZ, ABZ*, ABZZ*, ABZ*Z as extra monomials
extra_monos = []
for a in ncp.flatten(A):
    for b in ncp.flatten(B):
        for z in Z_ab:
            extra_monos += [a*b*z, a*b*Dagger(z), a*b*z*Dagger(z), a*b*Dagger(z)*z]

if TIMMING:
    tic = time.time()

for cls in CLASSES:

    zero_pos = CLASS_ZERO_POS.get(cls, [])

    if VERBOSE:
        print(f'Correlation type: {cls}')
        if zero_pos:
            print(f'Zero positions: {zero_pos}')
        else:
            print(f'No zero constraints')

    inputs = OPT_INPUT_MAP.get(cls, (0,0))
    if VERBOSE:
        print(f'Chosen input xy={inputs[0]}{inputs[1]}')
    
    if cls == 'chsh':
        ZERO_TOLs = [ZERO_PROB]
    else:
        ZERO_TOLs = [ZERO_PROB, 1e-3]
    N_POINT = len(ZERO_TOLs)
    MAX_WINs = np.zeros(N_POINT)
    WIN_PROBs = np.zeros(N_POINT)
    ENTROPYs = np.zeros(N_POINT)
    LAMBDAs = np.zeros((N_POINT, len(zero_pos)+1))
    C_LAMBDAs = np.zeros(N_POINT)
    # for gamma in GAMMAs:
    #     GAMMA_A, GAMMA_B = math.sqrt(gamma), math.sqrt(gamma)
    #     inp_probs = np.zeros((2,2))
    #     inp_probs[inputs[0]][inputs[1]] = GAMMA_A*GAMMA_B
    #     inp_probs[inputs[0]^1][inputs[1]] = (1-GAMMA_A)*GAMMA_B
    #     inp_probs[inputs[0]][inputs[1]^1] = GAMMA_A*(1-GAMMA_B)
    #     inp_probs[inputs[0]^1][inputs[1]^1] = (1-GAMMA_A)*(1-GAMMA_B)

    for i in range(N_POINT):
        zero_tol = ZERO_TOLs[i]
        if VERBOSE:
            print(f'Zero probability tolerance: {zero_tol}')
        zero_constraints = zeroConstr(P, zero_pos, zero_tol)

        if VERBOSE:
            print('Start compute winning probability quantum bound>>>')

        # Compute the quantum bound first
        sdp_Q = ncp.SdpRelaxation(P.get_all_operators(), verbose=max(VERBOSE-3, 0))
        sdp_Q.get_relaxation(level=4, objective=-winProb(P, inp_probs=INP_PROB),
                            substitutions = P.substitutions,
                            momentinequalities = zero_constraints)
        sdp_Q.solve(*SOLVER_CONFIG)

        if VERBOSE:
            print('Status\tPrimal\tDual')
            print(f'{sdp_Q.status}\t{sdp_Q.primal}\t{sdp_Q.dual}')
            print('End of computing quantum bound.<<<')

        if sdp_Q.status != 'optimal' and sdp_Q.status != 'primal-dual feasible':
            print('Cannot compute quantum bound correctly!', file=sys.stderr)
            break
        
        max_win = truncate(-sdp_Q.primal, ACCURATE_DIGIT)
        MAX_WINs[i] = max_win
        # N_POINT = 21
        # P_WINs = np.flip(intervalSpacingBeta((C_BOUND, max_win), num_points = N_POINT))
        # WIN_PROBs = np.zeros(N_POINT)
        # ENTROPYs = np.zeros(N_POINT)
        # LAMBDAs = np.zeros((N_POINT, len(zero_pos)+1))
        # C_LAMBDAs = np.zeros(N_POINT)

        # for i in range(N_POINT):
        win_prob = max_win
        if VERBOSE:
            print(f'win prob: {win_prob}')
        results = singleRoundEntropy('blind', P, Z_ab, M, inputs,
                                        win_prob_func = winProb, win_prob = win_prob,
                                        scenario = scenario, inp_probs = INP_PROB,
                                        win_tol = WIN_TOL, zero_class = cls,
                                        zero_tol = zero_tol, substs = substs,
                                        extra_monos = extra_monos, level = LEVEL,
                                        n_worker_quad = N_WORKER_QUAD,
                                        solver_config = SOLVER_CONFIG,
                                        quad_end = False, verbose = VERBOSE)
        win_prob, entropy, lambda_, c_lambda = results
        WIN_PROBs[i] = win_prob
        ENTROPYs[i] = entropy
        if VERBOSE:
            print(f'entropy: {entropy}')
        LAMBDAs[i] = lambda_
        C_LAMBDAs[i] = c_lambda

    if VERBOSE or SAVEDATA:
        if cls == 'chsh':
            metadata = ['max_win', 'win_prob', 'entropy', 'lambda', 'c_lambda']
            # metadata = ['win_prob', 'entropy', 'lambda', 'c_lambda']
        else:
            zero_pos_str = zeroPos2str(zero_pos)
            zero_pos_str = [f'lambda_{pos}' for pos in zero_pos_str]
            metadata = ['zero_tol', 'max_win', 'win_prob', 'entropy',
                        'lambda', *zero_pos_str, 'c_lambda']
            # metadata = ['win_prob', 'entropy', 'lambda', *zero_pos_str, 'c_lambda']
        headline = ', '.join(metadata)
    
    if VERBOSE:
        print(headline)
        
        if cls == 'chsh':
            for max_win, win_prob, entropy, lambda_, c_lambda in \
                zip(MAX_WINs, WIN_PROBs, ENTROPYs, LAMBDAs, C_LAMBDAs):
                line = f'{max_win:.6f}\t{win_prob:.6f}\t{entropy:.6f}\t{lambda_[0]:.6f}\t{c_lambda:.6f}'
                # line = f'{win_prob:.6f}\t{entropy:.6f}\t{np.squeeze(lambda_):.6f}\t{c_lambda:.6f}'
                print(line)
        else:
            for zero_tol, max_win, win_prob, entropy, lambda_, c_lambda in \
                zip(ZERO_TOLs, MAX_WINs, WIN_PROBs, ENTROPYs, LAMBDAs, C_LAMBDAs):
            # for win_prob, entropy, lambda_, c_lambda in zip(WIN_PROBs, ENTROPYs, LAMBDAs, C_LAMBDAs):
                lambda_vals = [f'{val:.6f}' for val in lambda_]
                lambda_str = '\t'.join(lambda_vals)
                line = f'{zero_tol:.5g}\t{max_win:.6f}\t{win_prob:.6f}\t{entropy:.6f}\t' \
                        +lambda_str+f'\t{c_lambda:.6f}'
                # line = f'{win_prob:.6f}\t{entropy:.6f}\t'+lambda_str+f'\t{c_lambda:.6f}'
                print(line)

        print("\n")

    # Write to file
    if SAVEDATA:
        if cls == 'chsh':
            data = np.vstack((MAX_WINs, WIN_PROBs, ENTROPYs, np.squeeze(LAMBDAs), C_LAMBDAs)).T
            # data = np.vstack((WIN_PROBs, ENTROPYs, np.squeeze(LAMBDAs), C_LAMBDAs)).T
        else:
            data = np.vstack((ZERO_TOLs, MAX_WINs, WIN_PROBs, ENTROPYs, LAMBDAs.T, C_LAMBDAs)).T
            # data = np.vstack((WIN_PROBs, ENTROPYs, np.array(LAMBDAs).T, C_LAMBDAs)).T

        # HEADER = ', '.join(metadata)
        # MAX_WIN = f'MAX_WIN_PROB\n{max_win:.5g}'

        COM = 'br-ztols'
        CLS = cls
        INP = f'xy_{inputs[0]}{inputs[1]}'
        WTOL = f'wtol_{WIN_TOL:.0e}'
        ZTOL = f'ztol_{zero_tol:.0e}'
        QUAD = f'M_{M*2}'
        # GAM = f'gam_{GAMMA*100:.0f}'
        # OUT_FILE = f'{COM}-{CLS}-{INP}-{WTOL}-{ZTOL}-{QUAD}.csv'
        OUT_FILE = f'{COM}-{CLS}-{INP}-{WTOL}-{QUAD}.csv'
        OUT_PATH = os.path.join(OUT_DIR, OUT_FILE)
        
        if os.path.exists(OUT_PATH):
            with open(OUT_PATH, 'ab') as file:
                file.write(b'\n')
                # file.write(bytes(MAX_WIN, 'utf-8') + b'\n')
                np.savetxt(file, data, fmt='%.5g', delimiter=',', header=headline)
        else:
            with open(OUT_PATH, 'wb') as file:
                # file.write(bytes(MAX_WIN, 'utf-8') + b'\n')
                np.savetxt(file, data, fmt='%.5g', delimiter=',', header=headline)
                
if TIMMING:
    toc = time.time()
    print(f'Elapsed time: {toc-tic}')
