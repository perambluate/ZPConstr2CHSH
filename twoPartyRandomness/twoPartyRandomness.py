"""
    This script is used to compute the global randomness in terms of the lower bound on
    the von Neumann entropy H(AB|XYE) by Brown-Fawzi-Fawzi method (arXiv:2106.13692v2)
    for the correlations with zero-probability constraints (Quantum 7, 1054 (2023)).
"""
import ncpol2sdpa as ncp
from sympy.physics.quantum.dagger import Dagger
import numpy as np
from joblib import Parallel, delayed
import time
import sys, os

### Add current directory to Python path
sys.path.append('.')
from common_func.SDP_helper import *

SAVEDATA = False                 # Set the data into file
TIMMING = True                  # True for timming
OUT_DIR = './data'              # Folder for the data to save

LEVEL = 2                       # NPA relaxation level
M = 6                           # Num of terms in Gauss-Radau quadrature = 2*M
VERBOSE = 1                     # Relate to how detail of the info will be printed
N_WORKER_LOOP = 2               # Number of workers for the outer loop
N_WORKER_QUAD = 1               # Number of workers for parallelly computing quadrature
N_WORKER_SDP = 6                # Number of threads for solving a single SDP
N_WORKER_NPA = 6                # Number of threads for generating a NPA moment matrix
PRIMAL_DUAL_GAP = 5e-5          # Allowable gap between primal and dual
SOLVER_CONFIG = ['mosek', {'dparam.intpnt_co_tol_rel_gap': PRIMAL_DUAL_GAP,
                           'iparam.num_threads': N_WORKER_SDP,
                           'iparam.infeas_report_level': 4}]
ACCURATE_DIGIT = 6              # Achievable precision of the solver
WIN_TOL = 1e-5                  # Relax the precise winning prob constraint to a range with epsilon
ZERO_PROB = 1e-6                # Treat this value as zero for zero-probability (ZP) constraints
PARAMSCAN = True                # Do the ZP tolerance scanning or not

## Positions of zeros for each class
CLASS_ZERO_POS = zero_position_map()

## Inputs that give maximal rate
OPT_INPUT_MAP = {'chsh': (0,0), '1': (0,1), '2a': (1,1), '2b': (0,1),
                 '2c': (0,1), '3a': (0,0), '3b': (0,1)}

## Bound/Max winning probability
C_BOUND = 0.75                  # Local bound

## Classes of correlations to run
if PARAMSCAN:
    CLASSES = ['2a', 'chsh']
else:
    CLASSES = ['1', '2a', '2b', '2c', '3a', '3b', 'chsh']

## Printing precision of numpy arrays
np.set_printoptions(precision=5)

def winProb(P, scenario = [[2,2],[2,2]], inp_probs = []):
    """
        Function of winning probability of nonlocal game
        Here we use the CHSH game for example
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

# Setup of the scenario for Alice and Bob
configA = [2,2]
configB = [2,2]
SCENARIO = (configA, configB)
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

    ## Tolerable errors for zero-probability constraints
    N_PARAM = 1
    if PARAMSCAN:
        if cls != 'chsh':
            PARAMs = [{'wtol': 1e-5, 'ztol': 1e-6},
                      {'wtol': 1e-3, 'ztol': 1e-3}]
        else:
            PARAMs = [{'wtol': 1e-3, 'ztol': 0}]
        N_PARAM = len(PARAMs)
        N_POINT = len(PARAMs)
        MAX_WINs = np.zeros(N_POINT)
        ENTROPYs = np.zeros(N_POINT)
        LAMBDAs = np.zeros((N_POINT, len(zero_pos)+1))
        C_LAMBDAs = np.zeros(N_POINT)

    for i in range(N_PARAM):
        if PARAMSCAN:
            zero_tol = PARAMs[i]['ztol']
            win_tol = PARAMs[i]['wtol']
        else:
            zero_tol = ZERO_PROB
            win_tol = WIN_TOL
        
        if VERBOSE:
            print(f'Zero probability tolerance: {zero_tol}')
            print(f'Winning probability tolerance: {win_tol}')
        
        zero_constraints = zeroConstr(P, zero_pos, zero_tol)

        if VERBOSE:
            print('Start compute winning probability quantum bound>>>')

        # Compute the quantum bound first
        sdp_Q = ncp.SdpRelaxation(P.get_all_operators(), verbose=max(VERBOSE-3, 0))
        sdp_Q.get_relaxation(level=2, objective = -winProb(P),
                            substitutions = P.substitutions,
                            momentinequalities = zero_constraints)
        sdp_Q.solve(*SOLVER_CONFIG)

        if VERBOSE:
            print('Status\tPrimal\tDual')
            print(f'{sdp_Q.status}\t{sdp_Q.primal}\t{sdp_Q.dual}')
            if zero_pos:
                max_zero_pos = max([sdp_Q[P(pos[0], pos[1])] for pos in zero_pos])
                print(f'Max val in zero position: {max_zero_pos:.9g}')

        if sdp_Q.status != 'optimal' and sdp_Q.status != 'primal-dual feasible':
            print('Cannot compute quantum bound correctly!', file=sys.stderr)
            break

        if VERBOSE:
            print('End of computing quantum bound.<<<\n')
        
        max_win = truncate(-sdp_Q.primal, ACCURATE_DIGIT)
        if PARAMSCAN:
            MAX_WINs[i] = max_win
        else:
            N_POINT = 20
            P_WINs = np.flip(intervalSpacingBeta((C_BOUND, max_win), num_points = N_POINT,
                                                a = 1.2, b = 0.5))
            ENTROPYs = np.zeros(N_POINT)
            LAMBDAs = np.zeros((N_POINT, len(zero_pos)+1))
            C_LAMBDAs = np.zeros(N_POINT)
                
    # The selected inputs to compute entropy
    inputs = OPT_INPUT_MAP.get(cls, (0,0))

    if VERBOSE:
        print(f'Chosen input xy={inputs[0]}{inputs[1]}')

    if not PARAMSCAN:
        scan_params = [(win_prob, WIN_TOL, ZERO_PROB) for win_prob in P_WINs]
    else:
        WIN_TOLs = [PARAMs[i]['wtol'] for i in range(N_PARAM)]
        ZERO_TOLs = [PARAMs[i]['ztol'] for i in range(N_PARAM)]
        scan_params = list(zip(MAX_WINs, WIN_TOLs, ZERO_TOLs))
    
    results = Parallel(n_jobs=N_WORKER_LOOP, verbose=0)(
                        delayed(singleRoundEntropy)('two', P, Z_ab, M, inputs, scenario = SCENARIO,
                                                    win_prob = scan_params[i][0],
                                                    win_tol = scan_params[i][1],
                                                    zero_tol = scan_params[i][2],
                                                    win_prob_func = winProb,
                                                    zero_class = cls, substs = substs,
                                                    extra_monos = extra_monos, level = LEVEL,
                                                    quad_end = True, quad_ep = .999,
                                                    n_worker_quad = N_WORKER_QUAD,
                                                    n_worker_npa = N_WORKER_NPA,
                                                    solver_config = SOLVER_CONFIG,
                                                    verbose = VERBOSE) for i in range(N_POINT))
    # print(results)
    ENTROPYs, LAMBDAs, C_LAMBDAs = zip(*results)

    if VERBOSE or SAVEDATA:
        if cls == 'chsh':
            if PARAMSCAN:
                metadata = ['max_win', 'win_tol', 'win_prob', 'entropy', 'lambda', 'c_lambda']
                data = np.vstack((MAX_WINs, WIN_TOLs, MAX_WINs, ENTROPYs, np.squeeze(LAMBDAs), C_LAMBDAs)).T
            else:
                metadata = ['win_prob', 'entropy', 'lambda', 'c_lambda']
                data = np.vstack((P_WINs, ENTROPYs, np.squeeze(LAMBDAs), C_LAMBDAs)).T
        else:
            lambda_zp = zeroPos2str(zero_pos)
            lambda_zp = [f'lambda_{pos}' for pos in lambda_zp]
            if PARAMSCAN:
                metadata = ['zero_tol', 'max_win', 'win_tol', 'win_prob',
                            'entropy', 'lambda', *lambda_zp, 'c_lambda']
                data = np.vstack((ZERO_TOLs, MAX_WINs, WIN_TOLs, MAX_WINs,
                                  ENTROPYs, np.array(LAMBDAs).T, C_LAMBDAs)).T
            else:
                metadata = ['win_prob', 'entropy', 'lambda', *lambda_zp, 'c_lambda']
                data = np.vstack((P_WINs, ENTROPYs, np.array(LAMBDAs).T, C_LAMBDAs)).T

        headline = ', '.join(metadata)
    
        if VERBOSE:
            print(headline)        
            for data_per_line in data:
                zp_tol = data_per_line[0]
                print(f'{zp_tol:.5g}\t'+ \
                      '\t'.join([f'{val:.6f}' for val in data_per_line[1:]]))
            print("\n")

        # Write to file
        if SAVEDATA:
            if not PARAMSCAN:
                MAX_WIN = f'MAX_WIN_PROB\n{max_win:.5f}'

            COM = 'tpr'
            CLS = cls
            INP = f'xy_{inputs[0]}{inputs[1]}'
            WTOL = f'wtol_{WIN_TOL:.0e}'
            ZTOL = f'ztol_{zero_tol:.0e}'
            PDGAP = f'pdgap_{PRIMAL_DUAL_GAP:.0e}'
            QUAD = f'M_{M*2}'
            NP = f'N_{N_POINT}'
            TAIL = 'wplb'
            if PARAMSCAN:
                OUT_FILE = f'{COM}-{CLS}-{INP}-{PDGAP}-{QUAD}-{NP}'
            else:
                OUT_FILE = f'{COM}-{CLS}-{INP}-{WTOL}-{ZTOL}-{PDGAP}-{QUAD}-{NP}'
            
            if TAIL:
                OUT_FILE = f'{OUT_FILE}-{TAIL}.csv'
            else:
                OUT_FILE += '.csv'
            
            OUT_PATH = os.path.join(OUT_DIR, OUT_FILE)
            
            mode = 'ab' if os.path.exists(OUT_PATH) else 'wb'
            with open(OUT_PATH, mode) as file:
                if not PARAMSCAN:
                    file.write(bytes(MAX_WIN, 'utf-8') + b'\n')
                np.savetxt(file, data, fmt='%.6f', delimiter=',', header=headline)
                
if TIMMING:
    toc = time.time()
    print(f'Elapsed time: {toc-tic}')

