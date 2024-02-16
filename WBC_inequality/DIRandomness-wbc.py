"""
    This script is used to compute the von Neumann entropy H(A|XYBE)
      with the Brown-Fawzi-Fawzi method (arXiv:2106.13692v2)
      based on the family of Bell in equality found by Wooltoron et al.
      (PRL. vol. 129, no. 15, 5, 150403)
      for blind randomness in the point of view of other player in CHSH non-local game
"""
import ncpol2sdpa as ncp
from sympy.physics.quantum.dagger import Dagger
import math
from math import pi
import numpy as np
from functools import partial
from joblib import Parallel, delayed
import time
import sys, os

### Add current directory to Python path
sys.path.append('..')
from common_func.SDP_helper import *

SAVEDATA = True                 # Set the data into file
TIMMING = True                  # True for timming
OUT_DIR = './'        # Folder for the data to save
RTYPE = 'two'
LEVEL = 2                       # NPA relaxation level
M = 9                           # Number of terms in Gauss-Radau quadrature = 2*M
VERBOSE = 2                     # Relate to how detail of the info will be printed
N_WORKER_QUAD = 4               # Number of workers for parallelly computing quadrature
N_WORKER_SDP = 4                # Number of threads for solving a single SDP
PRIMAL_DUAL_GAP = 1e-6          # Allowable gap between primal and dual
SOLVER_CONFIG = ['mosek', {#'dparam.presolve_tol_x': 1e-10,
                           'dparam.intpnt_co_tol_rel_gap': PRIMAL_DUAL_GAP,
                           'iparam.num_threads': N_WORKER_SDP,
                           'iparam.infeas_report_level': 4}]
# SOLVER_CONFIG = ['sdpa']
ACCURATE_DIGIT = 5              # Achievable precision of the solver
WIN_TOL = 1e-4                  # Relax the precise winning prob constraint to a range with epsilon 2e-5

## Printing precision of numpy arrays
np.set_printoptions(precision=5)

def w_delta_ns_bound(delta):
    return (1 + 2/math.sin(delta) + 1/math.cos(2*delta))/4

def delta_bell_func(P, delta, scenario = [[2,2],[2,2]], inp_probs = np.empty(0)):
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
    weights = np.ones((2, 2))
    weights[0][1] = 1/math.sin(delta)
    weights[1][0] = 1/math.sin(delta)
    weights[1][1] = 1/math.cos(2*delta)
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
                        win_prob += P([a,b], [x,y])*prob*weights[x][y]
    return win_prob/w_delta_ns_bound(delta)

CHSH = [[1, 0, 0, 1], # x0, y0
        [1, 0, 0, 1], # x0, y1
        [1, 0, 0, 1], # x1, y0
        [0, 1, 1, 0]] # x1, y1

if VERBOSE:
    print(f'Rand type: {RTYPE}')
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

if RTYPE == 'blind' or RTYPE == 'two':
    BEST_INP = (0,0)
elif RTYPE == 'one':
    BEST_INP = 0

# Generate Eve's operators
if RTYPE == 'blind' or RTYPE == 'two':
    Z = ncp.generate_operators('Z', 4, hermitian=False)
elif RTYPE == 'one':
    Z = ncp.generate_operators('Z', 2, hermitian=False)

# Make Eve's operators commute with Alice and Bob's
for a in P.get_all_operators():
    for z in Z:
        substs.update({z*a: a*z, Dagger(z)*a: a*Dagger(z)})

# Generate ABZ, ABZ*, ABZZ*, ABZ*Z as extra monomials
extra_monos = []
for a in ncp.flatten(A):
    for b in ncp.flatten(B):
        for z in Z:
            extra_monos += [a*b*z, a*b*Dagger(z)]
            if RTYPE == 'one':
                extra_monos += [a*Dagger(z)*z, a*z*Dagger(z)]
            else:
                extra_monos += [a*b*z*Dagger(z), a*b*Dagger(z)*z]

w_CHSH = 0
for x in range(2):
    for y in range(2):
        for a in range(2):
            for b in range(2):
                inp = x*2+y
                out = a*2+b
                w_CHSH += CHSH[inp][out]*P([a,b], [x,y])/4

if TIMMING:
    tic = time.time()

    DELTAs = [pi/6] #[pi/6, pi/8, pi/10, pi/50, pi/100, pi/1e3, pi/1e4, pi/1e5, pi/1e6]
    N_POINT = len(DELTAs)
    Q_BOUNDs = np.zeros(N_POINT)
    SCOREs = np.zeros(N_POINT)
    ENTROPYs = np.zeros(N_POINT)
    LAMBDAs = np.zeros(N_POINT)
    C_LAMBDAs = np.zeros(N_POINT)

    for i in range(N_POINT):
        delta = DELTAs[i]
        bell_func = partial(delta_bell_func, delta = delta)

        if VERBOSE:
            print(f'Set delta={delta:.5g}')
            print('Start compute winning probability quantum bound >>>')

        # Compute the quantum bound first
        sdp_Q = ncp.SdpRelaxation(P.get_all_operators(), verbose=max(VERBOSE-3, 0))
        sdp_Q.get_relaxation(level=LEVEL, objective=-bell_func(P),
                             substitutions = P.substitutions)
        sdp_Q.solve(*SOLVER_CONFIG)
        qub_succ = (sdp_Q.status == 'optimal') or (sdp_Q.status == 'primal-dual feasible')

        if VERBOSE:
            print('Status\tPrimal\tDual')
            print(f'{sdp_Q.status}\t{sdp_Q.primal}\t{sdp_Q.dual}')
            print('Print probabilities P(ab|xy)')
            printProb(sdp_Q, P)

        qbound = truncate(-sdp_Q.primal, ACCURATE_DIGIT)
        Q_BOUNDs = qbound

        # sdp_Q.get_relaxation(level=LEVEL, objective=bell_func(P),
        #                     substitutions = P.substitutions,
        #                     momentinequalities = [])
        # sdp_Q.solve(*SOLVER_CONFIG)
        # qlb_succ = (sdp_Q.status == 'optimal') or (sdp_Q.status == 'primal-dual feasible')

        # min_p_win = truncate(sdp_Q.primal, ACCURATE_DIGIT)

        # if VERBOSE:
        #     print('Status\tPrimal\tDual')
        #     print(f'{sdp_Q.status}\t{sdp_Q.primal}\t{sdp_Q.dual}')
        #     print('Print probabilities P(ab|xy)')
        #     printProb(sdp_Q, P)
        #     print('End of computing quantum bound. <<<')

        if not qub_succ:
            print('Cannot compute quantum bound correctly!', file=sys.stderr)
        
        score = qbound
        SCOREs[i] = score

        
        results = singleRoundEntropy(RTYPE, P, Z, M, BEST_INP, bell_func, score,
                                     scenario = scenario, win_tol = WIN_TOL,
                                     substs = substs, extra_monos = extra_monos,
                                     level = LEVEL, n_worker_quad = N_WORKER_QUAD,
                                     solver_config = SOLVER_CONFIG, verbose = VERBOSE)
        win_prob, entropy, lambda_, c_lambda = results
        ENTROPYs[i] = entropy
        LAMBDAs[i] = lambda_
        C_LAMBDAs[i] = c_lambda
        # print(f'Entropy:{entropy:.5g}')
        # print(f'WinProb:{win_prob:.5g}')
        # print(f'Lambda:{lambda_:.5g}')
        # print(f'C_lambda:{c_lambda:.5g}')

    if VERBOSE or SAVEDATA:
        metadata = ['delta', 'qbound', 'entropy', 'lambda', 'c_lambda']
        headline = '\t'.join(metadata)
    
    if VERBOSE:
        print(headline)
        
        for delta, score, entropy, lambda_, c_lambda in \
            zip(DELTAs, SCOREs, ENTROPYs, LAMBDAs, C_LAMBDAs):
            line = '\t'.join((f'{delta:.5g}', f'{score:.6f}', f'{entropy:.6f}',
                             f'{lambda_:.6f}', f'{c_lambda:.6f}'))
            print(line)

        print("\n")
    
    # Write to file
    if SAVEDATA:
        data = np.vstack((DELTAs, SCOREs, ENTROPYs, LAMBDAs, C_LAMBDAs)).T
        # MAX_P_WIN = f'MAX_WIN_PROB\n{max_win:.5g}'
        # PRETXT = bytes(MAX_P_WIN, 'utf-8') + b'\n'

        COM = f'wbc_{RTYPE}'
        WTOL = f'wtol_{WIN_TOL:.0e}'
        QUAD = f'M_{M*2}'
        OUT_FILE = f'{COM}-{WTOL}-{QUAD}.csv'
        OUT_PATH = os.path.join(OUT_DIR, OUT_FILE)
        
        if os.path.exists(OUT_PATH):
            with open(OUT_PATH, 'ab') as file:
                file.write(b'\n')
                # file.write(PRETXT)
                np.savetxt(file, data, fmt='%.5g', delimiter=',', header=headline)
        else:
            with open(OUT_PATH, 'wb') as file:
                # file.write(PRETXT)
                np.savetxt(file, data, fmt='%.5g', delimiter=',', header=headline)

if TIMMING:
    toc = time.time()
    print(f'Elapsed time: {toc-tic}')