"""
    This script is used to compute the randomness (including global, local, and blind)
    in terms of the von Neumann entropy (global: H(AB|XYE), global: H(AB|XYE), global: H(A|XYBE))
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
sys.path.append('.')
from common_func.SDP_helper import *

SAVEDATA = True                 # Set the data into file
TIMMING = True                  # True for timming
OUT_DIR = './WBC_inequality'    # Folder for the data to save
RTYPE = 'one'
LEVEL = 2                       # NPA relaxation level
M = 6                           # Number of terms in Gauss-Radau quadrature = 2*M
VERBOSE = 2                     # Relate to how detail of the info will be printed
N_WORKER_QUAD = 1               # Number of workers for parallelly computing quadrature
N_WORKER_SDP = 6                # Number of threads for solving a single SDP
N_WORKER_NPA = 6                # Number of threads for generating a NPA moment matrix
PRIMAL_DUAL_GAP = 5e-5          # Allowable gap between primal and dual 1e-6
SOLVER_CONFIG = ['mosek', {'dparam.intpnt_co_tol_rel_gap': PRIMAL_DUAL_GAP,
                           'iparam.num_threads': N_WORKER_SDP,
                           'iparam.infeas_report_level': 4}]
ACCURATE_DIGIT = 6              # Achievable precision of the solver
WIN_TOL = 1e-4                  # Relax the precise winning prob constraint to a range with epsilon 2e-5 1e-4

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
        Q_BOUNDs[i] = qbound

        if not qub_succ:
            print('Cannot compute quantum bound correctly!', file=sys.stderr)
        
        score = qbound
        SCOREs[i] = score

        
        results = singleRoundEntropy(RTYPE, P, Z, M, BEST_INP, bell_func, score,
                                     scenario = scenario, win_tol = WIN_TOL,
                                     substs = substs, extra_monos = extra_monos,
                                     level = LEVEL, quad_end = True, quad_ep = .999,
                                     n_worker_quad = N_WORKER_QUAD,
                                     n_worker_npa = N_WORKER_NPA,
                                     solver_config = SOLVER_CONFIG, verbose = VERBOSE)
        entropy, lambda_, c_lambda = results
        ENTROPYs[i] = entropy
        LAMBDAs[i] = lambda_
        C_LAMBDAs[i] = c_lambda

    if VERBOSE or SAVEDATA:
        metadata = ['delta', 'qbound', 'score', 'entropy', 'lambda', 'c_lambda']
        headline = ', '.join(metadata)
    
    if VERBOSE:
        print(headline)        
        for data_per_line in zip(DELTAs, Q_BOUNDs, SCOREs, ENTROPYs, LAMBDAs, C_LAMBDAs):
            line = f'{data_per_line[0]:.5g}'+'\t'.join([f'{val:.6f}' for val in data_per_line[1:]])
            print(line)

        print("\n")
    
    # Write to file
    if SAVEDATA:
        data = np.vstack((DELTAs, SCOREs, ENTROPYs, LAMBDAs, C_LAMBDAs)).T

        COM = f'wbc-{RTYPE}'
        WTOL = f'wtol_{WIN_TOL:.0e}'
        PDGAP = f'pdgap_{PRIMAL_DUAL_GAP:.0e}'
        QUAD = f'M_{M*2}'
        NP = f'N_{N_POINT}'
        OUT_FILE = f'{COM}-{WTOL}-{PDGAP}-{QUAD}-{NP}.csv'
        OUT_PATH = os.path.join(OUT_DIR, OUT_FILE)
        
        mode = 'ab' if os.path.exists(OUT_PATH) else 'wb'
        with open(OUT_PATH, mode) as file:
            np.savetxt(file, data, fmt='%.6f', delimiter=',', header=headline)

if TIMMING:
    toc = time.time()
    print(f'Elapsed time: {toc-tic}')

