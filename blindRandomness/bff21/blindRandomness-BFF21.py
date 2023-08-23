"""
    This script is used to compute the von Neumann entropy H(A|XYBE)
      with the Brown-Fawzi-Fawzi method (arXiv:2106.13692v2)
      for blind randomness in the point of view of other player in CHSH non-local game
"""
import ncpol2sdpa as ncp
from sympy.physics.quantum.dagger import Dagger
import math
import numpy as np
from joblib import Parallel, delayed
import time
import sys, os

### Add current directory to Python path
sys.path.append('..')
from common_func.SDP_helper import *

SAVEDATA = False                # Set the data into file
TIMMING = True                  # True for timming
OUTDIR = './data/bff21_zero'   # Folder for the data to save

LEVEL = 2                       # NPA relaxation level
M = 6                           # Num of terms in Gauss-Radau quadrature = 2*M
VERBOSE = 2                     # Relate to how detail of the info will be printed
N_WORKER_QUAD = 4               # Number of workers for parallelly computing quadrature
N_WORKER_LOOP = 1               # Number of workers for the outer loop
N_WORKER_SDP = 4                # Number of threads for solving a single SDP
PRIMAL_DUAL_GAP = 1e-5          # Allowable gap between primal and dual
SOLVER_CONFIG = ['mosek', {'dparam.presolve_tol_x': 1e-10,
                           'dparam.intpnt_co_tol_rel_gap': PRIMAL_DUAL_GAP,
                           'iparam.num_threads': N_WORKER_SDP,
                           'iparam.infeas_report_level': 4}]
# SOLVER_CONFIG = ['sdpa']
ACCURATE_DIGIT = 4              # Achievable precision of the solver
WIN_TOL = 1e-4                  # Relax the precise winning prob constraint to a range with epsilon
ZERO_PROB = 1e-9                # Treat this value as zero for zero probability constraints
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

## Classes of correlations to run
ZERO_CLASS = ['2a'] # ['1','2a','2b','2b_swap','2c','3a','3b']
## Tolerable errors for zero-probability constraints
ZERO_TOLs = [1e-9] #[1e-9, 1e-5, 1e-3]

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
    # print(f'Zero probability tol err: {ZERO_ERR}')
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

for zero_class in ZERO_CLASS:

    zero_pos = CLASS_ZERO_POS.get(zero_class, [])

    if VERBOSE:
        print(f'Correlation type: {zero_class}')
        if zero_pos:
            print(f'Zero positions: {zero_pos}')
        else:
            print(f'No zero constraints')

    # zero_constraints = zeroConstr(P, zero_pos, ZERO_PROB)

    # # Compute the quantum bound first
    # sdp_Q = ncp.SdpRelaxation(P.get_all_operators(), verbose=max(VERBOSE-3, 0))
    # sdp_Q.get_relaxation(level=LEVEL, objective=-winProb(P),
    #                     substitutions = P.substitutions,
    #                     momentinequalities = zero_constraints)
    # sdp_Q.solve(*SOLVER_CONFIG)

    # if VERBOSE:
    #     print(sdp_Q.status, sdp_Q.primal, sdp_Q.dual)

    # if sdp_Q.status != 'optimal' and sdp_Q.status != 'primal-dual feasible':
    #     print('Cannot compute quantum bound correctly!', file=sys.stderr)
    #     break

    # P_WIN_Q = truncate(-sdp_Q.primal, ACCURATE_DIGIT)
    # P_WINs = [CLASS_MAX_WIN[zero_class], 0.77]
    # NUM_SLICE = len(P_WINs)

    INPUTS = [OPT_INPUT_MAP.get(zero_class, (0,0))] # [(0,0), (0,1), (1,0), (1,1)]

    for inputs in INPUTS:
        if VERBOSE:
            print(f'Chosen input xy={inputs[0]}{inputs[1]}')
        
        GAMMAs = [0.25]
        for gamma in GAMMAs:
            GAMMA_A, GAMMA_B = math.sqrt(gamma), math.sqrt(gamma)
            inp_probs = np.zeros((2,2))
            inp_probs[inputs[0]][inputs[1]] = GAMMA_A*GAMMA_B
            inp_probs[inputs[0]^1][inputs[1]] = (1-GAMMA_A)*GAMMA_B
            inp_probs[inputs[0]][inputs[1]^1] = GAMMA_A*(1-GAMMA_B)
            inp_probs[inputs[0]^1][inputs[1]^1] = (1-GAMMA_A)*(1-GAMMA_B)

        # zero_tol_arr = ZERO_TOLs if zero_class != 'chsh' else [ZERO_PROB]

        # for zero_tol in zero_tol_arr:
            zero_tol = ZERO_PROB
            if VERBOSE:
                print(f'Zero probability tolerance: {zero_tol}')
            zero_constraints = zeroConstr(P, zero_pos, zero_tol)

            if VERBOSE:
                print('Start compute winning probability quantum bound>>>')

            # Compute the quantum bound first
            sdp_Q = ncp.SdpRelaxation(P.get_all_operators(), verbose=max(VERBOSE-3, 0))
            sdp_Q.get_relaxation(level=LEVEL, objective=-winProb(P, inp_probs=inp_probs),
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
            
            max_p_win = truncate(-sdp_Q.primal, ACCURATE_DIGIT)
            p_win_mid = (max_p_win + C_BOUND) / 2
            ### Generate 20 winning probability: 14 in the interval (p_win_mid, max_p_win];
            ###                                  6  in the interval [C_BOUND], p_win_mid]
            # P_WINs = [*np.linspace(max_p_win, p_win_mid, 6, endpoint=False),
            #           *np.linspace(p_win_mid, C_BOUND, 4)]
            P_WINs = [max_p_win]
            NUM_SLICE = len(P_WINs)

            results = Parallel(n_jobs=N_WORKER_LOOP, verbose = 0)(
                                delayed(singleRoundEntropy)('blind', P, Z_ab, M, inputs,
                                                            winProb, win_prob,
                                                            scenario = scenario,
                                                            inp_probs = inp_probs,
                                                            win_tol = WIN_TOL,
                                                            zero_class = zero_class,
                                                            zero_tol = zero_tol,
                                                            substs = substs,
                                                            extra_monos = extra_monos,
                                                            level = LEVEL,
                                                            n_worker_quad = N_WORKER_QUAD,
                                                            solver_config = SOLVER_CONFIG,
                                                            verbose = VERBOSE) \
                                                            for win_prob in P_WINs)

            #print(results)

            if VERBOSE or SAVEDATA:
                if zero_class == 'chsh':
                    metadata = ['winning_prob', 'entropy', 'lambda', 'c_lambda']
                else:
                    zero_pos_str = zeroPos2str(zero_pos)
                    zero_pos_str = [f'lambda_{pos}' for pos in zero_pos_str]
                    metadata = ['winning_prob', 'entropy', 'lambda', *zero_pos_str, 'c_lambda']
            
            if VERBOSE:
                headline = '\t'.join(metadata)
                print(headline)
                
                for win_prob, entropy, lambda_, c_lambda in results:                
                    if zero_class == 'chsh':
                        lambda_str = f'{lambda_:.5g}'         
                    else:
                        lambda_vals = [f'{val:5g}' for val in lambda_]
                        lambda_str = '\t'.join(lambda_vals)
                    line = f'{win_prob:.5g}\t{entropy:.5g}\t'+lambda_str+f'\t{c_lambda:.5g}'
                    print(line)

                print("\n")

            # Write to file
            if SAVEDATA:
                if zero_class == 'chsh':
                    data = np.array(results)
                else:
                    data = [[win_prob, entropy, *lambda_, c_lambda] \
                            for win_prob, entropy, lambda_, c_lambda in results]
                    data = np.array(data)

                HEADER = ', '.join(metadata)
                MAX_P_WIN = f'MAX_WIN_PROB\n{max_p_win:.5g}'

                COM = 'biased_inp'
                CLS = zero_class
                INP = f'xy_{inputs[0]}{inputs[1]}'
                WTOL = f'wtol_{WIN_TOL:.0e}'
                ZTOL = f'ztol_{zero_tol:.0e}'
                QUAD = f'M_{M*2}'
                GAM = f'gam_{gamma*100:.0f}'
                OUTFILE = f'{COM}-{CLS}-{INP}-{GAM}-{WTOL}-{ZTOL}-{QUAD}.csv'
                OUTPATH = os.path.join(OUTDIR, OUTFILE)
                
                if os.path.exists(OUTPATH):
                    with open(OUTPATH, 'ab') as file:
                        file.write(b'\n')
                        file.write(bytes(MAX_P_WIN, 'utf-8') + b'\n')
                        np.savetxt(file, data, fmt='%.5g', delimiter=',', header=HEADER)
                else:
                    with open(OUTPATH, 'wb') as file:
                        file.write(bytes(MAX_P_WIN, 'utf-8') + b'\n')
                        np.savetxt(file, data, fmt='%.5g', delimiter=',', header=HEADER)
                
if TIMMING:
    toc = time.time()
    print(f'Elapsed time: {toc-tic}')
