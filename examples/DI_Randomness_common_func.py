import ncpol2sdpa as ncp
import numpy as np
from sympy.physics.quantum.dagger import Dagger
import chaospy
from math import log, log2, pi, cos, sin, sqrt
from time import time
from joblib import Parallel, delayed
from functools import partial
import sys
import os
import re

## Add current directory to Python path
sys.path.append('.')
from common_func.SDP_helper import *
from DIStrategyUtils.QuantumStrategy import *

TIMMING = True                  # True for timming
OUT_DIR = './'                  # Folder for the data to save
LEVEL = 2                       # NPA relaxation level
M = 6                           # Num of terms in Gauss-Radau quadrature = 2*M
QUAD_ENDPOINT = .999            # Right endpoint of Gauss-Radau quadrature
VERBOSE = 1                     # Relate to how detail of the info will be printed
N_WORKER_QUAD = 4               # Number of workers for parallelly computing quadrature
N_WORKER_SDP = 4                # Number of threads for solving a single SDP
N_WORKER_NPA = 4                # Number of threads for generating a NPA moment matrix
PRIMAL_DUAL_GAP = 5e-5          # Allowable gap between primal and dual
SOLVER_CONFIG = ['mosek', {'dparam.intpnt_co_tol_rel_gap': PRIMAL_DUAL_GAP,
                           'iparam.num_threads': N_WORKER_SDP,
                           'iparam.infeas_report_level': 4}]
TRUNCATE_DIGIT = 6
RAND_TYPE = 'one'
WP_TOL = 1e-5
ZP_TOL = 1e-6

## Printing precision of numpy arrays
np.set_printoptions(precision=5)

# Setup of the scenario for Alice and Bob
configA = [2,2]
configB = [2,2]
dim_A = 2
dim_B = 2
if RAND_TYPE in ('one', 'blind'):
    dim_K = dim_A
else:
    dim_K = dim_A*dim_B
dim_X = len(configA)
dim_Y = len(configB)
SCENARIO = (configA, configB)
INP_CONFIG = tuple(len(SCENARIO[i]) for i in range(len(SCENARIO)))

# Choose uniform probability distribution
INP_PROB = np.ones(INP_CONFIG)/np.prod(INP_CONFIG)

# Inputs to gen randomness
if RAND_TYPE == 'one':
    INPUT = 0
else:
    INPUT = (0,0)

############################## Init for BFF21 computation ##############################
# Generate probability distribution P(ab|xy): P([a,b],[x,y])
P = ncp.Probability(configA, configB)

# Generate the set of POVMs for each measurement per party.
#   Note that we use CG form here, for example if Alice first measurement has two outcomes,
#   i.e., {A0, I - A0}, then the POVM for second outcome is no need to record.
#   In the case of CHSH, we have A = [[A0], [A1]]; B = [[B0], [B1]].
A, B = P.parties
# Generate the set of substitutable operators, e.g., A0B0 = B0A0.
substs = P.substitutions

####################### Score (winning-/zero-probability) constraints #######################
## CHSH game predicate
CHSH_Coeff = np.zeros((2,2,2,2))
CHSH_Coeff[0,0] = [[0, 1], [1, 0]]
CHSH_Coeff[0,1] = [[0, 1], [1, 0]]
CHSH_Coeff[1,0] = [[0, 1], [1, 0]]
CHSH_Coeff[1,1] = [[1, 0], [0, 1]]
CHSH_Coeff = CHSH_Coeff*INP_PROB

def BellFunction(P, coeff):
    expr = 0
    for x in range(dim_X):
        for y in range(dim_Y):
            for a in range(dim_A):
                for b in range(dim_B):
                    expr += coeff[x,y][a,b] * P([a,b],[x,y])
    return expr

CHSH_BellFunc = lambda P: BellFunction(P, CHSH_Coeff)
CHSH_BellExpr = BellFunction(P, CHSH_Coeff)

## Positions of zeros probability constraints
CLASS = '2a'
CLASS_ZERO_POS = zero_position_map()
ZP_POS = zero_position_map().get(CLASS, [])

zero_constr = []
if ZP_POS:
    if VERBOSE:
        print(f'Zero tolerance: {ZP_TOL}')
    zero_constr = [ZP_TOL - P(*pos) for pos in ZP_POS]
        
####################### Find CHSH winning probability quantum bound #######################
# Init SdpRelaxation
sdp_Q = ncp.SdpRelaxation(P.get_all_operators(), verbose=max(VERBOSE-1, 0))

# Generate moment matrix and constraints up to given level.
#   Note that the ncpol package only provide minimization,
#   so a negative sign is need to do the maximization.
sdp_Q.get_relaxation(level = LEVEL, objective = -CHSH_BellExpr,
                    substitutions = P.substitutions,
                    momentinequalities = zero_constr)
sdp_Q.solve(*SOLVER_CONFIG)

print('Status\tPrimal\tDual')
print(f'{sdp_Q.status}\t{sdp_Q.primal}\t{sdp_Q.dual}')

# Maximal quantum winning probability estimated with NPA set of given level
max_win = truncate(-sdp_Q.primal, TRUNCATE_DIGIT)

# Generate Eve's operators
Zs = ncp.generate_operators('Z', dim_K, hermitian=False)

# Make Eve's operators commute with Alice and Bob's
# Generate ABZ, ABZ*, ABZZ*, ABZ*Z as extra monomials
extra_monos = []
for a in P.get_all_operators():
    for z in Zs:
        substs.update({z*a: a*z, Dagger(z)*a: a*Dagger(z)})

for a in ncp.flatten(A):
    for b in ncp.flatten(B):
        for z in Zs:
            extra_monos += [a*b*z, a*b*Dagger(z), a*b*z*Dagger(z), a*b*Dagger(z)*z]

if TIMMING:
    tic = time()

entropy, _lambda, c_lambda = singleRoundEntropy(RAND_TYPE, P, Zs, M, INPUT, scenario = SCENARIO,
                                                win_prob_func = CHSH_BellFunc, win_prob = max_win,
                                                zero_class = CLASS, win_tol = WP_TOL, zero_tol = ZP_TOL,
                                                substs = substs, extra_monos = extra_monos,
                                                level = LEVEL, quad_end = True, quad_ep = QUAD_ENDPOINT,
                                                n_worker_quad = N_WORKER_QUAD, n_worker_npa = N_WORKER_NPA,
                                                solver_config = SOLVER_CONFIG, verbose = VERBOSE)

if TIMMING:
    toc = time()
    print(f'Elapsed time: {toc-tic}')

