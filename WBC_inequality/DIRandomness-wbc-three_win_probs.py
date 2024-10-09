import ncpol2sdpa as ncp
import numpy as np
from sympy.physics.quantum.dagger import Dagger
import chaospy
from math import log, pi, cos, sin
from time import time
from joblib import Parallel, delayed
import sys
import os
sys.path.append('..')
from common_func.SDP_helper import *
from DIStrategyUtils.QuantumStrategy import *

def ShanonnEntropy(p):
    return -np.sum(np.multiply(p, np.log2(p, np.zeros_like(p), where=(p!=0))))


SAVEDATA = True                 # Set the data into file
TIMMING = True                  # True for timming
OUT_DIR = './'        # Folder for the data to save
LEVEL = 2                       # NPA relaxation level
M = 9                           # Num of terms in Gauss-Radau quadrature = 2*M
VERBOSE = 2                     # Relate to how detail of the info will be printed
N_WORKER_QUAD = 4               # Number of workers for parallelly computing quadrature
N_WORKER_LOOP = 1               # Number of workers for the outer loop
N_WORKER_SDP = 4                # Number of threads for solving a single SDP
PRIMAL_DUAL_GAP = 1e-5          # Allowable gap between primal and dual
SOLVER_CONFIG = ['mosek', {'dparam.intpnt_co_tol_rel_gap': PRIMAL_DUAL_GAP,
                           'iparam.num_threads': N_WORKER_SDP,
                           'iparam.infeas_report_auto': 1,
                           'iparam.infeas_report_level': 10}]
ACCURATE_DIGIT = 5
WIN_TOL = 1e-4
RTYPE = 'blind'

## Printing precision of numpy arrays
np.set_printoptions(precision=5)

# Generate nodes, weights of quadrature up to 2*M terms
QUAD_ENDPOINT = 1 #.9999
T, W = chaospy.quad_gauss_radau(M, chaospy.Uniform(0, QUAD_ENDPOINT), QUAD_ENDPOINT)
T = T[0]
NUM_NODE = len(T)

# Setup of the scenario for Alice and Bob
configA = [2,2]
configB = [2,2]
dim_A = 2
dim_B = 2
dim_X = len(configA)
dim_Y = len(configB)
SCENARIO = (configA, configB)
OUT_COMB = np.array(np.meshgrid(*SCENARIO)).T
INP_CONFIG = tuple(len(SCENARIO[i]) for i in range(len(SCENARIO)))
INP_COMB = genCombinations(INP_CONFIG)

# Choose uniform probability distribution
INP_PROB = np.ones(INP_CONFIG)/np.prod(INP_CONFIG)

################################### Init for BFF21 computation ###################################
# Generate probability distribution P(ab|xy): P([a,b],[x,y])
P = ncp.Probability(configA, configB)

# Generate the set of POVMs for each measurement per party.
#   Note that we use CG form here, for example if Alice first measurement has two outcomes,
#   i.e., {A0, I - A0}, then the POVM for second outcome is no need to record.
#   In the case of CHSH, we have A = [[A0], [A1]]; B = [[B0], [B1]].
A, B = P.parties
# Generate the set of substitutable operators, e.g., A0B0 = B0A0.
substs = P.substitutions

# Inputs to gen randomness
if RTYPE == 'blind' or RTYPE == 'two':
    BEST_INP = (0,0)
elif RTYPE == 'one':
    BEST_INP = 0

# Generate Eve's operators
if RTYPE == 'one':
    Zs = ncp.generate_operators('Z', dim_A, hermitian=False)
else:
    Zs = ncp.generate_operators('Z', dim_A*dim_B, hermitian=False)

# Make Eve's operators commute with Alice and Bob's
for a in P.get_all_operators():
    for z in Zs:
        substs.update({z*a: a*z, Dagger(z)*a: a*Dagger(z)})

# Generate ABZ, ABZ*, ABZZ*, ABZ*Z as extra monomials
extra_monos = []
for a in ncp.flatten(A):
    for z in Zs:
        if RTYPE == 'one':
            extra_monos += [a*Dagger(z)*z, a*z*Dagger(z)]
        for b in ncp.flatten(B):
            extra_monos += [a*b*z, a*b*Dagger(z)]
            if RTYPE in ['two', 'blind']:
                extra_monos += [a*b*z*Dagger(z), a*b*Dagger(z)*z]

# Construct the NPA relaxation for later SDP problem
ops = P.get_all_operators()+Zs

# Coefficients of the Bell inequalities
### Coefficients for CHSH Bell inequalities
CHSH_Ceoff = np.zeros((2,2,2,2))
CHSH_Ceoff[0,0] = [[0, 1], [1, 0]]
CHSH_Ceoff[0,1] = [[0, 1], [1, 0]]
CHSH_Ceoff[1,0] = [[0, 1], [1, 0]]
CHSH_Ceoff[1,1] = [[1, 0], [0, 1]]
CHSH_Ceoff = CHSH_Ceoff

### Coefficients for I_delta family Bell inequalities
delta = pi/6
alpha = 1/sin(delta)
beta = 1/cos(2*delta)
I_delta_Coeff = np.zeros((2,2,2,2))
I_delta_Coeff[0,0] = [[1, 0], [0, 1]]
I_delta_Coeff[0,1] = [[alpha, 0], [0, alpha]]
I_delta_Coeff[1,0] = [[alpha, 0], [0, alpha]]
I_delta_Coeff[1,1] = [[0, beta], [beta, 0]]

# Setup the strategy that gives the correlation and Bell value
### Params characterising the state and measurements to create correlation
ANG_MAP = {'chsh': {'theta': pi/4, 'alpha': 0, 'alpha1': -pi/2,
                    'beta': 3*pi/4, 'beta1': -3*pi/4},
           'I_delta': {'theta': pi/4, 'alpha': 0, 'alpha1': delta+pi/2,
                       'beta': pi/2, 'beta1': -delta}}

CLASS = 'I_delta'

angles = ANG_MAP[CLASS]
strategy = StrategiesInCHSHScenario(angles)

if VERBOSE >= 3:
    print(strategy.correlations)

####################### Score constraint #######################
SCORES = ['1', 'alpha', 'beta']
BellFunction = lambda coeff: sum([coeff[x,y][a,b] * P([a,b],[x,y]) for a in range(2) for b in range(2) \
                                                                    for x in range(2) for y in range(2)])
# print(win_prob_expr)
# win_prob = strategy.BellVal(CHSH_Ceoff)
# win_prob = strategy.BellVal(I_delta_Coeff)
# Bell_val_constr = [win_prob_expr - win_prob]
ScoreConstr = []
ScoreCoeffs = []
p_wins = []
MaxScore = [INP_PROB[0,0], INP_PROB[0,1] + INP_PROB[1,0], INP_PROB[1,1]]
# print(MaxScore)
for j in range(3):
    Coeff = np.zeros((2,2,2,2))
    if j == 0:      # for (x,y) = (0,0)
        Coeff[0,0] = [[1,0], [0,1]]
    elif j == 1:    # for (x,y) = (0,1) or (1,0)
        Coeff[0,1] = [[1,0], [0,1]]
        Coeff[1,0] = [[1,0], [0,1]]
    else:           # for (x,y) = (1,1)
        Coeff[1,1] = [[0,1], [1,0]]

    Coeff = Coeff * INP_PROB
    win_tol = MaxScore[j] * WIN_TOL
    ScoreCoeffs += [Coeff]        
    win_prob_expr = BellFunction(Coeff)
    win_prob = strategy.BellVal(Coeff)
    p_wins += [win_prob]
    ScoreConstr += [win_prob_expr - win_prob + win_tol,
                    -win_prob_expr + win_prob + win_tol]

entropy = 0 #-1/(len(T)**2 * log(2)) + W[-1]/(T[-1]*log(2))
entropy_quad = []
p_win_quad = []
lambda_quad = []

### Optimize for each node of the Gauss-Radau quadrature
def task(node_idx):
    coeff = W[node_idx]/(T[node_idx] * log(2))
    obj_func = OBJ_FUNC_MAP[RTYPE]
    obj = obj_func(P.parties, BEST_INP, Zs, T[node_idx])
    
    # sdp.set_objective(obj)

    sdp = ncp.SdpRelaxation(ops, parallel = True,
                            number_of_threads = N_WORKER_SDP, verbose = max(VERBOSE-3, 0))
    sdp.get_relaxation(level = LEVEL, objective = obj,
                        substitutions = substs,
                        momentequalities = [], # Bell_val_constr,
                        momentinequalities = ScoreConstr,
                        extramonomials = extra_monos)
    sdp.solve(*SOLVER_CONFIG)
    if VERBOSE >= 2:
        print(sdp.status, sdp.primal, sdp.dual)

    assert sdp.status == 'optimal' or ('feasible' in sdp.status and 'infeasible' not in sdp.status), \
        f'Status {sdp.status}, not promised solved'
    
    if sdp.status != 'optimal' and VERBOSE >= 1:
        print('Solution does not reach optimal!')

    ent_in_quad = coeff * (1 + sdp.dual)

    if VERBOSE >= 3:
        print("Probability from SDP:")
        for y in range(dim_Y):
            for b in range(dim_B):
                s = f'\t'.join([f"{sdp[P([a,b],[x,y],['A','B'])]:.5f}" \
                                for x in range(dim_X) for a in range(dim_A)])
                print(s)

    p_wins_in_quad = []
    lambdas_in_quad = []
    for j in range(int(len(ScoreConstr)/2)):
        id = 2*j
        score_coeff = ScoreCoeffs[j]
        win_prob_expr = BellFunction(score_coeff)
        p_wins_in_quad += [sdp[win_prob_expr]]
        p_win_dual_vec = [sdp.get_dual(ScoreConstr[id]), sdp.get_dual(ScoreConstr[id+1])]
        p_win_dual_vec = np.squeeze(p_win_dual_vec)
        p_win_dual_var = np.abs(p_win_dual_vec[0]-p_win_dual_vec[1])

        lambdas_in_quad += [coeff * p_win_dual_var]

    return ent_in_quad, p_wins_in_quad, lambdas_in_quad

if TIMMING:
        tic = time()

results =  Parallel(n_jobs=N_WORKER_QUAD, verbose=max(VERBOSE-1, 0))( \
                                            delayed(task)(i) for i in range(NUM_NODE))

# print(results)
entropy_quad, p_win_quad, lambda_quad = zip(*results)

entropy = sum(entropy_quad)
max_p_win = np.max(p_win_quad, axis=0)
lambda_vec = np.sum(np.array(lambda_quad), axis=0)
c_lambda = entropy - np.dot(lambda_vec, max_p_win)

if VERBOSE:
    print(f"Scores: {p_wins}\n")
    print(f"Entropy: {entropy:.5g}\n")
    print(f"lambda_vec: {lambda_vec}\n")
    print(f"C_lambda: {c_lambda:.5g}\n")


if TIMMING:
    toc = time()
    print(f'Elapsed time: {toc-tic}')

p_wins_str = [f'win_{s}' for s in SCORES]
lambda_str = [f'lambda_{s}' for s in SCORES]
metadata = [*p_wins_str, 'entropy', *lambda_str, 'c_lambda']
headline = ', '.join(metadata)

if SAVEDATA:
    data = np.array([*p_wins, entropy, *lambda_vec, c_lambda])
    data = data[np.newaxis]         # Make an 1D array 2D

    HEAD = 'three_scores'
    QUAD = f'M_{M*2}'
    WTOL = f'wtol_{WIN_TOL:.0e}'
    OUT_FILE = f'{HEAD}-wbc-{RTYPE}-{WTOL}-{QUAD}.csv'
    OUT_PATH = os.path.join(OUT_DIR, OUT_FILE)
    
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, 'ab') as file:
            file.write(b'\n')
            np.savetxt(file, data, fmt='%.5g', delimiter=',', header=headline)
    else:
        with open(OUT_PATH, 'wb') as file:
            np.savetxt(file, data, fmt='%.5g', delimiter=',', header=headline)

