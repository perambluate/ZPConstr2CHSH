import numpy as np
import json
import sys
import os

TOP_DIR = './other_examples'

### Add directory to Python path
sys.path.append('.')
from ZPConstr2CHSH.DIRandomness import *
from ZPConstr2CHSH.finiteRateCalculator import *

# Params related to accuracy & performance of computation
NPA_LEVEL = 2
RADAU_QUAD_PARAMS = {'n_quad': 12, 'endpoint': .999, 'keep_endpoint': True}
PARALLEL_PARAMS = {'nthread_sdp': 4, 'nthread_quad': 4}
PRIMAL_DUAL_GAP = 5e-5
SOLVER_CONFIG = ['mosek', {'dparam.intpnt_co_tol_rel_gap': PRIMAL_DUAL_GAP,
                            'iparam.num_threads': PARALLEL_PARAMS['nthread_sdp'],
                            'iparam.infeas_report_level': 4}]
TRUNCATE_DIGIT = 6
np.set_printoptions(precision=5)    # Printing precision of numpy arrays
VERBOSE = 1
TIMING = True

# Protocol-related params
RAND_TYPE = 'loc'
CLASS = '2a'
inputs = (0,0)
WTOL = 1e-5
ZTOL = 1e-6

INP_PROB = np.array([[1/4, 1/4],[1/4, 1/4]])
CHSH_predicate = [[[[0,1], [1,0]],
                   [[0,1], [1,0]]],
                  [[[0,1], [1,0]],
                   [[1,0], [0,1]]]]
CHSH_predicate = np.array(CHSH_predicate)*INP_PROB


protocol_params = {'chi': RAND_TYPE,
                   'kappa': CLASS, 'inputs': inputs,
                   'WP_predicate': CHSH_predicate.tolist(),
                   'WP_check_direction': 'lb',
                   'wexp': None, 'wtol': WTOL, 'ztol': ZTOL}

with open(os.path.join(TOP_DIR, 'protocol.json'), 'w') as f:
    json.dump(protocol_params, f)

if TIMING:
    tic = time.time()
############################# Asymptotic rate computation #############################
AsymRateCalculator = DIRandZPC_Calculator(RAND_TYPE, npa_level = NPA_LEVEL,
                                  nthread_sdp = PARALLEL_PARAMS['nthread_sdp'],
                                  nthread_quad = PARALLEL_PARAMS['nthread_quad'],
                                  radau_quad_params = RADAU_QUAD_PARAMS,
                                  protocol_params = protocol_params,
                                  solver_config = SOLVER_CONFIG,
                                  verbose = VERBOSE)

print('Quantum bound computation')
min_win_Q = AsymRateCalculator.win_prob_Q_bound('min', TRUNCATE_DIGIT)
print(f'Win prob Q lower bound: {min_win_Q}')
max_win_Q = AsymRateCalculator.win_prob_Q_bound('max', TRUNCATE_DIGIT)
print(f'Win prob Q upper bound: {max_win_Q}')
print('')

win_prob = max_win_Q
AsymRateCalculator.update_protocol_params({'wexp': win_prob})

print('Asymptotic rate computation')
entropy, lambda_, c_lambda = AsymRateCalculator.asymptotic_rate_with_Lagrange_dual()
print(f'lambda_vector: {lambda_}')
print(f'c_lambda: {c_lambda}')
print('')

############################# Finite rate computation #############################
N = 2e12
EPSILON = 1e-8
EPSCOM_UB = 1e-6

finrate_params = {'kappa': CLASS, 'asym_rate': entropy,
                  'WP_check_direction': 'lb',
                  'inp_prob': [[1/4, 1/4],[1/4, 1/4]],
                  'n': N, 'gamma': None, 'wexp': win_prob,
                  'wtol': WTOL, 'ztol': ZTOL, 'eps': EPSILON,
                  'lambda': lambda_[0]}
wp_Qbound = {'max': max_win_Q, 'min': min_win_Q}

# Freely tunable params
## Param related to alpha-Renyi entropy
BETA_SLICE = 50
BETAs = np.geomspace(1e-4, 1e-11, num=BETA_SLICE)

## Testing ratio
GAMMA_SLICE = 1000
GAMMAs = np.geomspace(1, 1e-5, num=GAMMA_SLICE)

## Another tunable param
NU_PRIME_SLICE = 11
NU_PRIMEs = np.linspace(0.4, 0.6, num=NU_PRIME_SLICE)

print('Finite rate computation')
FinRateCalculator = finiteRateCalculator(RAND_TYPE, finrate_params, wp_Qbound, nthreads = 8)
opt_rate, opt_gam, opt_beta, opt_nup = \
        FinRateCalculator.fin_rate_opt_epscom_bound(n = N, epscom_bound = EPSCOM_UB,
                                                    beta_list = BETAs,
                                                    nup_list = NU_PRIMEs,
                                                    gam_list = GAMMAs)

print(f'Finite rate: {opt_rate:.5g}')
print(f'Optimal gamma: {opt_gam}')
print(f'Optimal beta: {opt_beta}')
print(f'Optimal nu_prime: {opt_nup}')
print('')

if TIMING:
    toc = time.time()
    print(f'Elapsed time: {toc-tic}')

finrate_params.update({'fin_rate': opt_rate,
                       'gamma': opt_gam,
                       'beta': opt_beta,
                       'nu_prime': opt_nup})
with open(os.path.join(TOP_DIR, 'finrate_params.json'), 'w') as f:
    json.dump(finrate_params, f)

