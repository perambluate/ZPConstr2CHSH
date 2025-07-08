import numpy as np
import json
import sys
import os
from argparse import ArgumentParser

TOP_DIR = './examples'

### Add directory to Python path
sys.path.append('.')
from ZPConstr2CHSH.DIRandomness import *
from ZPConstr2CHSH.finiteRateCalculator import *

def asym_rate_computation(protocol_params):
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

    win_prob = max_win_Q
    AsymRateCalculator.update_protocol_params({'wexp': win_prob})

    print('Asymptotic rate computation >>>>>>')
    entropy, lambda_, c_lambda = AsymRateCalculator.asymptotic_rate_with_Lagrange_dual()
    print(f'lambda_vector: {lambda_}')
    print(f'c_lambda: {c_lambda}')

    return entropy, win_prob, max_win_Q, min_win_Q, lambda_

def finite_rate_computation(finrate_params, wp_Qbound):
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

    print('Finite rate computation >>>>>>')
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

# Params for asymptotic rate computation
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

# Params for finite rate computation
N = 2e12
EPSILON = 1e-8
EPSCOM_UB = 1e-6

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-t", dest="test_class", required=True,
                        choices=['asym', 'fin', 'both'],
                        help="choose the class to test")
    args = parser.parse_args()

    if args.test_class == 'asym':
        # Prepare testing data
        if not os.path.exists(os.path.join(TOP_DIR, 'protocol.json')):
            protocol_params = {'chi': RAND_TYPE,
                            'kappa': CLASS, 'inputs': inputs,
                            'WP_predicate': CHSH_predicate.tolist(),
                            'WP_check_direction': 'lb',
                            'wexp': None, 'wtol': WTOL, 'ztol': ZTOL}

            with open(os.path.join(TOP_DIR, 'protocol.json'), 'w') as f:
                json.dump(protocol_params, f)
        else:
            with open(os.path.join(TOP_DIR, 'protocol.json'), 'r') as f:
                protocol_params = json.load(f)
        
        tic = time.time()
        entropy, win_prob, max_win_Q, min_win_Q, lambda_ = asym_rate_computation(protocol_params)
        toc = time.time()
        print(f'Elapsed time: {toc-tic}')

    elif args.test_class == 'fin':
        # Prepare testing data
        if not os.path.exists(os.path.join(TOP_DIR, 'finrate_params.json')) or \
            not os.path.exists(os.path.join(TOP_DIR, 'wp_Qbound.json')):
            print('Testing data not found, computing it...')
            tic = time.time()
            entropy, win_prob, max_win_Q, min_win_Q, lambda_ = asym_rate_computation(protocol_params)
            toc = time.time()
            print(f'Elapsed time (asym): {toc-tic}\n')
            
            finrate_params = {'kappa': CLASS, 'asym_rate': entropy,
                            'WP_check_direction': 'lb',
                            'inp_prob': [[1/4, 1/4],[1/4, 1/4]],
                            'n': N, 'gamma': None, 'wexp': win_prob,
                            'wtol': WTOL, 'ztol': ZTOL, 'eps': EPSILON,
                            'lambda': lambda_[0]}
            wp_Qbound = {'max': max_win_Q, 'min': min_win_Q}

            # Save params for future testing
            with open(os.path.join(TOP_DIR, 'finrate_params.json'), 'w') as f:
                json.dump(finrate_params, f)
            
            with open(os.path.join(TOP_DIR, 'wp_Qbound.json'), 'w') as f:
                json.dump(wp_Qbound, f)
        else:
            with open(os.path.join(TOP_DIR, 'finrate_params.json'), 'r') as f:
                finrate_params = json.load(f)
            
            with open(os.path.join(TOP_DIR, 'wp_Qbound.json'), 'r') as f:
                wp_Qbound = json.load(f)
        
        tic = time.time()
        finite_rate_computation(finrate_params, wp_Qbound)
        toc = time.time()
        print(f'Elapsed time (fin): {toc-tic}')
    
    elif args.test_class == 'both':
        # Prepare testing data
        if not os.path.exists(os.path.join(TOP_DIR, 'protocol.json')):
            protocol_params = {'chi': RAND_TYPE,
                            'kappa': CLASS, 'inputs': inputs,
                            'WP_predicate': CHSH_predicate.tolist(),
                            'WP_check_direction': 'lb',
                            'wexp': None, 'wtol': WTOL, 'ztol': ZTOL}

            with open(os.path.join(TOP_DIR, 'protocol.json'), 'w') as f:
                json.dump(protocol_params, f)
        else:
            with open(os.path.join(TOP_DIR, 'protocol.json'), 'r') as f:
                protocol_params = json.load(f)
        
        tic = time.time()
        entropy, win_prob, max_win_Q, min_win_Q, lambda_ = asym_rate_computation(protocol_params)
        toc = time.time()
        print(f'Elapsed time (asym): {toc-tic}\n')

        if not os.path.exists(os.path.join(TOP_DIR, 'finrate_params.json')) or \
            not os.path.exists(os.path.join(TOP_DIR, 'wp_Qbound.json')):
            print('Use params from asymptotic rate computation to compute finite rates...')
            finrate_params = {'kappa': CLASS, 'asym_rate': entropy,
                            'WP_check_direction': 'lb',
                            'inp_prob': [[1/4, 1/4],[1/4, 1/4]],
                            'n': N, 'gamma': None, 'wexp': win_prob,
                            'wtol': WTOL, 'ztol': ZTOL, 'eps': EPSILON,
                            'lambda': lambda_[0]}
            wp_Qbound = {'max': max_win_Q, 'min': min_win_Q}

            # Save params for future testing
            with open(os.path.join(TOP_DIR, 'finrate_params.json'), 'w') as f:
                json.dump(finrate_params, f)
            
            with open(os.path.join(TOP_DIR, 'wp_Qbound.json'), 'w') as f:
                json.dump(wp_Qbound, f)
        else:
            with open(os.path.join(TOP_DIR, 'finrate_params.json'), 'r') as f:
                finrate_params = json.load(f)
            
            with open(os.path.join(TOP_DIR, 'wp_Qbound.json'), 'r') as f:
                wp_Qbound = json.load(f)
        
        tic = time.time()
        finite_rate_computation(finrate_params, wp_Qbound)
        toc = time.time()
        print(f'Elapsed time (fin): {toc-tic}')
