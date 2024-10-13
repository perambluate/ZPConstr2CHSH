from scipy import stats
import sys
import os
import re

### Add directory to Python path
sys.path.append('.')

from ZPConstr2CHSH.DIRandomness import *

RAND_TYPE = 'bli'
NPA_LEVEL = 2
RADAU_QUAD_PARAMS = {'n_quad': 12, 'endpoint': .999, 'keep_endpoint': True}
PARALLEL_PARAMS = {'nthread_sdp': 4, 'nthread_quad': 4}
PRIMAL_DUAL_GAP = 5e-5
SOLVER_CONFIG = ['mosek', {'dparam.intpnt_co_tol_rel_gap': PRIMAL_DUAL_GAP,
                            'iparam.num_threads': PARALLEL_PARAMS['nthread_sdp'],
                            'iparam.infeas_report_level': 4}]
VERBOSE = 1
INP_PROB = np.array([[1/4, 1/4],[1/4, 1/4]])
WIN_C_BOUND = 0.75
SAVEDATA = True
TIMMING = True
OUT_DIR = './thesis/data/asym_rate/new'

def intervalSpacingBeta(interval, num_points, endpoint = True, a = 1.2, b = 0.8):
    dist = stats.beta(a, b)
    pp = np.linspace(*dist.cdf([0, 1]), num=num_points, endpoint = endpoint)
    spacing_arr = interval[0] + dist.ppf(pp) * (interval[1] - interval[0])
    return spacing_arr

## Printing precision of numpy arrays
np.set_printoptions(precision=5)

## Predicate for CHSH game
CHSH_predicate = [[[[0,1], [1,0]],
                   [[0,1], [1,0]]],
                  [[[0,1], [1,0]],
                   [[1,0], [0,1]]]]
CHSH_predicate = np.array(CHSH_predicate)*INP_PROB

CLASSES = ['1', '2a', '2b', '2b_swap', '2c', '3a', '3b', 'chsh']
if RAND_TYPE == 'glo' and ('2b' in CLASSES) and ('2b_swap' in CLASSES):
    CLASSES.remove('2b_swap')

for zpc_class in CLASSES:
    print(f'NSQB class: {zpc_class}')
    best_inp = ZPC_BEST_INP_MAP[RAND_TYPE][zpc_class]
    if RAND_TYPE == 'loc':
        best_inp = (best_inp, 0)
    
    n_zp_checks = len(ZP_POSITION_MAP.get(zpc_class,[]))
    protocol_params = {'kappa': zpc_class, 'inputs': best_inp,
                       'WP_predicate': CHSH_predicate,
                       'WP_check_direction': 'lb',
                       'wexp': None, 'wtol': 1e-5, 'ztol': 1e-6}

    if VERBOSE >= 2:
        print('Compute win prob quantum bound')
    Calculator = DIRandZPC_Calculator(RAND_TYPE, npa_level = NPA_LEVEL,
                                      nthread_sdp = PARALLEL_PARAMS['nthread_sdp'],
                                      nthread_quad = PARALLEL_PARAMS['nthread_quad'],
                                      radau_quad_params = RADAU_QUAD_PARAMS,
                                      protocol_params = protocol_params,
                                      solver_config = SOLVER_CONFIG,
                                      verbose = VERBOSE)

    min_win_Q = Calculator.win_prob_Q_bound('min', 6)
    print(f'Win prob Q lower bound: {min_win_Q}')
    max_win_Q = Calculator.win_prob_Q_bound('max', 6)
    print(f'Win prob Q upper bound: {max_win_Q}')

    N_POINT = 20
    WIN_PROBs = intervalSpacingBeta((max_win_Q, WIN_C_BOUND),
                                    num_points = N_POINT,
                                    a = 0.5, b = 1.2)
    ENTROPYs = np.zeros(N_POINT)
    if n_zp_checks:
        LAMBDAs = np.zeros((N_POINT, n_zp_checks+1))
    else:
        LAMBDAs = np.zeros(N_POINT)
    C_LAMBDAs = np.zeros(N_POINT)

    if TIMMING:
        tic = time.time()

    for i in range(N_POINT):
        win_prob = WIN_PROBs[i]
        print(f'Win prob (lb): {win_prob}')
        Calculator.update_protocol_params({'wexp': win_prob})
        entropy, lambda_, c_lambda = Calculator.asymptotic_rate_with_Lagrange_dual()
        ENTROPYs[i] = entropy
        LAMBDAs[i] = lambda_
        C_LAMBDAs[i] = c_lambda


    if VERBOSE or SAVEDATA:
        if zpc_class == 'chsh':
            metadata = ['win_prob', 'entropy', 'lambda', 'c_lambda']
            data = np.vstack((WIN_PROBs, ENTROPYs, np.squeeze(LAMBDAs), C_LAMBDAs)).T
        else:
            lambda_zp = filter(lambda k: 'ZP' in k, Calculator.constr_dict.keys())
            lambda_zp = [re.sub('ZP_check', 'lambda', name) for name in lambda_zp]
            metadata = ['win_prob', 'entropy', 'lambda', *lambda_zp, 'c_lambda']
            data = np.vstack((WIN_PROBs, ENTROPYs, np.array(LAMBDAs).T, C_LAMBDAs)).T
        headline = ', '.join(metadata)
        
        if VERBOSE:
            print(headline)

            for data_per_line in data:
                print('\t'.join([f'{val:.6f}' for val in data_per_line[:]]))

        # Write to file
        if SAVEDATA:
            BEFORE_HEADLINE = f'MIN_WIN_PROB\n{min_win_Q:.5f}' + '\n' \
                                + f'MAX_WIN_PROB\n{max_win_Q:.5f}'

            CLS = zpc_class
            if RAND_TYPE == 'loc':
                INP = f"x_{best_inp[0]}"
            else:
                INP = f"xy_{best_inp[0]}{best_inp[1]}"
            WTOL = f"wtol_{protocol_params['wtol']:.0e}"
            ZTOL = f"ztol_{protocol_params['ztol']:.0e}"
            PDGAP = f"pdgap_{PRIMAL_DUAL_GAP:.0e}"
            QUAD = f"M_{Calculator.radau_quad_params['n_quad']}"
            NP = f"N_{N_POINT}"
            TAIL = "wplb"
            OUT_FILE = f'{RAND_TYPE}-{CLS}-{INP}-{WTOL}-{ZTOL}-{PDGAP}-{QUAD}-{NP}'
            
            if TAIL:
                OUT_FILE = f'{OUT_FILE}-{TAIL}.csv'
            else:
                OUT_FILE += '.csv'
            
            OUT_PATH = os.path.join(OUT_DIR, OUT_FILE)
            
            mode = 'ab' if os.path.exists(OUT_PATH) else 'wb'
            n_cols = data.shape[1] - 2
            with open(OUT_PATH, mode) as file:
                file.write(bytes(BEFORE_HEADLINE, 'utf-8') + b'\n')
                np.savetxt(file, data, fmt=["%.5g"]*2+["%.6f"]*n_cols,
                           delimiter=',', header=headline)


if TIMMING:
    toc = time.time()
    print(f'Elapsed time: {toc-tic}')

