import sys
import os
import re

### Add directory to Python path
sys.path.append('.')

from ZPConstr2CHSH.DIRandomness import *

RAND_TYPE = 'loc'
NPA_LEVEL = 2
RADAU_QUAD_PARAMS = {'n_quad': 12, 'endpoint': .999, 'keep_endpoint': True}
PARALLEL_PARAMS = {'nthread_sdp': 4, 'nthread_quad': 4}
PRIMAL_DUAL_GAP = 5e-5
SOLVER_CONFIG = ['mosek', {'dparam.intpnt_co_tol_rel_gap': PRIMAL_DUAL_GAP,
                            'iparam.num_threads': PARALLEL_PARAMS['nthread_sdp'],
                            'iparam.infeas_report_level': 4}]
TRUNCATE_DIGIT = 6
VERBOSE = 1
INP_PROB = np.array([[1/4, 1/4],[1/4, 1/4]])
WIN_C_BOUND = 0.25
SAVEDATA = True
TIMMING = True
OUT_DIR = './thesis/data/best_setup/new'

TOL_PARAMS = [{'wtol': 1e-5, 'ztol': 1e-6},
              {'wtol': 1e-3, 'ztol': 1e-3}]
N_PARAM = 2

## Printing precision of numpy arrays
np.set_printoptions(precision=5)
CHSH_predicate = [[[[0,1], [1,0]],
                   [[0,1], [1,0]]],
                  [[[0,1], [1,0]],
                   [[1,0], [0,1]]]]
CHSH_predicate = np.array(CHSH_predicate)*INP_PROB

CLASSES = ['chsh', BEST_ZPC_CLASS[RAND_TYPE]]

for zpc_class in CLASSES:
    print(f'NSQB class: {zpc_class}')
    best_inp = ZPC_BEST_INP_MAP[RAND_TYPE][zpc_class]
    if RAND_TYPE == 'loc':
        best_inp = (best_inp, 0)
    
    n_zp_checks = len(ZP_POSITION_MAP.get(zpc_class, []))
    ZERO_TOLs, WIN_TOLs, MIN_WINs, MAX_WINs, WIN_PROBs, ENTROPYs = np.zeros((6, N_PARAM))
    if n_zp_checks:
        LAMBDAs = np.zeros((N_PARAM, n_zp_checks+1))
    else:
        LAMBDAs = np.zeros(N_PARAM)
    C_LAMBDAs = np.zeros(N_PARAM)

    for i in range(N_PARAM):
        WIN_TOLs[i] = TOL_PARAMS[i]['wtol']
        ZERO_TOLs[i] = TOL_PARAMS[i]['ztol']
    
        protocol_params = {'kappa': zpc_class, 'inputs': best_inp,
                           'WP_predicate': CHSH_predicate,
                           'WP_check_direction': 'lb',
                           'wexp': None, 'wtol': WIN_TOLs[i], 'ztol': ZERO_TOLs[i]}

        if VERBOSE >= 2:
            print('Compute win prob quantum bound')
        
        Calculator = DIRandZPC_Calculator(RAND_TYPE, npa_level = NPA_LEVEL,
                                          nthread_sdp = PARALLEL_PARAMS['nthread_sdp'],
                                          nthread_quad = PARALLEL_PARAMS['nthread_quad'],
                                          radau_quad_params = RADAU_QUAD_PARAMS,
                                          protocol_params = protocol_params,
                                          solver_config = SOLVER_CONFIG,
                                          verbose = VERBOSE)

        min_win_Q = Calculator.win_prob_Q_bound('min', TRUNCATE_DIGIT)
        print(f'Win prob Q lower bound: {min_win_Q}')
        MIN_WINs[i] = min_win_Q
        max_win_Q = Calculator.win_prob_Q_bound('max', TRUNCATE_DIGIT)
        print(f'Win prob Q upper bound: {max_win_Q}')
        MAX_WINs[i] = max_win_Q
        WIN_PROBs[i] = max_win_Q

        if TIMMING:
            tic = time.time()

        Calculator.update_protocol_params({'wexp': WIN_PROBs[i]})
        entropy, lambda_, c_lambda = Calculator.asymptotic_rate_with_Lagrange_dual()
        ENTROPYs[i] = entropy
        LAMBDAs[i] = lambda_
        C_LAMBDAs[i] = c_lambda


    if VERBOSE or SAVEDATA:
        if zpc_class == 'chsh':
            metadata = ['win_tol', 'max_win', 'min_win', 'win_prob',
                        'entropy', 'lambda', 'c_lambda']
            data = np.vstack((WIN_TOLs, MAX_WINs, MIN_WINs, WIN_PROBs,
                              ENTROPYs, np.squeeze(LAMBDAs), C_LAMBDAs)).T
        else:
            lambda_zp = filter(lambda k: 'ZP' in k, Calculator.constr_dict.keys())
            lambda_zp = [re.sub('ZP_check', 'lambda', name) for name in lambda_zp]
            metadata = ['zero_tol', 'win_tol', 'max_win', 'min_win', 'win_prob',
                        'entropy', 'lambda', *lambda_zp, 'c_lambda']
            data = np.vstack((ZERO_TOLs, WIN_TOLs, MAX_WINs, MIN_WINs, WIN_PROBs,
                                  ENTROPYs, np.array(LAMBDAs).T, C_LAMBDAs)).T
        headline = ', '.join(metadata)
        
        if VERBOSE:
            print(headline)

            for data_per_line in data:
                if zpc_class == 'chsh':
                    wp_tol = data_per_line[0]
                    print(f'{wp_tol:.5g}\t'+
                            '\t'.join([f'{val:.6f}' for val in data_per_line[1:]]))
                else:
                    zp_tol = data_per_line[0]
                    wp_tol = data_per_line[1]
                    print(f'{zp_tol:.5g}\t'+f'{wp_tol:.5g}\t'+
                                '\t'.join([f'{val:.6f}' for val in data_per_line[1:]]))

        # Write to file
        if SAVEDATA:
            CLS = zpc_class
            if RAND_TYPE == 'loc':
                INP = f"x_{best_inp[0]}"
            else:
                INP = f"xy_{best_inp[0]}{best_inp[1]}"
            PDGAP = f"pdgap_{PRIMAL_DUAL_GAP:.0e}"
            QUAD = f"M_{Calculator.radau_quad_params['n_quad']}"
            NP = f"N_{N_PARAM}"
            TAIL = "wplb"
            OUT_FILE = f'{RAND_TYPE}-{CLS}-{INP}-{PDGAP}-{QUAD}-{NP}-{TAIL}.csv'
            OUT_PATH = os.path.join(OUT_DIR, OUT_FILE)
            
            mode = 'ab' if os.path.exists(OUT_PATH) else 'wb'
            n_cols = data.shape[1] - 2
            with open(OUT_PATH, mode) as file:
                np.savetxt(file, data, fmt=["%.5g"]*2+["%.6f"]*n_cols,
                           delimiter=',', header=headline)


if TIMMING:
    toc = time.time()
    print(f'Elapsed time: {toc-tic}')

