import numpy as np
from math import log, log2, sqrt, e, floor
from functools import partial
from collections.abc import Iterable
from joblib import Parallel, delayed
from time import time
import re
import sys, os

sys.path.append('.')
from minTradeoffProp import *
from common_func.plotting_helper import inp_rand_consumption

TOP_DIR = './WBC_inequality'
DATA_DIR = os.path.join(TOP_DIR, 'data')

TIMMING = True                  # True for timming
LEVEL = 4                       # NPA relaxation level
VERBOSE = 1                     # Relate to how detail of the info will be printed
N_WORKER_LOOP = 4               # Number of workers for the outer loop
N_WORKER_SDP = 4                # Number of threads for solving a single SDP
PRIMAL_DUAL_GAP = 1e-5          # Allowable gap between primal and dual
SOLVER_CONFIG = ['mosek', {'dparam.intpnt_co_tol_rel_gap': PRIMAL_DUAL_GAP,
                            'iparam.num_threads': N_WORKER_SDP,
                            'iparam.infeas_report_auto': 1,
                            'iparam.infeas_report_level': 10}]

def fin_rate_correction(n, beta, nu_vec, gamma, lambda_vec, c_lambda, dim_K, eps, score_coeffs, win_tol,
                        scenario, level = 2, solver_config = SOLVER_CONFIG, verbose = VERBOSE):
    MTF = MinTradeoffFunction(scenario, lambda_vec, c_lambda, score_coeffs,
                              level, solver_config, verbose)
    max_f = MTF.Max(nu_vec, gamma)
    min_f = MTF.Min_Sigma()
    var_f = MTF.Var_Sigma(nu_vec, gamma)
    
    ln2 = log(2)
    logdK = 2*log2(dim_K)
    log_nonabort_prob = log2(eps)
    # print(f'log P(non-abort): {log_nonabort_prob}')
    max2min_f = max_f - min_f

    if sqrt(1 - eps**2) == 1:
        eps_term = -log2(1/2*eps**2)
    else:
        eps_term = -log2(1 - sqrt(1 - eps**2) )
    
    with np.errstate(over='raise'):
        try:
            two2power_max2min_f = 2 ** (logdK + max2min_f)
            K_beta = (two2power_max2min_f ** beta) * (log(two2power_max2min_f + e**2)) **3

        except FloatingPointError:
            try:
                K_beta = (2 ** (beta * (logdK + max2min_f) ) ) * ( (logdK + max2min_f)/log2(e) ) **3
            except FloatingPointError:
                return 0

    V = log2(1+2*(dim_K**2)) + sqrt(2+var_f)
    # print(f'V: {V}')
    # print(f'eps_term: {eps_term}')
    # print(f'K_beta: {K_beta}')
    Delta = (ln2/2) * beta * (V**2) \
            + 1/n * ((1+beta)/beta * eps_term + (1+2*beta)/beta * log_nonabort_prob) \
            + 1/(6*ln2) * (beta**2)/((1-beta)**3) * K_beta

    return Delta

def fin_rate(n, beta, nu, gamma, asym_rate, lambda_vec, c_lambda,
             dim_K, eps, score_coeffs, win_tol, scenario, inp_prob):
    rate = asym_rate - fin_rate_correction(n, beta, nu, gamma, lambda_vec, c_lambda, dim_K,
                                           eps, score_coeffs, win_tol, scenario) \
                     - inp_rand_consumption(gamma, inp_prob)
    return max(rate, 0)

if __name__ == '__main__':
    RTYPE = 'one'
    configA = [2,2]
    configB = [2,2]
    SCENARIO = (configA, configB)
    INP_CONFIG = tuple(len(SCENARIO[i]) for i in range(len(SCENARIO)))
    INP_PROB = np.ones(INP_CONFIG)/np.prod(INP_CONFIG)
    
    NU_VEC = [1, 0, 0]
    n = 1e6
    EPSILON = 1e-12
    WIN_TOL = 1e-4
    win_tol = np.array([1/4, 1/2, 1/4])*WIN_TOL
    ### Get data to compute finite rate
    EPS = f'eps_{EPSILON:.0e}'
    WTOL = f'wtol_{WIN_TOL:.0e}'
    QUAD = 'M_18'
    HEAD = 'three_scores'
    WTOL = f'wtol_{WIN_TOL:.0e}'
    MTFDATAFILE = f'{HEAD}-wbc-{RTYPE}-{WTOL}-{QUAD}.csv'
    MTFDATAPATH = os.path.join(DATA_DIR, MTFDATAFILE)
    data_mtf = np.genfromtxt(MTFDATAPATH, delimiter=",", skip_header = 1, max_rows = 1)
    NUM_SCORE = 3
    asym_rate = data_mtf[NUM_SCORE]
    lambda_vec = data_mtf[-(NUM_SCORE+1):-1]
    c_lambda = data_mtf[-1]
    print(lambda_vec)

    ScoreCoeffs = []
    for s in range(NUM_SCORE):
        Coeff = np.zeros((2,2,2,2))
        if s == 0:      # for (x,y) = (0,0)
            Coeff[0,0] = [[0,1], [1,0]]
        elif s == 1:    # for (x,y) = (0,1) or (1,0)
            Coeff[0,1] = [[0,1], [1,0]]
            Coeff[1,0] = [[0,1], [1,0]]
        else:           # for (x,y) = (1,1)
            Coeff[1,1] = [[1,0], [0,1]]

        ScoreCoeffs += [Coeff]

    ### Num of rounds
    if RTYPE == 'blind':
        N_RANGE = (1e6, 1e15)
    elif RTYPE == 'one':
        N_RANGE = (5e5, 1e15)
    elif RTYPE == 'two':
        N_RANGE = (5e5, 1e15)
    N_SLICE = 100
    Ns = np.geomspace(*N_RANGE, num=N_SLICE)
    N_POINT = re.sub('e\+0+|e\+', 'e', f'N_{N_RANGE[0]:.0e}_{N_RANGE[1]:.0e}_{N_SLICE}')
    #### Testing ratio
    GAMMA_SLICE = 20
    GAMMAs = 1/np.geomspace(10, 10000, num=GAMMA_SLICE)

    BETA_SLICE = 20
    BETAs = np.geomspace(1e-4, 1e-11, num=BETA_SLICE)
    
    fr_func = partial(fin_rate, asym_rate = asym_rate, lambda_vec = lambda_vec, c_lambda = c_lambda,
                      dim_K = 2, eps = EPSILON, score_coeffs = ScoreCoeffs, win_tol = win_tol,
                      scenario = SCENARIO, inp_prob = INP_PROB)
    
    def opt_fin_rate(n, beta_arr, gam_arr):
        return max([fr_func(n, beta, NU_VEC, gamma) for beta in beta_arr for gamma in gam_arr])
    
    if TIMMING:
        tic = time()
    
    FRs = Parallel(n_jobs=N_WORKER_LOOP, verbose = 0)(delayed(opt_fin_rate)(n, BETAs, GAMMAs) for n in Ns)

    if TIMMING:
        toc = time()
        print(f'Elapsed time: {toc-tic}')
    
    data2save = np.column_stack((Ns, FRs))
    HEADER = 'rounds, rate'
    print(HEADER)
    for n, rate in zip(Ns, FRs):
        line = f'{n:.5g}\t{rate:.5g}'
        print(line)
    
    OUTCSV = f'{RTYPE}-qbound-{EPS}-{WTOL}-{QUAD}-{N_POINT}-1.csv'
    OUTCSV_PATH = os.path.join(DATA_DIR, OUTCSV)
    np.savetxt(OUTCSV_PATH, data2save, fmt='%.5g', delimiter=',', header=HEADER)

