import numpy as np
import os, sys
import re

### Add current directory to Python path
sys.path.append('.')
from ZPConstr2CHSH.finiteRateCalculator import *
from ZPConstr2CHSH.DIRandomness import BEST_ZPC_CLASS, ZPC_BEST_INP_MAP

RAND_TYPE = 'loc'
EPSILON = 1e-8
BOUNDED_EPSCOM = False
EPSCOM_UB = 1e-6
NTHREADS = 8
N_NARROW = False

### Freely tunable params
#### Param related to alpha-Renyi entropy
BETA_SLICE = 50
BETAs = np.geomspace(1e-4, 1e-11, num=BETA_SLICE)

#### Testing ratio
GAMMA_SLICE = 1000
GAMMAs = np.geomspace(1, 1e-5, num=GAMMA_SLICE)

#### Another tunable param
NU_PRIME_SLICE = 11
NU_PRIMEs = np.linspace(0.4, 0.6, num=NU_PRIME_SLICE)

### Num of rounds
MAX_N = 1e15
N_SLICE = 200
if not N_NARROW:
    if RAND_TYPE == 'bli':
        MIN_N = 1e6
    elif RAND_TYPE in ('loc', 'glo'):
        MIN_N = 5e5
    Ns = np.geomspace(MIN_N, MAX_N, num=N_SLICE, dtype=int)
    N_POINT = re.sub('e\+0+|e\+', 'e', f'N_{MIN_N:.0e}_{MAX_N:.0e}_{N_SLICE}')

CLASSES = ['chsh', BEST_ZPC_CLASS[RAND_TYPE]]

for cls in CLASSES:
    ### Read param from file
    if RAND_TYPE == 'loc':
        HEAD = 'opr'
    elif RAND_TYPE =='glo':
        HEAD = 'tpr'
    else:
        HEAD = 'br'
    
    best_inp = ZPC_BEST_INP_MAP[RAND_TYPE][cls]
    if RAND_TYPE == 'loc':
        INP = f"x_{best_inp}"
    else:
        INP = f"xy_{best_inp[0]}{best_inp[1]}"
    PDG = 'pdgap_5e-05'
    QUAD = 'M_12'
    NP = 'N_2'
    TAIL = 'wplb'
    PARAM_FILE = f'{HEAD}-{cls}-{INP}-{PDG}-{QUAD}-{NP}-{TAIL}.csv'
    ### Data paths
    DATA_DIR = './thesis/data/best_setup/old'
    PARAM_FILE_PATH = os.path.join(DATA_DIR, PARAM_FILE)

    ### Load data
    data_mtf = np.genfromtxt(PARAM_FILE_PATH, delimiter=",", skip_header = 1)

    for data_line in data_mtf:
        if cls != 'chsh':
            ztol = data_line[0]
            data_line = data_line[1:]
        else:
            ztol = 0

        wtol, max_win, min_win, wexp, ent, lambda_ = data_line[0:6]
        finrate_params = {'kappa': cls, 'asym_rate': ent,
                          'WP_check_direction': 'lb',
                          'inp_prob': [[1/4, 1/4],[1/4, 1/4]],
                          'n': None, 'gamma': None, 'wexp': wexp,
                          'wtol': wtol, 'ztol': ztol, 'eps': EPSILON,
                          'lambda': lambda_}
        wp_Qbound = {'max': max_win, 'min': min_win}

        Calculator = finiteRateCalculator(RAND_TYPE, finrate_params, wp_Qbound, NTHREADS)
        if not BOUNDED_EPSCOM:
            EPSCOM_UB = -1

        ### Num of rounds
        if N_NARROW:
            MIN_N = 50/(wtol**2)
            Ns = np.geomspace(MIN_N, MAX_N, num=N_SLICE, dtype=int)
            N_POINT = re.sub('e\+0+|e\+', 'e', f'N_{MIN_N:.0e}_{MAX_N:.0e}_{N_SLICE}')

        optFRs, optGAMMAs, optBETAs, optNUPs = \
                    Calculator.opt_fin_rate_for_ns(Ns, BETAs, NU_PRIMEs, GAMMAs, EPSCOM_UB)

        HEADER = 'rounds, rate, gamma, beta, nuprime'
        data2save = np.column_stack((Ns, optFRs, optGAMMAs, optBETAs, optNUPs))

        HEAD = f'fin-{RAND_TYPE}'
        WEXP = f'w_{wexp*10000:.0f}'.rstrip('0')
        ZTOL = f'ztol_{ztol:.0e}' if cls != 'chsh' else ''
        WTOL = f'wtol_{wtol:.0e}'
        EPS = f'eps_{EPSILON:.0e}'
        EPSCUB = f'epscub_{EPSCOM_UB:.0e}' if BOUNDED_EPSCOM else ''
        if N_NARROW and 'narrow_n' not in TAIL:
            TAIL += '-narrow_n'
        out_name_elements = [HEAD, cls, WEXP, EPS, WTOL, ZTOL, QUAD, EPSCUB, N_POINT, TAIL]
        OUT_NAME = '-'.join([s for s in out_name_elements if s != ''])
        OUT_FILE = f'{OUT_NAME}.csv'
        OUT_DIR = './thesis/data/fin_rate'
        OUT_PATH = os.path.join(OUT_DIR, OUT_FILE)
        np.savetxt(OUT_PATH, data2save, fmt='%.5g', delimiter=',', header=HEADER)

