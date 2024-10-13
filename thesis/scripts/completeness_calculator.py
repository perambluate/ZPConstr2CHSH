from joblib import Parallel, delayed
from functools import partial
from scipy.stats import binom
import numpy as np
from math import floor, exp
import os, sys

### Add current directory to Python path
# (Note: this works when running the command in the repo dir.)
sys.path.append('.')
from common_func.plotting_helper import cls_inp_map, \
                                        fin_rate_testing, \
                                        opt_with_gamma

### Max winning probability from honest strategies
MAX_WIN_MAP = {'chsh': 0.8535, '1': 0.8294, '2a': 0.8125,
               '2b': 0.8125, '2b_swap': 0.8125, '2c': 0.8039,
               '3a': 0.7951, '3b': 0.7837}

def completeness1(n, gamma, wtol, wexp):
    k = floor(n*gamma*(wexp-wtol))
    p= gamma*wexp
    return binom.cdf(k, n, p)

def completeness2(n, gamma, wtol, wexp):
    k = floor(n*gamma*(1-(wexp-wtol)))
    p= gamma*(1-wexp)
    return 1 - binom.cdf(k, n, p)

def nonaborting_prob(n_zero, n, gamma, wtol, ztol):
    epsw = 1 - exp(-2*(wtol**2)*gamma*n)
    epsz = 1 - exp(-2*((ztol/2)**2)*gamma*n)
    return epsw * (epsz**n_zero)

TYPE = 'blind'                      # Type of randomness (one/two/blind)
EPSILON = 1e-8                      # Smoothness of smooth min entropy (related to secrecy)
INP_DIST = [1/4]*4                  # Input distribution
d_K = 4 if TYPE == 'two' else 2     # Dimention of the target sys

### Number of rounds
Ns = [1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14]

### Freely tunable params
#### Param related to alpha-Renyi entropy
BETA_SLICE = 50
BETAs = np.geomspace(1e-4, 1e-11, num=BETA_SLICE)

#### Testing ratio
GAMMA_SLICE = 100
GAMMAs = 1/np.geomspace(1, 100000, num=GAMMA_SLICE)

#### Another tunable param
NU_PRIME_SLICE = 50
NU_PRIMEs = np.linspace(1, 0.1, num=NU_PRIME_SLICE)

### Get data to compute finite rate
EPS = f'eps_{EPSILON:.0e}'
PDG = 'pdgap_5e-05'
QUAD = 'M_12'

### Data paths
DATA_DIR = './thesis/data/best_setup/old'

if TYPE == 'blind':
    HEAD = 'br'
    BEST_CLS = '3b'
elif TYPE =='one' or TYPE == 'two':
    HEAD = TYPE[0] + 'pr'
    BEST_CLS = '2a'

CLASS_INPUT_MAP = cls_inp_map(TYPE)

### Classes of correlations
CLASSES = ['chsh', BEST_CLS]

data2save = []

### Run over all classes in CLASSES
for cls in CLASSES:
    ### Specify the file record the data
    CLS = cls
    NP = 'N_2'
    inp = CLASS_INPUT_MAP[cls]
    if TYPE == 'one':
        INP = f'x_{inp}'
    else:
        INP = f'xy_{inp}'
    
    MTF_DATA_FILE = f'{HEAD}-{CLS}-{INP}-{PDG}-{QUAD}-{NP}-wplb.csv'
    MTF_DATA_PATH = os.path.join(DATA_DIR, MTF_DATA_FILE)
    ### Load data
    data_mtf = np.genfromtxt(MTF_DATA_PATH, delimiter=",", skip_header = 1)
    # print(data_mtf)
    
    ### Choose min-tradeoff function with expected winning prob
    if len(data_mtf.shape) == 1:
        data_mtf = np.array([data_mtf])

    #### For different w_exp (or other parameters) in the data
    for i in range(data_mtf.shape[0]):
        if cls == 'chsh':
            zero_tol = 0
            shift = 0
        else:
            zero_tol = data_mtf[i][0]
            shift = 1
        
        win_tol, max_win, min_win, win_prob, asym_rate, lambda_ = data_mtf[i][shift:shift+6]

        ### Construct key rate function with fixed param (only leave n, beta tunable)
        kr_func = partial(fin_rate_testing, asym_rate = asym_rate, d_K = d_K,
                                            lambda_ = lambda_, epsilon = EPSILON,
                                            zero_tol = zero_tol, zero_class=cls,
                                            max_win = max_win, min_win = min_win)

        def task(n):
            rate, gamma = opt_with_gamma(n, beta_arr = BETAs, nup_arr = NU_PRIMEs, gam_arr = GAMMAs,
                                         inp_dist = INP_DIST, fin_rate_func = kr_func)
            epscom1 = completeness1(n, gamma, win_tol, MAX_WIN_MAP[cls])
            # epscom2 = completeness2(n, gamma, win_tol, MAX_WIN_MAP[cls])
            return (cls, str(i+1), n, rate, gamma, epscom1)
        
        results = Parallel(n_jobs=8, verbose=0)(delayed(task)(n) for n in Ns)
        data2save += results

data2save = np.array(data2save, dtype=[('', 'U4')]*2+[('', np.double)]*4)
print(f"class settings n     rate     gamma     epscom ")
for row in data2save:
    print(f"{row[0]:5} {row[1]:8} {row[2]:.1g} {row[3]:.4e} {row[4]:.5e} {row[5]:.5e}")

OUT_FILE = f'completeness-{TYPE}.csv'
OUT_DIR = './thesis/data/completeness'
OUT_PATH = os.path.join(OUT_DIR, OUT_FILE)
HEADER = ', '.join(['class', 'settings', 'n', 'rate', 'gamma', 'epscom'])
mode = 'ab' if os.path.exists(OUT_PATH) else 'wb'
with open(OUT_PATH, mode) as file:
    np.savetxt(file, data2save, fmt=["%s"]*2+["%.5g"]*4, delimiter=',', header=HEADER)

