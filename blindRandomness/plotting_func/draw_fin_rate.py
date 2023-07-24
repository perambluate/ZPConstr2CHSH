"""
    A script to plot finite rates with optimized correction terms (beta and nu_prime).
      - Notations (case insensitive)
        n     : number of rounds
        w_exp : expected winning probability
        wtol  : tolerance of winning probability
        ztol  : zero tolerance
        quad  : number of quadratures
        gamma : ratio of testing rounds to whole rounds
        class : class of specific correlation defined by zero-probability constraints
        beta, nu_prime : parameters to optimize for the correction terms
      - Directory and file names
        Note the directories where the data save maybe different,
            please change <TOP_DIR> and <DATA_DIR> according to the path you save.
        We assume the data files are named in the following formate:
            lr_bff21-<class>-xy_<inputs>-M_<num_quad>-wtol_<win_tol>-ztol_<zero_tol>.csv
        Choose the path <OUT_DIR> to where you want to output the figrues.
        The output figures are saved with the following naming formate:
            <COMMON>(-<CLASS>)(-<WIN_EXP>)-<EPS>-<WTOL>-<GAM>-<QUAD>-<TAIL>.png
        (Note: The two parts in the parentheses are sometimes omited depending on the content of the figure.)
      - Choices of default settings
        Change <settings> to apply the settings we used to produce the figures in our paper.
            settings = 1 : a sinlge plot of class 1, 2c, 3b, and CHSH with w_exp at quantum bound
            settings = 2 : many figures for all classes, each contains two subplots corresponding to
                            w_exp at quantum bound and 0.77 with three zero tolerances (1e-9, 1e-5, 1e-3)
            settings = 3 : a plot with a break axis for class 1, 2c, 3b, and CHSH at w_exp = 0.7525;
                            an extra zero tolerance 1e-3 for class 3b to show the robustness
"""

from matplotlib.ticker import ScalarFormatter,LogLocator
from matplotlib import pyplot as plt
from matplotlib import cycler
from functools import partial
from joblib import Parallel, delayed
import numpy as np
import itertools
import math
import os, sys


# Extend ScalarFormatter
class MyScalarFormatter(ScalarFormatter):
    # Override '_set_format' with your own
    def _set_format(self):
        self.format = '%.2f'  # Show 2 decimals without redundant zeros

### Choice of default plottig settings
settings = 2

### An option to print values of data
PRINT_DATA = False

### An option to the data into a csv file
SAVE_CSV = False

### To save file or not
SAVE = True
### To show figure or not
SHOW = False

### Change the GUI backend to get rid of the multi-threading runtime error with tkinter
if not SHOW:
    plt.switch_backend('agg')

### Data paths
TOP_DIR = './'
if settings == 3:
    DATA_DIR = os.path.join(TOP_DIR, 'data/bff21_zero')
elif settings == 1 or settings == 2:
    DATA_DIR = os.path.join(TOP_DIR, 'data/bff21_zero/w_max_77')

CLASS_INPUT_MAP = {'CHSH': '00', '1': '01', '2a': '11',
                   '2b': '01', '2b_swap': '11', '2c': '10',
                   '3a': '11', '3b': '10'}

######### Plotting settings #########
# FIG_SIZE = (16, 9)      # for aspect ratio 16:9
FIG_SIZE = (12, 9)    # for aspect ratio 4:3
DPI = 200
if settings == 1:
    #### Single plot for max win prob 
    SUBPLOT_PARAM = {'left': 0.085, 'right': 0.98, 'bottom': 0.11, 'top': 0.95, 'wspace': 0.28}
elif settings == 2:
    #### Two subplots for max win and 0.77
    SUBPLOT_PARAM = {'left': 0.085, 'right': 0.98, 'bottom': 0.1, 'top': 0.905, 'wspace': 0.28}
else:
    #### Plot for win 0.7525 with break axis
    SUBPLOT_PARAM = {'left': 0.125, 'right': 0.98, 'bottom': 0.12, 'top': 0.95, 'wspace': 0.28}


### Default font size
if settings == 1 or settings == 3:
    titlesize = 30
    ticksize = 28
    legendsize = 28
elif settings == 2:
    titlesize = 36
    ticksize = 34
    legendsize = 32

### Default line width
if settings == 1 or settings == 2:
    linewidth = 3
elif settings == 3:
    linewidth = 3.5

### Set matplotlib plotting params
mplParams = {
    #### Default font (family and size)
    "font.family": "serif",
    "font.sans-serif": "Latin Modern Roman",
    "font.size": legendsize,
    "figure.titlesize": titlesize,
    "axes.labelsize": titlesize,
    "xtick.labelsize": ticksize,
    "ytick.labelsize": ticksize,
    "legend.fontsize": legendsize,
    #### Line width
    "lines.linewidth": linewidth,
    #### Xtick-axis padding
    "xtick.major.pad": 10,
    #### Use Latex
    "text.usetex": True,
    #### Make text and math font bold
    # "font.weight": "bold",
    # "text.latex.preamble": r"\boldmath"
    #### Line colors
    # "axes.prop_cycle": cycler(color=\
    #                    ['b','green','red','c','m', 'olive', 'gray', 'purple'])
}

plt.rcParams.update(mplParams)

### Parallel settings
N_JOB = 8

### Constant parameters
CHSH_W_Q = (2 + math.sqrt(2) )/4    # CHSH win prob
EPSILON = 1e-12                     # Smoothness of smooth min entropy (related to secrecy)
WIN_TOL = 1e-4                      # Tolerant error for win prob
GAMMA = 1e-2                        # Testing ratio

### General File Name Settings
OUT_DIR = os.path.join(TOP_DIR, 'figures/corrected_FER')
if settings == 2:
    COMMON = 'FER-w_max_77'
elif settings == 1 or settings == 3:
    COMMON = 'FER'
EPS = f'eps_{EPSILON:.0e}'
WTOL = f'wtol_{WIN_TOL:.0e}'
GAM = f'gam_{GAMMA:.0e}'
TAIL = 'slide'
FORMAT = 'png'

### Num of rounds
if settings == 3:
    N_RANGE = (1e8, 1e14)
elif settings == 1 or settings == 2:
    N_RANGE = (1e7, 1e14)
Ns = np.geomspace(*N_RANGE, num=200)

### Freely tunable params
#### Param related to alpha-Renyi entropy
BETA_SLICE = 150
BETAs = np.geomspace(1e-4, 1e-11, num=BETA_SLICE)
#### Another tunable param
NU_PRIME_SLICE = 50
NU_PRIMEs = np.linspace(1, 0.1, num=NU_PRIME_SLICE)

### Finite extractable rate for blind randomness extraction with testing
def key_rate_testing(n, beta, nu_prime, gamma, asym_rate, lambda_, c_lambda,
                        epsilon = EPSILON, win_tol = WIN_TOL, zero_tol = 0,
                        zero_class = 'CHSH', max_p_win = CHSH_W_Q):
    
    ln2 = math.log(2)
    gamma_0 = (1-gamma)/gamma
    nu_0 = 1/(2*gamma) - gamma_0*nu_prime
    
    D = c_lambda**2 + (lambda_/(2*gamma))**2 - gamma_0/gamma * lambda_**2 * (1-nu_prime)*nu_prime
    if nu_0 > max_p_win:
        var_f = D - (lambda_*(max_p_win - nu_prime))**2
    elif nu_0 < (1 - max_p_win):
        var_f = D - (lambda_*(1 - max_p_win - nu_prime))**2
    else:
        var_f = D

    max2min_f = lambda_*(gamma_0 * (1-nu_prime) + max_p_win)
    if math.sqrt(1 - epsilon**2) == 1:
        epsi_term = math.log2(1/2*epsilon**2)
    else:
        epsi_term = math.log2(1 - math.sqrt(1 - epsilon**2) )
    
    epsi_win = math.e ** (-2 * win_tol**2 * n)
    zero_tol = zero_tol/2
    epsi_zero = math.e ** (-2 * zero_tol**2 * n)
    try:
        log_prob = math.log2(1 - epsi_win)
        if zero_class != 'CHSH':
            n_zero = int(zero_class[0])
            log_prob += n_zero * math.log2(1 - epsi_zero)
    except ValueError:
        log_prob = math.log2(sys.float_info.min)

    with np.errstate(over='raise'):
        try:
            lamb_term = 2 ** (2 + max2min_f)
            K_beta = 1/(6*ln2) * (beta**2)/((1-beta)**3) \
                      * (lamb_term ** beta) \
                      * (math.log(lamb_term + math.e**2)) **3

        except FloatingPointError:
            K_beta = 1/(6*ln2) * (beta**2)/((1-beta)**3) \
                      * (2 ** (beta * (2 + max2min_f) ) ) \
                      * ( (2 + max2min_f)/math.log2(math.e) ) **3

    key_rate =  asym_rate \
                - ln2/2 * beta * (math.log2(9) + math.sqrt(2 + var_f)) ** 2 \
                + 1/(beta * n) * ( (1+beta)*epsi_term + (1+2*beta)*log_prob) \
                - K_beta    
    return key_rate if key_rate > 0 else 0

### Helper function to save plot
def savePlot(fname, format, out_dir):
    out_path = os.path.join(out_dir, f'{fname}.{format}')
    plt.savefig(out_path, format = format)

#### Plot for max win prob and win prob at 0.7525
if settings == 1 or settings == 3:
    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)
    if settings == 3:
        ### Add subplot to produce breaking axis for win 0.7525
        gs = fig.add_gridspec(2, hspace=0)
        axs = gs.subplots(sharex=True)

if settings == 1 or settings == 3:
    #### Single plot for max win prob and win prob at 0.7525
    CLASSES = ['CHSH','1','2c','3b']
elif settings == 2:
    #### Two subplots for max win and 0.77
    CLASSES = ['1','2a','2b','2b_swap','2c','3a','3b']

if settings == 1 or settings == 3:
    #### Colors for different classes
    # COLORS = ('blue','firebrick','forestgreen','gray')
    COLORS = ('blue','forestgreen','firebrick','gray')
    # COLORS = ('blue','green','red','cyan','magenta', 'olive', 'gray', 'purple')
elif settings == 2:
    #### Colors for different winning probabilities
    COLORS = ('darkcyan','darkred')

color_iter = itertools.cycle(COLORS)

################################ Iter for different parameters ################################
### Run over all classes in CLASSES
for class_ in CLASSES:
    input_ = CLASS_INPUT_MAP[class_]

    #### Two subplots for max win and 0.77
    if settings == 2:
        fig, axs = plt.subplots(1, 2, figsize=FIG_SIZE, dpi=DPI)

    #### Line styles for different zero tolerances
    LINES = itertools.cycle(('solid', 'dashed', 'dashdot', 'dotted'))
    #### Colors for different classes
    if settings != 2:
        color = next(color_iter)

    ### Specify the file record the data
    HEAD = 'lr_bff21'
    CLASS = f'class_{class_}' if class_ != 'CHSH' else 'CHSH'
    INPUT = f'xy_{input_}'
    QUAD = 'M_12'
    # WTOL = f'wtol_{WIN_TOL:.0e}'

    ZERO_TOLs = []
    ### Tolerant error for winning prob
    if settings == 2:
        #### Two subplots for max win and 0.77 
        ZERO_TOLs.extend([1e-9, 1e-5, 1e-3])
    elif settings == 1 or settings == 3:
        #### Single plot for max win prob
        ZERO_TOLs.append(1e-9)
        if settings == 3 and class_ == '3b':
            ZERO_TOLs.append(1e-3)

    ### Run over all zero tolerances in ZERO_TOLs
    for zero_tol in ZERO_TOLs:
        ZTOL = f'ztol_{zero_tol:.0e}'
        DATA_FILE = f'{HEAD}-{CLASS}-{INPUT}-{QUAD}-{WTOL}-{ZTOL}.csv'
        DATA_PATH = os.path.join(DATA_DIR, DATA_FILE)

        ### Get the maximum winnin probability
        with open(DATA_PATH) as file:
            max_p_win = float(file.readlines()[1])

        ### Load data
        data_mtf = np.genfromtxt(DATA_PATH, delimiter=",", skip_header = 3)
        # print(data_mtf)
        
        line = next(LINES)
        ### Colors for different winning probabilities
        if settings == 2:
            color_iter = itertools.cycle(COLORS)

        #### Two subplots for max win and 0.77
        titles = []

        ### Choose min-tradeoff function with expected winning prob
        if len(data_mtf.shape) == 1:
            data_mtf = np.array([data_mtf])
        data_len = data_mtf.shape[0] if settings == 2 else 1

        #### For different w_exp (or other parameters) in the data
        for i in range(data_len):
            win_prob, asym_rate, lambda_ = data_mtf[i][:3]
            c_lambda = data_mtf[i][-1]
            if class_ != 'CHSH':
                lambda_zeros = data_mtf[i][3:-1]
                c_lambda -= sum(lambda_zeros)*zero_tol

################################ Compute key rate ################################
            ### Construct key rate function with fixed param (only leave n, beta tunable)
            kr_func = partial(key_rate_testing, gamma = GAMMA, asym_rate = asym_rate,
                                lambda_ = lambda_, c_lambda = c_lambda,
                                zero_tol = zero_tol, zero_class=class_, max_p_win = max_p_win)
            
            def opt_all(n, beta_arr = BETAs, nu_prime_arr = NU_PRIMEs):
                    return np.max(np.array([kr_func(n = n, beta = beta, nu_prime = nu_prime) \
                                            for nu_prime in nu_prime_arr for beta in beta_arr]))

            KRs = Parallel(n_jobs=N_JOB, verbose = 0)(delayed(opt_all)(N) for N in Ns)
            KRs = np.array(KRs)
            
            ## Draw line with optimal parameters

            ### Colors for different zero tolerances
            if settings == 2:
                color = next(color_iter)
            
            ### Set labels of legends
            if settings == 2:
                #### Two subplots for max win and 0.77
                label = r'$\displaystyle \delta_{zero}$'+f'={zero_tol:.0e}'
            elif settings == 1 or settings == 3:
                label = f'class {class_}' if class_ != 'CHSH' else 'CHSH'
                if settings == 1:
                    #### Single plot for max win prob
                    label += r' $\displaystyle x^{*}y^{*}$'+f'={input_}' + \
                                r' $\displaystyle w_{exp}$'+f'={win_prob:.4f}'
                elif settings == 3:
                    #### Plot for win 0.7525 with break axis
                    label += r' $\displaystyle x^{*}y^{*}$'+f'={input_}' + \
                                r' $\displaystyle \delta_{zero}$'+f'={zero_tol:.0e}'

            if PRINT_DATA:
                print(np.column_stack((Ns, KRs)))

            if SAVE_CSV:
                data2save = np.column_stack((Ns, KRs))
                HEADER = 'num of rounds in log\trate'
                WIN_EXP = f'w_{win_prob*10000:.0f}'.rstrip('0')
                ZTOL = f'ztol_{zero_tol:.0e}'
                OUTFILE = f'{CLASS}-{WIN_EXP}-{EPS}-{WTOL}-{ZTOL}-{GAM}-{QUAD}.csv'
                np.savetxt(OUTFILE, data2save, fmt='%.5g', delimiter=',', header=HEADER)

################################ Plotting lines ################################
            if settings == 1:
                #### Single plot for max win prob
                plt.plot(Ns, KRs,
                    linestyle = line, color = color, label = label)
            elif settings == 2:
                #### Two subplots for max win and 0.77
                axs[i].plot(Ns, KRs,
                        linestyle = line, color = color, label = label)
                titles.append(r'$\displaystyle w_{exp}$'+f'={win_prob:.4g}')            
            elif settings == 3:
                #### Plot for win 0.7525 with break axis
                axs[0].plot(Ns, KRs,
                        linestyle = line, color = color, label = label)            
                axs[1].plot(Ns, KRs,
                        linestyle = line, color = color, label = label)


################################ Save figure ################################
    YLABEL = r'$\displaystyle r$'+' (bit per round)'
    XLABEL = r'$\displaystyle n$'+' (number of rounds)'
    if settings == 2:
        ### For two subplots
        axs[0].label_outer()
        axs[0].set_ylabel(ylabel=YLABEL)
        axs[0].legend(loc='best')
        axs[1].legend(loc='center right', bbox_to_anchor=(1, 0.55))

        for i in range(len(axs)):
            axs[i].set_xlabel(xlabel=XLABEL)
            axs[i].set_xscale("log")
            axs[i].set_title(titles[i], y=1, pad=20)
            axs[i].yaxis.set_major_formatter(MyScalarFormatter(useMathText=True))
            axs[i].yaxis.major.formatter.set_powerlimits((-1,2))
            axs[i].xaxis.set_major_locator(LogLocator(numticks=8))
        
        ### Apply the graphic settings
        plt.subplots_adjust(**SUBPLOT_PARAM)

        fig.tight_layout(pad=1)

        if SAVE:
            OUT_NAME = f'{COMMON}-{CLASS}-{INPUT}-{EPS}-{WTOL}-{GAM}-{QUAD}-{TAIL}'
            savePlot(OUT_NAME, FORMAT, OUT_DIR)
        if SHOW:
            plt.show()

if settings == 1 or settings == 3:
    if settings == 1:
        #### Single plot for max win prob
        plt.xscale("log")
        plt.ylabel(YLABEL)
        plt.xlabel(XLABEL)
        plt.legend(prop={"weight":"bold"})

    elif settings == 3:
        ### For break axis
        axs[1].set_xlabel(xlabel=XLABEL)
        axs[0].legend(loc='best')
        axs[0].set_ylim(top=0.03, bottom=0.00085)
        axs[1].set_ylim(top=0.0008, bottom=0)

        for ax in axs:
            ax.set_xscale("log")

        WIN_EXP = r'$\displaystyle {w}_{exp}=0.7525$'
        fig.text(0.16, 0.85, WIN_EXP, va='center')
        fig.supylabel(YLABEL)

    ### Apply the graphic settings
    plt.subplots_adjust(**SUBPLOT_PARAM)

    ### Save file
    if SAVE:
        if settings == 1:
            WIN_EXP = 'QBOUND'
        elif settings == 3:
            WIN_EXP = 'w_7525'
            # WIN_EXP = f'w_{win_prob*10000:.0f}'.rstrip('0')
        OUT_NAME = f'{COMMON}-{WIN_EXP}-{EPS}-{WTOL}-{GAM}-{QUAD}-{TAIL}'
        savePlot(OUT_NAME, FORMAT, OUT_DIR)
    if SHOW:
        plt.show()
