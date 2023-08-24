from matplotlib import rcParams, cycler
from matplotlib import pyplot as plt
import numpy as np
import os

### Data paths
TOP_DIR = './'
DATA_DIR = os.path.join(TOP_DIR, 'data/BFF21/mtf')
CLASS_INPUT_MAP = {'CHSH': '00', '1': '01', '2a': '11',
                   '2b': '01', '2b_swap': '11', '2c': '10',
                   '3a': '11', '3b': '10'}

### Figure settings
rcParams['axes.prop_cycle'] = cycler(
                                color=['b','green','r','c','m','orange','limegreen','saddlebrown']) 
FIG_SIZE = (24, 12)
DPI = 100
FIG_MARGINAL = {'left': 0.08, 'right': 0.95, 'bottom': 0.08, 'top': 0.95}

rcParams["figure.figsize"] = [24, 16]
rcParams["figure.autolayout"] = True

CLASSES = ['CHSH', '2a']
YLIMS = [(0, 0.6), (0, 0.85)]
XLIMS = [(0.81, 0.855), (0.75, 0.815)]
NUM_CLASS = len(CLASSES)
#fig, axes = plt.subplots(NUM_CLASS)

for j in range(NUM_CLASS):
    CLASS = CLASSES[j]
    INPUT = f'xy_{CLASS_INPUT_MAP[CLASS]}'
    if CLASS != 'CHSH':
        CLASS = f'class_{CLASS}'
    MTF_FILE = f'min_tradeoff-{CLASS}-{INPUT}-M_12-dev_1e-05-BFF21.csv'
    MTF_PATH = os.path.join(DATA_DIR, MTF_FILE)


    ### Load data
    data_mtf = np.genfromtxt(MTF_PATH, delimiter=",", skip_header = 1)

    data = data_mtf.T[0:2]
    #print(data)

    plt.figure(figsize=FIG_SIZE, dpi=DPI)
    plt.subplots_adjust(**FIG_MARGINAL)

    plt.plot(*data, linestyle = '-', color='black', marker = 'o', label='H(A|BXYE)')
    #axes[j].plot(*data, linestyle = '-', color='black', marker = 'o', label='H(A|BXYE)')

    NUM_LINES = 18 #data_mtf.shape[0]
    for i in range(0, NUM_LINES, 3):
        win_prob, _, lambda_  = data_mtf[i][:3]
        c_lambda = data_mtf[i][-1]
        tradeoff_values = data[0] * lambda_ + c_lambda
        # tradeoff_values[tradeoff_values < 0] = 0
        plt.plot(data[0], tradeoff_values, label = f'win_prob={win_prob:.4g}')
        #axes[j].plot(data[0], tradeoff_values, label = f'win_prob={win_prob:.4g}')

    ax = plt.gca()
    ax.set_xlim(XLIMS[j])
    ax.set_ylim(YLIMS[j])
    #axes[j].set_xlim(XLIMS[j])
    #axes[j].set_ylim(YLIMS[j])
    
    plt.legend(loc='best', fontsize=26)
    #axes[j].legend(loc='best', fontsize=26)

    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    #axes[j].tick_params(axis='both', which='major', labelsize=24)
    #axes[j].xaxis.set_tick_params(pad=10)
    
    X_TITLE = 'CHSH winning probability'
    plt.xlabel(X_TITLE, fontsize=28)
    #axes[j].set_xlabel(X_TITLE, fontsize=28)
    plt.ylabel('randomness (bit)', fontsize=28)
    #axes[j].set_ylabel('randomness (bit)', fontsize=28)
    #axes[j].label_outer()
    
    TITLE = CLASS if CLASS == 'CHSH' else f"class {CLASS.lstrip('class_')}"
    #axes[j].set_title(TITLE, fontsize=32, pad=12)
   
    #axes[0].xaxis.label.set_visible(False)

    OUT_DIR = os.path.join(TOP_DIR, 'figures/min_tradeoff_func')
    OUT_NAME = f'mtf-{CLASS}-M12-BFF21-trim-1'
    FORMAT = 'png'
    OUT_FILE = f'{OUT_NAME}.{FORMAT}'
    OUT_PATH = os.path.join(OUT_DIR, OUT_FILE)

    if os.path.exists(OUT_PATH):
        ans = input(f"File '{OUT_FILE}' exists, do u want to replace it? [Y/y]")
        if ans == 'y' or ans == 'Y':
            plt.savefig(OUT_PATH, format = FORMAT)
        else:
            print(f'Current file name is "{OUT_FILE}"')
            print('Change the file name to save.')
    else:
        plt.savefig(OUT_PATH, format = FORMAT)
    '''

    plt.show()
    '''
