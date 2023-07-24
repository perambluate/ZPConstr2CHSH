"""
    This script is used to compute the von Neumann entropy H(A|XYE) 
    for local randomness in the point of view of other player in CHSH non-local game
"""
import ncpol2sdpa as ncp
import mosek
import chaospy
from sympy.physics.quantum.dagger import Dagger
import math
import numpy as np
from joblib import Parallel, delayed
import time
import sys, os

LEVEL = 2                       # NPA relaxation level
M = 6                           # Num of terms in Gauss-Radau quadrature = 2*M
VERBOSE = 1                     # Relate to how detail of the info will be printed
N_WORKER_QUAD = 4               # Number of workers for parallelly computing quadrature
N_WORKER_LOOP = 1               # Number of workers for the outer loop
N_WORKER_SDP = 4                # Number of threads for solving a single SDP
PRIMAL_DUAL_GAP = 1e-5          # Allowable gap between primal and dual
SOLVER_CONFIG = ['mosek', {'dparam.presolve_tol_x': 1e-10,
                           'dparam.intpnt_co_tol_rel_gap': PRIMAL_DUAL_GAP,
                           'iparam.num_threads': N_WORKER_SDP,
                           'iparam.infeas_report_level': 4}]
# SOLVER_CONFIG = ['sdpa']
ACCURATE_DIGIT = 4              # Achievable precision of the solver
WIN_TOL = 1e-4                  # Relax the precise winning prob constraint to a range with epsilon
ZERO_PROB = 1e-9                # Treat this value as zero for zero probability constraints
SAVE_DATA = True                # Set the data into file
TIMMING = True                  # True for timming

np.set_printoptions(precision=5)

KS_ZEROS = {'3a': [([0,0],[0,0]),([1,1],[0,1]),([1,1],[1,0])],
            '3b': [([0,0],[0,0]),([1,1],[0,0]),([1,0],[1,1])],
            '3b_swap': [([0,0],[0,0]),([1,1],[0,0]),([0,1],[1,1])],
            '2a': [([0,0],[0,0]),([1,1],[0,0])],
            '2b': [([0,0],[0,0]),([1,1],[1,0])],
            '2b_swap': [([0,0],[0,0]),([1,1],[0,1])],
            '2c': [([0,0],[0,0]),([1,0],[1,1])],
            '2c_swap': [([0,0],[0,0]),([0,1],[1,1])],
            '1' : [([0,0],[0,0])]}

C_BOUND = 0.75
CLASS_MAX_WIN = {'chsh': 0.8535, '1': 0.8294, '2a': 0.8125,
                 '2b': 0.8125, '2b_swap': 0.8125, '2c': 0.8039,
                 '3a': 0.7951, '3b': 0.7837}

ZERO_CLASS = ['chsh', '1', '2a', '2b', '2b_swap', '2c', '3a', '3b']
ZERO_TOLs = [1e-9] #, 1e-5, 1e-3]                 # Tolerate error for zero classes

def truncate(number, digit):
    """
        Truncate the floating number up to the given digit after point
        It will ignore any number after the digit
    """
    truncated_number = round(number, digit)
    if round(number, digit+1) < truncated_number:
        truncated_number -= 10 ** (-digit)
    return truncated_number

# Score (winning probability) constraint of non-local game
def winProb(P, configs = [[2,2],[2,2]], inp_probs = []):
    """
        Function of winning probability of nonlocal game
        Here we use a modified CHSH non-local game for example
        - P : P([a,b],[x,y])=P(ab|xy) is the probability that get the outputs (a,b) for given inputs (x,y)
        - configs : Tell the number of inputs and number of outputs for each input in each party
                    may be useful for general scenario
        - inp_probs : The probability of the inpus P(xy)
    """
    try:
        config_A = configs[0]
        num_x = len(config_A)
        config_B = configs[1]
        num_y = len(config_B)
    except IndexError:
        print(f'Wrong input configs: {configs}')
    win_prob = 0
    for x in range(num_x):
        for y in range(num_y):
            for a in range(config_A[x]):
                for b in range(config_B[y]):
                    # One should modify the following line for different winning condtion
                    if a^b == (x*y)^1:
                        if np.any(inp_probs):
                            try:
                                prob = inp_probs[x][y]
                            except IndexError:
                                print(f'Wrong input inp_probs: {inp_probs}')
                        else:
                            prob = 1/4
                        win_prob += P([a,b], [x,y])*prob
    return win_prob

def zeroConstr(P, positions, err_tol = 0):
    """
        Set the zero probability constraints for the positions with tolerable error 
    """
    constr = []
    for pos in positions:
        constr += [err_tol-P(*pos)]

    return constr

def probConstr(P, configs=[[2,2],[2,2]]):
    """
        Set the constraints for physical probabilities
    """
    try:
        config_A = configs[0]
        num_x = len(config_A)
        config_B = configs[1]
        num_y = len(config_B)
    except IndexError:
        print(f'Wrong input configs: {configs}')
    return [P([a,b],[x,y]) for x in range(num_x) for y in range(num_y) \
            for a in range(config_A[x]) for b in range(config_B[y])]

"""
# Constraints for Eve's operators for i-th term of the quadrature
# This can be check manually after optimization
def alphaConstr(Zs, t_i):
    alpha = max(1/t_i, 1/(1-t_i)) * 3/2
    constr = []
    for z in Zs:
        constr += [alpha - Dagger(z)*z]

    return constr
"""

# Objective function to compute the infimum in the inner summation
def objective(A, B, inputs, Z_ab, t_i):
    """
       The objective function in the inner summation in BFF21 method
       - A, B : Alice and Bob's measurements in CG form
       - inputs : Alice and Bob's inputs for the computation
       - Z_ab : Eve's operators
       - t_i : Nodes in Gauss-Radau quadrature
    """
    x_star, y_star = inputs
    meas_A = [A[x_star][0], 1-A[x_star][0]]
    meas_B = [B[y_star][0], 1-B[y_star][0]]
    obj = 0
    for a in range(2):
        for b in range(2):
            ab = a*2+b
            obj += meas_A[a]*meas_B[b]*(Z_ab[ab] + Dagger(Z_ab[ab]) \
                    + (1-t_i)*Dagger(Z_ab[ab])*Z_ab[ab])
            obj += t_i *(Z_ab[ab]*Dagger(Z_ab[ab]))
    return obj

def printProb(sdp, P, configs=[[2,2],[2,2]]):
    try:
        config_A = configs[0]
        num_x = len(config_A)
        config_B = configs[1]
        num_y = len(config_B)
    except IndexError:
        print(f'Wrong input configs: {configs}')
    for y in range(num_y):
        for b in range(config_B[y]):
            s = f'\t'.join([f'{sdp[P([a,b],[x,y])]:.5g}' \
                            for x in range(num_x) for a in range(config_A[x])])
            print(s)

def printNorm(sdp, ops):
    for op in ops:
        norm = sdp[Dagger(op)*op]
        print(f'{op}\tNorm: {norm:.5g}')

def zeroPos2str(pos_list):
    pos_list = np.array(pos_list)
    pos_str = []
    for pos in pos_list:
        flat_pos = pos.reshape(np.sum(pos.shape))
        pos_str.append( ''.join([str(i) for i in flat_pos]) )
    return pos_str


if VERBOSE:
    if SOLVER_CONFIG[0] == 'mosek':
        print(f'MOSEK primal-dual tol gap: {PRIMAL_DUAL_GAP}')
    # print(f'Zero probability tol err: {ZERO_ERR}')
    print(f'WinProb deviation: {WIN_TOL}')

# Setup of the scenario for Alice and Bob
A_config = [2,2]
B_config = [2,2]
P = ncp.Probability(A_config, B_config)
A, B = P.parties
substs = P.substitutions

# Generate Eve's operators
Z_ab = ncp.generate_operators('Z_ab', 4, hermitian=False)

# Make Eve's operators commute with Alice and Bob's
for a in P.get_all_operators():
    for z in Z_ab:
        substs.update({z*a: a*z, Dagger(z)*a: a*Dagger(z)})

# Generate ABZ, ABZ*, ABZZ*, ABZ*Z as extra monomials
extra_monos = []
for a in ncp.flatten(A):
    for b in ncp.flatten(B):
        for z in Z_ab:
            extra_monos += [a*b*z, a*b*Dagger(z), a*b*z*Dagger(z), a*b*Dagger(z)*z]

# SDP in quadrature summation
def innerQuad(level, P, Z_ab, inputs, quad_t, quad_w, p_win_constr, zero_constr, zero_class = '',
              substs = {}, extra_monos = []):
    A, B = P.parties
    obj = objective(A, B, inputs, Z_ab, quad_t)
    ops = P.get_all_operators()+Z_ab
    sdp = ncp.SdpRelaxation(ops, verbose=VERBOSE-3)

    # Setup the sdp relaxation with objective as some trivial const as placehold
    sdp.get_relaxation(level = level, objective = obj,
                        substitutions = substs,
                        momentinequalities = zero_constr + p_win_constr,
                        extramonomials = extra_monos)

    sdp.solve(*SOLVER_CONFIG)

    if VERBOSE > 2:
        print(sdp.status, sdp.primal, sdp.dual)

    if sdp.status != 'optimal' and sdp.status != 'primal-dual feasible':
        return
    
    coeff = quad_w/(quad_t * math.log(2))
    entropy_in_quad = coeff * (1 + sdp.dual)
    p_win_in_quad = sdp[winProb(P)]
    
    if VERBOSE >= 2:
        print(f'entropy in quad: {entropy_in_quad}')
        print(f'winning probability in quad: {p_win_in_quad}')
        if VERBOSE >= 3:
            printProb(sdp,P)
            printNorm(sdp,Z_ab)
    
    p_win_dual_vec = np.array([sdp.get_dual(constr) for constr in p_win_constr])
    p_win_dual_vec = np.squeeze(p_win_dual_vec)
    
    if not p_win_dual_vec.shape:
        p_win_dual_var = p_win_dual_vec*1
    else:
        p_win_dual_var = p_win_dual_vec[0]-p_win_dual_vec[1]
    
    zero_pos = KS_ZEROS.get(zero_class, [])
    if zero_pos:
        num_zero_constr = len(zero_pos)
        lambda_in_quad =  np.zeros(num_zero_constr+1)
        min_p_zero_in_quad = np.zeros(num_zero_constr)
        lambda_in_quad[0] = coeff * p_win_dual_var
        for i in range(num_zero_constr):
            zero_dual_var = np.squeeze(sdp.get_dual(zero_constr[i]))
            lambda_in_quad[i+1] = coeff * zero_dual_var
            #print(zero_dual_var)
            p_zero = sdp[P(*zero_pos[i])]
            min_p_zero_in_quad[i] = p_zero
            #print(p_zero)

        return p_win_in_quad, entropy_in_quad, lambda_in_quad, min_p_zero_in_quad      
    else:
        lambda_in_quad = coeff * p_win_dual_var
        return p_win_in_quad, entropy_in_quad, lambda_in_quad, np.empty(0)

# Solve SDP of single round entropy
def singleRoundEntropy(P, Z_ab, win_prob, M, inputs, zero_class = '', zero_tol = 0,
                        substs = {}, extra_monos = [], win_tol = WIN_TOL, level = LEVEL):    

    p_win_constr = [winProb(P)-win_prob+win_tol, -winProb(P)+win_prob+win_tol]
    zero_pos = KS_ZEROS.get(zero_class, [])
    zero_constr = zeroConstr(P, zero_pos, zero_tol)
    
    # Nodes, weights of quadrature up to 2*M terms
    T, W = chaospy.quad_gauss_radau(M, chaospy.Uniform(0, 1), 1)    
    T = T[0]
    NUM_NODE = len(T)
    if VERBOSE > 1:
        print(f'Number of terms summed in quadrature: {NUM_NODE}')
        print(f'Nodes of the Gauss-Radau quadrature:\n{T}')
        print(f'Weights of the Gauss-Radau quadrature:\n{W}')

    entropy = -1/(len(T)**2 * math.log(2)) + W[-1]/(T[-1]*math.log(2))

    # Optimize for each node of the Gauss-Radau quadrature
    # The last node located at 1 is a constant, we donot need to optimize
    results = Parallel(n_jobs=N_WORKER_QUAD, verbose=0)(
                                delayed(innerQuad)(level, P, Z_ab, inputs, T[i], W[i],
                                                    p_win_constr, zero_constr, zero_class,
                                                    substs, extra_monos) for i in range(NUM_NODE-1))
    # print(results)

    p_win_quad, entropy_quad, lambda_quad, p_zero_quad = zip(*results)
    # print(p_win_quad)
    # print(entropy_quad)
    # print(lambda_quad)
    # print(p_zero_quad)

    max_p_win = max(p_win_quad)
    entropy += np.sum(np.array(entropy_quad))
    lambda_ = np.sum(np.array(lambda_quad), axis=0)
    p_zero_quad = np.array(p_zero_quad)

    if np.any(p_zero_quad):
        min_p_zero = np.min(p_zero_quad, axis=0)
        c_lambda = entropy - lambda_[0] * max_p_win + np.sum(lambda_[1:] * min_p_zero)
    else:
        c_lambda = entropy - lambda_ * max_p_win
    return win_prob, entropy, lambda_, c_lambda


if TIMMING:
    tic = time.time()

for zero_class in ZERO_CLASS:

    ZERO_POS = KS_ZEROS.get(zero_class, [])

    if VERBOSE:
        print(f'Correlation type: {zero_class}')
        if ZERO_POS:
            print(f'Zero positions: {ZERO_POS}')
        else:
            print(f'No zero constraints')

    # zero_constraints = zeroConstr(P, ZERO_POS, ZERO_PROB)

    # # Compute the quantum bound first
    # sdp_Q = ncp.SdpRelaxation(P.get_all_operators(), verbose=VERBOSE-3)
    # sdp_Q.get_relaxation(level=LEVEL, objective=-winProb(P),
    #                     substitutions = P.substitutions,
    #                     momentinequalities = zero_constraints)
    # sdp_Q.solve(*SOLVER_CONFIG)

    # if VERBOSE:
    #     print(sdp_Q.status, sdp_Q.primal, sdp_Q.dual)

    # if sdp_Q.status != 'optimal' and sdp_Q.status != 'primal-dual feasible':
    #     print('Cannot compute quantum bound correctly!', file=sys.stderr)
    #     break

    # P_WIN_Q = truncate(-sdp_Q.primal, ACCURATE_DIGIT)
    # P_WINs = [CLASS_MAX_WIN[zero_class]]
    Q_BOUND = CLASS_MAX_WIN[zero_class]
    NUM_SLICE = 21
    P_WINs = np.linspace(Q_BOUND, C_BOUND, NUM_SLICE)

    zero_tol_arr = ZERO_TOLs if zero_class != 'chsh' else [ZERO_PROB]

    for zero_tol in zero_tol_arr:
        if VERBOSE:
            print(f'Zero probability tolerance: {zero_tol}')
        zero_constraints = zeroConstr(P, ZERO_POS, zero_tol)

        if VERBOSE:
            print('Start compute winning probability quantum bound>>>')

        # Compute the quantum bound first
        sdp_Q = ncp.SdpRelaxation(P.get_all_operators(), verbose=VERBOSE-3)
        sdp_Q.get_relaxation(level=LEVEL, objective=-winProb(P),
                            substitutions = P.substitutions,
                            momentinequalities = zero_constraints)
        sdp_Q.solve(*SOLVER_CONFIG)

        if VERBOSE:
            print('Status\tPrimal\tDual')
            print(f'{sdp_Q.status}\t{sdp_Q.primal}\t{sdp_Q.dual}')
            print('End of computing quantum bound.<<<')

        if sdp_Q.status != 'optimal' and sdp_Q.status != 'primal-dual feasible':
            print('Cannot compute quantum bound correctly!', file=sys.stderr)
            break
        
        max_p_win = truncate(-sdp_Q.primal, ACCURATE_DIGIT)
                  
        # The selected inputs to compute entropy
        # INPUTS = [OPT_INPUT_MAP.get(zero_class, (0,0))]
        INPUTS = [(0,0), (0,1), (1,0), (1,1)]

        for inputs in INPUTS:
            if VERBOSE:
                print(f'Chosen input xy={inputs[0]}{inputs[1]}')       

            results = Parallel(n_jobs=N_WORKER_LOOP, verbose = 0)(
                                delayed(singleRoundEntropy)(P, Z_ab, win_prob, M, inputs,
                                                            zero_class, zero_tol,
                                                            substs, extra_monos) \
                                                            for win_prob in P_WINs)
                
            #print(results)

            if VERBOSE or SAVE_DATA:
                if zero_class == 'chsh':
                    metadata = ['winning_prob', 'entropy', 'lambda', 'c_lambda']
                else:
                    zero_pos_str = zeroPos2str(ZERO_POS)
                    zero_pos_str = [f'lambda_{pos}' for pos in zero_pos_str]
                    metadata = ['winning_prob', 'entropy', 'lambda', *zero_pos_str, 'c_lambda']
            
            if VERBOSE:
                headline = '\t'.join(metadata)
                print(headline)
                
                for win_prob, entropy, lambda_, c_lambda in results:                
                    if zero_class == 'chsh':
                        lambda_str = f'{lambda_:.5g}'         
                    else:
                        lambda_vals = [f'{val:5g}' for val in lambda_]
                        lambda_str = '\t'.join(lambda_vals)
                    line = f'{win_prob:.5g}\t{entropy:.5g}\t'+lambda_str+f'\t{c_lambda:.5g}'
                    print(line)
                
                print("")

            # Write to file
            if SAVE_DATA:
                if zero_class == 'chsh':
                    data = np.array(results)
                else:
                    data = [[win_prob, entropy, *lambda_, c_lambda] \
                            for win_prob, entropy, lambda_, c_lambda in results]
                    data = np.array(data)

                HEADER = ', '.join(metadata)
                MAX_P_WIN = f'MAX_WIN_PROB\n{max_p_win:.5g}'

                COM = 'tpr'
                CLS = zero_class if zero_class == 'chsh' else f'cls_{zero_class}'
                INP = f'xy_{inputs[0]}{inputs[1]}'
                QUAD = f'M_{M*2}'
                WTOL = f'wtol_{WIN_TOL:.0e}'
                ZTOL = f'ztol_{zero_tol:.0e}'
                OUT_FILE = f'{COM}-{CLS}-{INP}-{QUAD}-{WTOL}-{ZTOL}.csv'
                OUT_DIR = './'
                OUT_PATH = os.path.join(OUT_DIR, OUT_FILE)
                
                if os.path.exists(OUT_PATH):
                    with open(OUT_PATH, 'ab') as file:
                        file.write(b'\n')
                        file.write(bytes(MAX_P_WIN, 'utf-8') + b'\n')
                        np.savetxt(file, data, fmt='%.5g', delimiter=',', header=HEADER)
                else:
                    with open(OUT_PATH, 'wb') as file:
                        file.write(bytes(MAX_P_WIN, 'utf-8') + b'\n')
                        np.savetxt(file, data, fmt='%.5g', delimiter=',', header=HEADER)
                
if TIMMING:
    toc = time.time()
    print(f'Elapsed time: {toc-tic}')
