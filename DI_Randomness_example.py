"""
An example script to show how to compute the device-indepnedent (DI) randomness 
    with Brown-Fawzi-Fawzi method (https://arxiv.org/abs/2106.13692)
    and the ncpol2sdpa package (https://github.com/peterjbrown519/ncpol2sdpa).
The type of DI randomness we consider here is global (two-party) randomness,
    extrated from both parties' outcomes in the CHSH game.
"""
import ncpol2sdpa as ncp
import numpy as np
from sympy.physics.quantum.dagger import Dagger
import chaospy
import math
from joblib import Parallel, delayed

LEVEL = 2                       # NPA relaxation level
M = 6                           # Num of terms in Gauss-Radau quadrature = 2*M
VERBOSE = 1                     # Relate to how detail of the info will be printed
N_WORKER_QUAD = 4               # Number of workers for parallelly computing quadrature
N_WORKER_SDP = 2                # Number of threads for solving a single SDP
PRIMAL_DUAL_GAP = 1e-5          # Allowable gap between primal and dual
SOLVER_CONFIG = ['mosek', {'dparam.intpnt_co_tol_rel_gap': PRIMAL_DUAL_GAP,
                           'iparam.num_threads': N_WORKER_SDP,
                           'iparam.infeas_report_level': 4}]
# SOLVER_CONFIG = ['sdpa']
ACCURATE_DIGIT = 4              # Achievable precision of the solver
WIN_TOL = 1e-4                  # Relax the precise winning prob constraint to a range with epsilon
ZERO_PROB = 1e-9                # Treat this value as zero for zero probability constraints
QUAD_END = True                 # Do the optimization for the last term (endpoint) of the Gauss-Radau quadrature or not

def truncate(number, digit):
    """
        Truncate the floating number up to the given digit after point
        It will ignore any number after the digit
    """
    truncated_number = round(number, digit)
    if round(number, digit+1) < truncated_number:
        truncated_number -= 10 ** (-digit)
    return truncated_number

def zeroConstr(P, positions, err_tol = 0):
    """
        Set the zero probability constraints for the positions with tolerable error.
        Note: err_tol - P([0,0], [0,0]) is treated as err_tol - P([0,0], [0,0]) >= 0.
    """
    constr = []
    for pos in positions:
        constr += [err_tol-P(*pos)]

    return constr

def winProb(P, scenario = [[2,2],[2,2]], inp_probs = []):
    """
        Function of winning probability of nonlocal game
        Here we use a modified CHSH non-local game for example
        - P:        P([a,b],[x,y])=P(ab|xy) is the probability that 
                        get the outputs (a,b) for given inputs (x,y)
        - scenario: Tell the number of inputs and number of outputs for each input
                        in each party may be useful for general scenario
        - inp_probs: The probability of the inpus P(xy)
    """
    try:
        configA = scenario[0]
        num_x = len(configA)
        configB = scenario[1]
        num_y = len(configB)
    except IndexError:
        print(f'Wrong input scenario: {scenario}')
    win_prob = 0
    for x in range(num_x):
        for y in range(num_y):
            for a in range(configA[x]):
                for b in range(configB[y]):
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

def inner_quad_obj_two(POVMs, inputs, Z_ab, t_i):
    """
        The inner-quadrature objective function to compute 
          the two-party randomness with BFF21 method
        - A, B : Alice and Bob's measurements in CG form
        - inputs : Alice and Bob's inputs for the computation
        - Z_ab : Eve's operators
        - t_i : Nodes in Gauss-Radau quadrature
    """
    x_star, y_star = inputs
    A, B = POVMs
    povmA = [*A[x_star], 1-sum(A[x_star])]
    povmB = [*B[y_star], 1-sum(B[y_star])]
    obj = 0
    for a in range(2):
        for b in range(2):
            ab = a*2+b
            obj += povmA[a]*povmB[b]*(Z_ab[ab] + Dagger(Z_ab[ab]) \
                    + (1-t_i)*Dagger(Z_ab[ab])*Z_ab[ab])
            obj += t_i *(Z_ab[ab]*Dagger(Z_ab[ab]))
    return obj

def innerQuad(P, Z, inputs, quad_t, quad_w, win_prob_expr, win_prob, win_tol, obj_func,
              zero_pos = [], zero_tol = 1e-9, substs = {}, extra_monos = [],
              level = 2, solver_config = [], verbose = 1):
    
    # Objective inside the quadrature summation
    obj = obj_func(P.parties, inputs, Z, quad_t)

    # Winning probability constraints
    p_win_constr = [win_prob_expr-win_prob+win_tol,
                   -win_prob_expr+win_prob+win_tol]
    
    # Zero-probability constraints
    zero_constr = zeroConstr(P, zero_pos, zero_tol)
    
    # Initialize an SDP problem and setup the objective and constraints
    ops = P.get_all_operators()+Z
    sdp = ncp.SdpRelaxation(ops, verbose=max(verbose-3, 0))
    sdp.get_relaxation(level = level, objective = obj,
                        substitutions = substs,
                        momentinequalities = zero_constr + p_win_constr,
                        extramonomials = extra_monos)

    sdp.solve(*solver_config)

    if verbose >= 2:
        print(sdp.status, sdp.primal, sdp.dual)

    assert sdp.status == 'optimal' or 'feasible' in sdp.status, 'Not solvable!'
    if sdp.status != 'optimal' and verbose >= 1:
        print('Solution does not reach optimal!')
    
    coeff = quad_w/(quad_t * math.log(2))
    entropy_in_quad = coeff * (1 + sdp.dual)
    p_win_in_quad = sdp[win_prob_expr]
    
    if verbose >= 2:
        print(f'entropy in quad: {entropy_in_quad}')
        print(f'winning probability in quad: {p_win_in_quad}')
    
    p_win_dual_vec = np.array([sdp.get_dual(constr) for constr in p_win_constr])
    p_win_dual_vec = np.squeeze(p_win_dual_vec)
    
    if not p_win_dual_vec.shape:
        p_win_dual_var = p_win_dual_vec*1
    elif len(p_win_dual_vec) == 1:
        p_win_dual_var = p_win_dual_var[0]
    else:
        # print(p_win_dual_vec)
        p_win_dual_var = p_win_dual_vec[0]-p_win_dual_vec[1]
    
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
    
def zeroPos2str(pos_list):
    pos_list = np.array(pos_list)
    pos_str = []
    for pos in pos_list:
        flat_pos = pos.reshape(np.sum(pos.shape))
        pos_str.append( ''.join([str(i) for i in flat_pos]) )
    return pos_str

# Setup of the scenario for Alice and Bob
configA = [2,2]
configB = [2,2]
SCENARIO = (configA, configB)

# Generate probability distribution P(ab|xy): P([a,b],[x,y])
P = ncp.Probability(configA, configB)

# Generate the set of POVMs for each measurement per party.
#   Note that we use CG form here, for example if Alice first measurement has two outcomes,
#   i.e., {A0, I - A0}, then the POVM for second outcome is no need to record.
#   In the case of CHSH, we have A = [[A0], [A1]]; B = [[B0], [B1]].
A, B = P.parties
# Generate the set of substitutable operators, e.g., A0B0 = B0A0.
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

# Put the constraints on the moment of the operators here.
#   E.g., the zero-probability constraints for class 2a: P(00|00) = P(11|00) = 0.
zero_pos  = [([0,0],[0,0]),([1,1],[0,0])]
moment_ineqs = zeroConstr(P, zero_pos, ZERO_PROB)

# Init SdpRelaxation
sdp_Q = ncp.SdpRelaxation(P.get_all_operators(), verbose=VERBOSE-3)

# Generate moment matrix and constraints up to given level.
#   Note that the ncpol package only provide minimization,
#   so a negative sign is need to do the maximization.
sdp_Q.get_relaxation(level = LEVEL, objective = -winProb(P),
                    substitutions = P.substitutions,
                    momentinequalities = moment_ineqs)
sdp_Q.solve(*SOLVER_CONFIG)

print('Status\tPrimal\tDual')
print(f'{sdp_Q.status}\t{sdp_Q.primal}\t{sdp_Q.dual}')

# Maximal quantum winning probability estimated with NPA set of given level
max_win = truncate(-sdp_Q.primal, ACCURATE_DIGIT)

inp_config = tuple(len(SCENARIO[i]) for i in range(len(SCENARIO)))
inp_probs = np.ones(inp_config)/np.prod(inp_config)

win_prob_expr = winProb(P, scenario = SCENARIO, inp_probs = inp_probs)

# Generate nodes, weights of quadrature up to 2*M terms
QUAD_ENDPOINT = .999 if QUAD_END else 1.
T, W = chaospy.quad_gauss_radau(M, chaospy.Uniform(0, QUAD_ENDPOINT), QUAD_ENDPOINT)    
T = T[0]
NUM_NODE = len(T)
if not QUAD_END:
    NUM_NODE = NUM_NODE - 1

inputs = (1,1)
win_prob = max_win

# Optimize for each node of the Gauss-Radau quadrature
#   The last node located at 1 is a constant, we donot need to optimize
results = Parallel(n_jobs=N_WORKER_QUAD, verbose=0)(
            delayed(innerQuad)(P, Z_ab, inputs, T[i], W[i], win_prob_expr,
                                win_prob, WIN_TOL, inner_quad_obj_two,
                                zero_pos, ZERO_PROB,
                                substs, extra_monos, LEVEL,
                                SOLVER_CONFIG, 1) for i in range(NUM_NODE))

print(f'results:{results}')

p_win_quad, entropy_quad, lambda_quad, p_zero_quad = zip(*results)

zero_pos_str = zeroPos2str(zero_pos)
lambda_zp_str = [f'lambda_{pos}' for pos in zero_pos_str]
zp_str = [f'p_zero_{pos}' for pos in zero_pos_str]
metadata = ['win_prob', 'entropy', 'lambda', *lambda_zp_str, *zp_str]
headline = ' '.join(metadata)
print(headline)

for p_win, entropy, lambda_, p_zero in zip(p_win_quad, entropy_quad, lambda_quad, p_zero_quad):
    lambda_vals = [f'{val:5g}' for val in lambda_]
    lambda_str = '\t'.join(lambda_vals)
    p_zero_vals = [f'{val:5g}' for val in p_zero]
    p_zero_str = '\t'.join(p_zero_vals)
    line = f'{p_win:.5g}\t{entropy:.5g}\t'+lambda_str+p_zero_str
    print(line)

print("")
# print(f'P_win_quad:{p_win_quad}')
# print(f'entropy_quad:{entropy_quad}')

# The lower bound on the von Neumann entropy H(AB|XYE') (asymptotic rate)
entropy = np.sum(np.array(entropy_quad))

print(entropy)
