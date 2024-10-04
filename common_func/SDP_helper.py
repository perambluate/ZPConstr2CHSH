import numpy as np
from sympy.physics.quantum.dagger import Dagger
from joblib import Parallel, delayed
import ncpol2sdpa as ncp
from scipy import stats
import chaospy
import math

def zero_position_map():
    return {'3a': [([0,0],[0,0]),([1,1],[0,1]),([1,1],[1,0])],
            '3b': [([0,0],[0,0]),([1,1],[0,0]),([1,0],[1,1])],
            '2a': [([0,0],[0,0]),([1,1],[0,0])],
            '2b': [([0,0],[0,0]),([1,1],[1,0])],
            '2b_swap': [([0,0],[0,0]),([1,1],[0,1])],
            '2c': [([0,0],[0,0]),([1,0],[1,1])],
            '1' : [([0,0],[0,0])]}


def inner_quad_obj_blind(POVMs, inputs, Z_ab, t_i):
    """
        The inner-quadrature objective function to compute 
          the blind randomness with BFF21 method
        - POVMs : Alice and Bob's POVMs in CG form
        - inputs : Alice and Bob's inputs for the computation
        - Z_ab : Eve's operators
        - t_i : Nodes in Gauss-Radau quadrature
    """
    x_star, y_star = inputs
    A, B = POVMs
    povmA = [*A[x_star], 1-sum(A[x_star])]
    povmB = [*B[y_star], 1-sum(B[y_star])]
    obj = 0
    num_out = [len(A[x_star])+1, len(B[y_star])+1]
    for a in range(num_out[0]):
        for b in range(num_out[1]):
            ab = a*num_out[0]+b
            obj += povmA[a]*povmB[b]*(Z_ab[ab] + Dagger(Z_ab[ab]) \
                    + (1-t_i)*Dagger(Z_ab[ab])*Z_ab[ab])
            obj += t_i * povmB[b]*(Z_ab[ab]*Dagger(Z_ab[ab]))
    return obj

def inner_quad_obj_one(POVMs, x_star, Z_a, t_i):
    """
        The inner-quadrature objective function to compute 
          the single-party randomness with BFF21 method
        - POVMs : Alice and Bob's POVMs in CG form
        - x_star : Alice for the key generation
        - Z_a : Eve's operators
        - t_i : Nodes in Gauss-Radau quadrature
    """
    A, _ = POVMs
    povmA = [*A[x_star], 1-sum(A[x_star])]
    obj = 0
    for a in range(2):
        obj += povmA[a]*(Z_a[a] + Dagger(Z_a[a]) + (1-t_i)*Dagger(Z_a[a])*Z_a[a])
        obj += t_i *(Z_a[a]*Dagger(Z_a[a]))
    return obj

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

OBJ_FUNC_MAP = {'blind': inner_quad_obj_blind,
                'one': inner_quad_obj_one,
                'two': inner_quad_obj_two}

def innerQuad(P, Z, inputs, quad_t, quad_w, obj_func, win_prob_expr, p_win_constr,
              zero_pos = [], zero_constr = [], substs = {}, extra_monos = [],
              sdp = None, npa_params = {}, level = 2, solver_config = [], verbose = 1):
    """
        Optimize the SDP in the qudrature with the node 'quad_t' and weight 'quad_w'.
        Inputs:
            - P: Probability object from ncpol2sdpa package
            - Z: Non-hermitian operators act on Eve's system required by BFF21 method
            - inputs: the target inputs to generate randomness
            - quad_t: the node of the quadrature
            - quad_w: the weight of the quadrature
            - obj_func: objective function
            - win_prob_expr: expr of the winning probability
            - p_win_constr: the win prob (WP) constraint
            - zero_pos: the pos of the prob to enforce zero-prob (ZP) constr
            - zero_constr: the ZP constr
            - substs: a dictionary recording the equivalence relationship between moments
            - extra_monos: extra monomials beyond the moments constructed by the
                           POVM elements of the legitimate parties
            - sdp: an SdpRelaxation object from ncpol2sdpa package
            - npa_params: parallel related parameters to construct the NPA relaxation
            - level: NPA level
            - solver_config: solver-related config;
                             formate: ['solver_name', {'solver_param': param_val}]
            - verbose: verbose level
        Outputs:
            - p_win_in_quad: the value of the winning probability when the SDP is solved
            - entropy_in_quad: entropy corresponding to the node of the quadrature
            - lambda_in_quad: dual variables for WP and ZP constraints
            - min_p_zero_in_quad: the minimal probability among those probabilities in the ZP constraints
    """
    
    # Objective inside the quadrature summation
    obj = obj_func(P.parties, inputs, Z, quad_t)
    
    if sdp is None:
        # Initialize an SDP problem and setup the objective and constraints
        ops = P.get_all_operators()+Z
        sdp = ncp.SdpRelaxation(ops, verbose=max(verbose-2, 0), **npa_params)
        sdp.get_relaxation(level = level, objective = obj,
                            substitutions = substs,
                            momentequalities = [],
                            momentinequalities = zero_constr + p_win_constr,
                            extramonomials = extra_monos)
        
    else:
        sdp.set_objective(obj)

    sdp.solve(*solver_config)

    if verbose:
        print(quad_t)
        print(sdp.status, sdp.primal, sdp.dual)

    assert sdp.status == 'optimal' or \
           ('feasible' in sdp.status and 'infeasible' not in sdp.status), \
           f'Status {sdp.status}, not promised solved'
    if sdp.status != 'optimal' and verbose >= 1:
        print('Solution does not reach optimal!')
    
    coeff = quad_w/(quad_t * math.log(2))
    entropy_in_quad = coeff * (1 + sdp.dual)
    p_win_in_quad = sdp[win_prob_expr]
    
    if verbose >= 2:
        print(f'Winning probability in quad: {p_win_in_quad}')
        if verbose >= 3:
            print(f'Entropy in quad: {entropy_in_quad}')
            if verbose >= 4:
                printProb(sdp,P)
                printNorm(sdp,Z)
    
    p_win_dual_vec = np.array([sdp.get_dual(constr) for constr in p_win_constr])
    p_win_dual_vec = np.squeeze(p_win_dual_vec)
    
    if not p_win_dual_vec.shape:
        p_win_dual_var = p_win_dual_vec*1
    elif len(p_win_dual_vec) == 1:
        p_win_dual_var = p_win_dual_vec[0]
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
        if verbose >= 2:
            print(f'Largest value among ZPs: {max(min_p_zero_in_quad)}')
        return p_win_in_quad, entropy_in_quad, lambda_in_quad, min_p_zero_in_quad
    else:
        lambda_in_quad = coeff * p_win_dual_var
        return p_win_in_quad, entropy_in_quad, lambda_in_quad, np.empty(0)

def singleRoundEntropy(rand_type, P, Z, M, inputs, win_prob_func, win_prob,
                       scenario = ([2,2],[2,2]), inp_probs = np.empty(0),
                       win_tol = 1e-4, zero_class = '', zero_tol = 1e-9,
                       substs = {}, extra_monos = [], level = 2,
                       quad_end = True, quad_ep = 1,
                       n_worker_quad = 1, n_worker_npa = 1,
                       solver_config = [], verbose = 1):
    """
        Compute the single-rounded (von Neumann) entropy for given type of DI randomness,
            inputs, winning probability, and the class of zero-probability.
        Inputs:
            - rand_type: type of DI randomness
            - P: Probability object from ncpol2sdpa package
            - Z: Non-hermitian operators act on Eve's system required by BFF21 method
            - M: num of terms in the quadrature
            - inputs: the target inputs to generate randomness
            - win_prob_func: a func of P that returns an expr of WP
            - win_prob: val of WP
            - scenario: Bell scenario;
                        formate: ([numOut_part1_meas1, numOut_part1_meas2, ...], ...)
                        e.g. ([3,3], [2,2,2]): 2-party,
                                               part1: 2-inp 3-out,
                                               part2: 3-inp, 2-out
            - inp_probs: input prob; inp_probs[x,y] = p(xy)
            - win_tol: tolerance for WP constr
            - zero_class: class of ZP constr
            - zero_tol: tolerance for ZP constr
            - substs: a dictionary recording the equivalence relationship between moments
            - extra_monos: extra monomials beyond the moments constructed by the
                           POVM elements of the legitimate parties
            - level: NPA level
            - quad_end: whether to compute and include the last term of quadrature or not
            - quad_ep: endpoint node of the quadrature; should be choosen within (0,1]
                       and as close to 1 as possible
            - n_worker_quad: num of workers to compute the quadrature parallelly
            - n_worker_npa: num of threads to construct the NPA hierarchy parallelly
            - solver_config: solver-related config;
                             formate: ['solver_name', {'solver_param': param_val}]
            - verbose: verbose level
        Outputs:
            - entropy: single-rounded (von Neumann) entropy
            - lambda_: dual vars for WP and ZP constr
            - c_lambda: constant in the Lagrange dual function
    """
    if verbose:
        print(f'Win prob: {win_prob}')
    
    inp_config = tuple(len(scenario[i]) for i in range(len(scenario)))
    inp_probs = np.array(inp_probs)
    if inp_probs.size == 0:
        inp_probs = np.ones(inp_config)/np.prod(inp_config)
    else:
        assert inp_probs.shape == inp_config, 'Wrong shape of inp_prob in singleRoundEntropy!'

    win_prob_expr = win_prob_func(P)

    # Nodes, weights of quadrature up to 2*M terms
    T, W = chaospy.quadrature.radau(M, chaospy.Uniform(0, quad_ep), quad_ep)
    T = T[0]
    
    if verbose >= 3:
        print(f'Nodes of the Gauss-Radau quadrature:\n{T}')
        print(f'Weights of the Gauss-Radau quadrature:\n{W}')

    NUM_NODE = len(T)
    if not quad_end:
        NUM_NODE = NUM_NODE - 1

    assert rand_type in OBJ_FUNC_MAP, "Wrong 'rand_type' when calling singleRoundEntropy()!"
    obj_func = OBJ_FUNC_MAP[rand_type]

    # Winning probability constraints
    p_win_constr = [win_prob_expr-win_prob+win_tol]
    
    # Zero-probability constraints
    zero_pos = zero_position_map().get(zero_class, [])
    zero_constr = zeroConstr(P, zero_pos, zero_tol)

    # Optimize for each node of the Gauss-Radau quadrature
    # The last node located at 1 is a constant, we donot need to optimize
    if n_worker_quad == 1:
        ops = P.get_all_operators()+Z
        sdp = ncp.SdpRelaxation(ops, verbose=max(verbose-2, 0))
        sdp.get_relaxation(level = level, objective = 0,
                            substitutions = substs,
                            momentequalities = [],
                            momentinequalities = zero_constr + p_win_constr,
                            extramonomials = extra_monos)
        p_win_quad = []
        entropy_quad = []
        lambda_quad = []
        p_zero_quad = []
        for i in reversed(range(NUM_NODE)):
            p_win, ent, lamb, p_zero = innerQuad(P, Z, inputs, T[i], W[i], obj_func, win_prob_expr,
                                                 p_win_constr, zero_pos, zero_constr, substs,
                                                 extra_monos, sdp = sdp, solver_config = solver_config,
                                                 verbose = verbose)
            p_win_quad += [p_win]
            entropy_quad += [ent]
            lambda_quad += [lamb]
            p_zero_quad += [p_zero]

    else:
        npa_params = dict(parallel=True, number_of_threads=n_worker_npa)
        results = Parallel(n_jobs=n_worker_quad, verbose=0)(
                    delayed(innerQuad)(P, Z, inputs, T[i], W[i], obj_func, win_prob_expr, p_win_constr,
                                       zero_pos, zero_constr, substs, extra_monos,
                                       npa_params = npa_params, level = level, solver_config = solver_config,
                                       verbose = verbose) for i in reversed(range(NUM_NODE)))

        p_win_quad, entropy_quad, lambda_quad, p_zero_quad = zip(*results)

    max_p_win = max(p_win_quad)
    entropy = np.sum(np.array(entropy_quad)) #-1/(len(T)**2 * math.log(2)) + W[-1]/(T[-1]*math.log(2))
    if verbose:
        print(f'Entropy: {entropy}')
    lambda_ = np.sum(np.array(lambda_quad), axis=0)
    p_zero_quad = np.array(p_zero_quad)

    if np.any(p_zero_quad):
        min_p_zero = np.min(p_zero_quad, axis=0)
        c_lambda = entropy - lambda_[0] * max_p_win + np.sum(lambda_[1:] * min_p_zero)
    else:
        c_lambda = entropy - lambda_ * max_p_win
    return entropy, lambda_, c_lambda

def BellFunction(operator_list, polynomial):
    """
        Construct Bell function from given name of operator and its coefficient
    """
    Bell_function = 0
    operator_names = [op.__repr__() for op in operator_list]
    for monomial, coeff in polynomial.items():
        if monomial == 'id':
            Bell_function += coeff
        else:
            ops = monomial.split('*')
            new_term = coeff
            for op in ops:
                try:
                    op_index = operator_names.index(op)
                except ValueError:
                    print(f'Operator {op} not in operator_list.')
                    return
                new_term *= operator_list[op_index]
            Bell_function += new_term
    return Bell_function

def maxBellFunc(scenario, bell_function,
                zero_positions = [], zero_tol = 0, constraints = {},
                level = 2, verbose = 1, solver_config = {}, showProb = False):
    """
        Maxmize given Bell function in given scenario
    """
    P = ncp.Probability(*scenario)
    all_operators = P.get_all_operators()
    
    sdp = ncp.SdpRelaxation(all_operators, verbose = verbose)
    
    objective = BellFunction(all_operators, bell_function)
    substs = P.substitutions
    
    constr = {'equalities': [], 'inequalities': [],
               'momentequalities': [], 'momentinequalities': []}

    if constraints:
        for key in list(set(constraints.keys()) & set(constr.keys())):
            polys = constraints[key]
            for poly in polys:
                constr[key].append(BellFunction(all_operators, poly))

    if zero_positions:
        constr['momentinequalities'] += zeroConstr(P, zero_positions, zero_tol)
    
    sdp.get_relaxation(level = level, objective = -objective,
                       substitutions = substs, **constr)
    sdp.solve(*solver_config)

    if verbose:
        print(sdp.status, sdp.primal, sdp.dual)

    if showProb:
        printProb(sdp, P, scenario)
    
    return sdp.status, sdp.primal, sdp.dual

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
        Set the zero probability constraints for the positions with tolerable error 
    """
    constr = []
    for pos in positions:
        constr += [err_tol-P(*pos)]

    return constr

def probConstr(P, scenario=[[2,2],[2,2]]):
    """
        Set the constraints for physical probabilities
    """
    try:
        configA = scenario[0]
        num_x = len(configA)
        configB = scenario[1]
        num_y = len(configB)
    except IndexError:
        print(f'Wrong input scenario: {scenario}')
    return [P([a,b],[x,y]) for x in range(num_x) for y in range(num_y) \
            for a in range(configA[x]) for b in range(configB[y])]

def alphaConstr(Zs, t_i):
    """
        Constraints for Eve's operators for i-th term of the quadrature
        This can be check manually after optimization
    """
    alpha = max(1/t_i, 1/(1-t_i)) * 3/2
    constr = []
    for z in Zs:
        constr += [alpha - Dagger(z)*z]

    return constr

def printProb(sdp, P, scenario=[[2,2],[2,2]]):
    """
        Print probabilities (only works in bipartite scenario).
    """
    try:
        configA = scenario[0]
        num_x = len(configA)
        configB = scenario[1]
        num_y = len(configB)
    except IndexError:
        print(f'Wrong input scenario: {scenario}')
    for y in range(num_y):
        for b in range(configB[y]):
            s = f'\t'.join([f'{sdp[P([a,b],[x,y])]:.4f}' \
                            for x in range(num_x) for a in range(configA[x])])
            print(s)

def printNorm(sdp, ops):
    """
        Print norm of the operators in SDP.
    """
    for op in ops:
        norm = sdp[Dagger(op)*op]
        print(f'{op}\tNorm: {norm:.5g}')

def zeroPos2str(pos_list):
    """
        Convert zero-probability position to string. (Useful when naming var in SDP)
    """
    pos_list = np.array(pos_list)
    pos_str = []
    for pos in pos_list:
        flat_pos = pos.reshape(np.sum(pos.shape))
        pos_str.append(''.join([str(i) for i in flat_pos]))
    return pos_str

def intervalSpacingBeta(interval, num_points, endpoint = True, a = 1.2, b = 0.8):
    """
        Generate points in the interval following Beta distribution.
    """
    dist = stats.beta(a, b)
    pp = np.linspace(*dist.cdf([0, 1]), num=num_points, endpoint = endpoint)
    spacing_arr = interval[0] + dist.ppf(pp) * (interval[1] - interval[0])
    return spacing_arr

