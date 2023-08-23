import numpy as np
from sympy.physics.quantum.dagger import Dagger
from joblib import Parallel, delayed
import ncpol2sdpa as ncp
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

def max_win_map():
    return  {'chsh': 0.8535, '1': 0.8294, '2a': 0.8125,
             '2b': 0.8125, '2b_swap': 0.8125, '2c': 0.8039,
             '3a': 0.7951, '3b': 0.7837}


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

def innerQuad(P, Z, inputs, quad_t, quad_w, win_prob_expr, win_prob, win_tol, obj_func,
              zero_class = '', zero_tol = 1e-9, substs = {}, extra_monos = [],
              level = 2, solver_config = [], verbose = 1):
    
    # Objective inside the quadrature summation
    obj = obj_func(P.parties, inputs, Z, quad_t)

    # Winning probability constraints
    p_win_constr = [win_prob_expr-win_prob+win_tol,
                   -win_prob_expr+win_prob+win_tol]
    
    # Zero-probability constraints
    zero_pos = zero_position_map().get(zero_class, [])
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
        if verbose >= 3:
            printProb(sdp,P)
            printNorm(sdp,Z)
    
    p_win_dual_vec = np.array([sdp.get_dual(constr) for constr in p_win_constr])
    p_win_dual_vec = np.squeeze(p_win_dual_vec)
    
    if not p_win_dual_vec.shape:
        p_win_dual_var = p_win_dual_vec*1
    elif len(p_win_dual_vec) == 1:
        p_win_dual_var = p_win_dual_var[0]
    else:
        # print(p_win_dual_vec)
        p_win_dual_var = p_win_dual_vec[0]-p_win_dual_vec[1]
    
    zero_pos = zero_position_map().get(zero_class, [])
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

def singleRoundEntropy(rand_type, P, Z, M, inputs, win_prob_func, win_prob,
                       scenario = ([2,2],[2,2]), inp_probs = np.empty(0), win_tol = 1e-4,
                       zero_class = '', zero_tol = 1e-9, substs = {}, extra_monos = [],
                       level = 2, n_worker_quad = 1, solver_config = [], verbose = 1):

    inp_config = tuple(len(scenario[i]) for i in range(len(scenario)))
    if inp_probs.size == 0:
        inp_probs = np.ones(inp_config)/np.prod(inp_config)
    else:
        assert inp_probs.shape == inp_config, 'Wrong shape of inp_prob in singleRoundEntropy!'

    win_prob_expr = win_prob_func(P, scenario = scenario, inp_probs = inp_probs)

    # Nodes, weights of quadrature up to 2*M terms
    T, W = chaospy.quad_gauss_radau(M, chaospy.Uniform(0, 1), 1)    
    T = T[0]
    NUM_NODE = len(T)
    if verbose >= 2:
        print(f'Number of terms summed in quadrature: {NUM_NODE}')
        print(f'Nodes of the Gauss-Radau quadrature:\n{T}')
        print(f'Weights of the Gauss-Radau quadrature:\n{W}')

    entropy = -1/(len(T)**2 * math.log(2)) + W[-1]/(T[-1]*math.log(2))

    assert rand_type in OBJ_FUNC_MAP, "Wrong 'rand_type' when calling singleRoundEntropy()!"
    obj_func = OBJ_FUNC_MAP[rand_type]

    # Optimize for each node of the Gauss-Radau quadrature
    # The last node located at 1 is a constant, we donot need to optimize
    results = Parallel(n_jobs=n_worker_quad, verbose=0)(
                delayed(innerQuad)(P, Z, inputs, T[i], W[i], win_prob_expr,
                                   win_prob, win_tol, obj_func,
                                   zero_class, zero_tol,
                                   substs, extra_monos, level,
                                   solver_config, verbose) for i in range(NUM_NODE-1))
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
    try:
        configA = scenario[0]
        num_x = len(configA)
        configB = scenario[1]
        num_y = len(configB)
    except IndexError:
        print(f'Wrong input scenario: {scenario}')
    for y in range(num_y):
        for b in range(configB[y]):
            s = f'\t'.join([f'{sdp[P([a,b],[x,y])]:.5g}' \
                            for x in range(num_x) for a in range(configA[x])])
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
