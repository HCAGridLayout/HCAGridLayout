import numpy as np
import operator
from scipy.optimize import (linear_sum_assignment, OptimizeResult)

from scipy._lib._util import check_random_state
import itertools
from scipy.optimize._qap import _common_input_validation, _split_matrix, _doubly_stochastic
# from lapjv import lapjv as lsa
from lsa import linear_sum_assignment as lsa
from scipy.sparse import csr_matrix


def _calc_score(A0, B0, additionAB, C, perm):
    ans = np.sum(A0 * B0[perm][:, perm])
    for A, B in additionAB:
        ans += np.sum(A * B[perm][:, perm])
    for i in range(len(perm)):
        ans += C[i][perm[i]]
    return ans


def quadratic_assignment_faq(A0, B0, additionAB, C, partial_match=None, rng=0, P0="barycenter", maxiter=30, tol=0.03, addition_sparse=True, emptyA=False):

    maxiter = operator.index(maxiter)

    # ValueError check
    A0, B0, partial_match = _common_input_validation(A0, B0, partial_match)
    if emptyA:
        A0 = np.zeros_like(A0)
    addition_n = len(additionAB)

    msg = None
    if isinstance(P0, str) and P0 not in {'barycenter', 'randomized'}:
        msg = "Invalid 'P0' parameter string"
    elif maxiter < 0:
        msg = "'maxiter' must be a positive integer"
    # elif tol < 0:
    #     msg = "'tol' must be a positive float"
    if msg is not None:
        raise ValueError(msg)

    rng = check_random_state(rng)
    n = len(A0)  # number of vertices in graphs
    n_seeds = len(partial_match)  # number of seeds
    n_unseed = n - n_seeds

    # [1] Algorithm 1 Line 1 - choose initialization
    if not isinstance(P0, str):
        P0 = np.atleast_2d(P0)
        if P0.shape != (n_unseed, n_unseed):
            msg = "`P0` matrix must have shape m' x m', where m'=n-m"
        elif ((P0 < 0).any() or not np.allclose(np.sum(P0, axis=0), 1)
              or not np.allclose(np.sum(P0, axis=1), 1)):
            msg = "`P0` matrix must be doubly stochastic"
        if msg is not None:
            raise ValueError(msg)
    elif P0 == 'barycenter':
        P0 = np.ones((n_unseed, n_unseed)) / n_unseed
    elif P0 == 'randomized':
        J = np.ones((n_unseed, n_unseed)) / n_unseed
        # generate a nxn matrix where each entry is a random number [0, 1]
        # would use rand, but Generators don't have it
        # would use random, but old mtrand.RandomStates don't have it
        K = _doubly_stochastic(rng.uniform(size=(n_unseed, n_unseed)))
        P0 = (J + K) / 2

    # check trivial cases
    if n == 0 or n_seeds == n:
        score = _calc_score(A0, B0, additionAB, C, partial_match[:, 1])
        res = {"col_ind": partial_match[:, 1], "fun": score, "nit": 0}
        return OptimizeResult(res)

    obj_func_scalar = 1

    nonseed_B = np.setdiff1d(range(n), partial_match[:, 1])

    nonseed_A = np.setdiff1d(range(n), partial_match[:, 0])
    perm_A = np.concatenate([partial_match[:, 0], nonseed_A])
    perm_B = np.concatenate([partial_match[:, 1], nonseed_B])

    # definitions according to Seeded Graph Matching [2].
    A0_11, A0_12, A0_21, A0_22 = _split_matrix(A0[perm_A][:, perm_A], n_seeds)
    B0_11, B0_12, B0_21, B0_22 = _split_matrix(B0[perm_B][:, perm_B], n_seeds)

    additionA_11, additionA_12, additionA_21, additionA_22 = [None]*addition_n, [None]*addition_n, [None]*addition_n, [None]*addition_n
    additionB_11, additionB_12, additionB_21, additionB_22 = [None]*addition_n, [None]*addition_n, [None]*addition_n, [None]*addition_n
    for id in range(addition_n):
        A, B = additionAB[id]
        additionA_11[id], additionA_12[id], additionA_21[id], additionA_22[id] = _split_matrix(A[perm_A][:, perm_A], n_seeds)
        additionB_11[id], additionB_12[id], additionB_21[id], additionB_22[id] = _split_matrix(B[perm_B][:, perm_B], n_seeds)
    if addition_sparse:
        ori_additionA_11, ori_additionA_12, ori_additionA_21, ori_additionA_22 = [None]*addition_n, [None]*addition_n, [None]*addition_n, [None]*addition_n
        ori_additionB_11, ori_additionB_12, ori_additionB_21, ori_additionB_22 = [None]*addition_n, [None]*addition_n, [None]*addition_n, [None]*addition_n
        for id in range(addition_n):
            ori_additionA_11[id], ori_additionA_12[id], ori_additionA_21[id], ori_additionA_22[id] = additionA_11[id], additionA_12[id], additionA_21[id], additionA_22[id]
            ori_additionB_11[id], ori_additionB_12[id], ori_additionB_21[id], ori_additionB_22[id] = additionB_11[id], additionB_12[id], additionB_21[id], additionB_22[id]
            additionA_11[id], additionA_12[id], additionA_21[id], additionA_22[id] = csr_matrix(additionA_11[id]), csr_matrix(additionA_12[id]), csr_matrix(additionA_21[id]), csr_matrix(additionA_22[id])
            # additionB_11[id], additionB_12[id], additionB_21[id], additionB_22[id] = csr_matrix(additionB_11[id]), csr_matrix(additionB_12[id]), csr_matrix(additionB_21[id]), csr_matrix(additionB_22[id])

    C_11, C_12, C_21, C_22 = _split_matrix(C[perm_A][:, perm_B], n_seeds)

    const_sum = A0_21 @ B0_21.T + A0_12.T @ B0_12
    for id in range(addition_n):
        const_sum += additionA_21[id] @ additionB_21[id].T + additionA_12[id].T @ additionB_12[id]

    P = P0
    # [1] Algorithm 1 Line 2 - loop while stopping criteria not met
    n_iter = 0
    # score = ((A0_22 @ P)*(P @ B0_22)).sum()
    # score += (P*C_22).sum()
    for n_iter in range(1, maxiter+1):
        # [1] Algorithm 1 Line 3 - compute the gradient of f(P) = -tr(APB^tP^t)
        # grad_fp = const_sum + A0_22 @ P @ B0_22.T + A0_22.T @ P @ B0_22
        grad_fp = const_sum.copy()
        if not emptyA:
            grad_fp += (A0_22 @ P @ B0_22.T + A0_22.T @ P @ B0_22)

        for id in range(addition_n):
            grad_fp += additionA_22[id] @ P @ additionB_22[id].T + additionA_22[id].T @ P @ additionB_22[id]

        c_alpha = 1.0
        # c_alpha = min(1.0, (3/2*maxiter-n_iter)/maxiter)
        grad_fp += C_22 * c_alpha

        # [1] Algorithm 1 Line 4 - get direction Q by solving Eq. 8

        shuffle_col = np.arange(len(grad_fp), dtype='int')
        # np.random.shuffle(shuffle_col)
        # cols = shuffle_col[lapjv(grad_fp[:, shuffle_col])[0]]
        cols = shuffle_col[lsa(grad_fp[:, shuffle_col])[0]]
        # cols = shuffle_col[linear_sum_assignment(grad_fp[:, shuffle_col], maximize=False)[1]]
        colsT = cols.copy()
        for i in range(len(cols)):
            colsT[cols[i]] = i

        Q = np.eye(n_unseed)[cols]

        # score = ((A0_22 @ Q)*(Q @ B0_22)).sum()
        # score += (Q*C_22).sum()

        # [1] Algorithm 1 Line 5 - compute the step size
        # Noting that e.g. trace(Ax) = trace(A)*x, expand and re-collect
        # terms as ax**2 + bx + c. c does not affect location of minimum
        # and can be ignored. Also, note that trace(A@B) = (A.T*B).sum();
        # apply where possible for efficiency.
        R = P - Q

        a = 0
        b = 0

        if not emptyA:
            b21 = ((R.T @ A0_21) * B0_21).sum()
            b12 = ((R.T @ A0_12.T) * B0_12.T).sum()
            AR22 = A0_22.T @ R
            BR22 = B0_22 @ R.T
            b22a = (AR22 * B0_22.T[cols]).sum()
            b22b = (A0_22 * BR22[cols]).sum()
            a += (AR22.T * BR22).sum()
            b += b21 + b12 + b22a + b22b

        for id in range(addition_n):
            if addition_sparse:
                # b21 = (ori_additionB_21[id] * (R.T @ additionA_21[id])).sum()
                # b12 = (ori_additionB_12[id].T * (R.T @ additionA_12[id].T)).sum()
                # AR22 = additionA_22[id].T @ R
                # BR22 = additionB_22[id] @ R.T
                # b22a = (ori_additionB_22[id].T * (AR22[colsT])).sum()
                # b22b = additionA_22[id].multiply(BR22[cols]).sum()
                # a += (AR22.T * BR22).sum()
                # b += b21 + b12 + b22a + b22b
                b21 = (ori_additionB_21[id] * (R.T @ additionA_21[id])).sum()
                b12 = (ori_additionB_12[id].T * (R.T @ additionA_12[id].T)).sum()
                AR22 = additionA_22[id].T @ R
                BR22 = additionB_22[id] @ R.T
                b22a = (ori_additionB_22[id].T * (AR22[colsT])).sum()
                b22b = (ori_additionA_22[id] * (BR22[cols])).sum()
                a += (AR22.T * BR22).sum()
                b += b21 + b12 + b22a + b22b
            else:
                b21 = ((R.T @ additionA_21[id]) * additionB_21[id]).sum()
                b12 = ((R.T @ additionA_12[id].T) * additionB_12[id].T).sum()
                AR22 = additionA_22[id].T @ R
                BR22 = additionB_22[id] @ R.T
                b22a = (AR22 * additionB_22[id].T[cols]).sum()
                b22b = (additionA_22[id] * BR22[cols]).sum()
                a += (AR22.T * BR22).sum()
                b += b21 + b12 + b22a + b22b

        b += (C_22 * R).sum() * c_alpha

        # critical point of ax^2 + bx + c is at x = -d/(2*e)
        # if a * obj_func_scalar > 0, it is a minimum
        # if minimum is not in [0, 1], only endpoints need to be considered
        if a*obj_func_scalar > 0 and 0 <= -b/(2*a) <= 1:
            alpha = -b/(2*a)
        else:
            alpha = np.argmin([0, (b + a)*obj_func_scalar])

        # if n_iter == maxiter:
        #     alpha = 0

        # alpha = 0

        # [1] Algorithm 1 Line 6 - Update P
        P_i1 = alpha * P + (1 - alpha) * Q
        if np.linalg.norm(P - P_i1) / np.sqrt(n_unseed) < tol:
            P = P_i1
            break
        P = P_i1

        # score = ((A0_22 @ P)*(P @ B0_22)).sum()
        # score += (P*C_22).sum()

        # perm = np.concatenate((np.arange(n_seeds), cols + n_seeds))
        # unshuffled_perm = np.zeros(n, dtype=int)
        # unshuffled_perm[perm_A] = perm_B[perm]
        # score = _calc_score(A0, B0, additionAB, np.zeros((len(A0), len(A0))), unshuffled_perm)
        # print("score it", n_iter, score)

    # [1] Algorithm 1 Line 7 - end main loop

    # [1] Algorithm 1 Line 8 - project onto the set of permutation matrices

    # _, col = linear_sum_assignment(P, maximize=True)
    # _, col = linear_sum_assignment(P @ B0_22, maximize=False)
    # col = lsa(-P)[0]
    # col = lapjv(-P)[0]
    col = lsa(P @ B0_22)[0]

    perm = np.concatenate((np.arange(n_seeds), col + n_seeds))

    unshuffled_perm = np.zeros(n, dtype=int)
    unshuffled_perm[perm_A] = perm_B[perm]

    score = _calc_score(A0, B0, additionAB, C, unshuffled_perm)
    res = {"col_ind": unshuffled_perm, "fun": score, "nit": n_iter}
    return OptimizeResult(res)