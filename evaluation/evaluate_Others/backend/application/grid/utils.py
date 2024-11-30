import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from scipy.spatial.distance import squareform
import time
from networkx.drawing.layout import _process_params, rescale_layout, random_layout, circular_layout, _kamada_kawai_costfn

def AUC(y):
    cnt = 0
    a_20 = 0
    a_full = 0
    for i in range(len(y)):
        cnt += y[i]
        if i == 19:
            a_20 = cnt
    a_full = cnt
    return a_full, a_20


def kamada_kawai_layout(
    G, dist=None, pos=None, weight="weight", scale=1, center=None, dim=2, tol=None, options=None,
    stability=None
):

    G, center = _process_params(G, center, dim)
    nNodes = len(G)
    if nNodes == 0:
        return {}

    if dist is None:
        dist = dict(nx.shortest_path_length(G, weight=weight))
    dist_mtx = 1e6 * np.ones((nNodes, nNodes))
    for row, nr in enumerate(G):
        if nr not in dist:
            continue
        rdist = dist[nr]
        for col, nc in enumerate(G):
            if nc not in rdist:
                continue
            dist_mtx[row][col] = rdist[nc]

    if pos is None:
        if dim >= 3:
            pos = random_layout(G, dim=dim)
        elif dim == 2:
            pos = circular_layout(G, dim=dim)
        else:
            pos = dict(zip(G, np.linspace(0, 1, len(G))))
    pos_arr = np.array([pos[n] for n in G])

    pos = _kamada_kawai_solve(dist_mtx, pos_arr, dim, tol=tol, options=options, stability=stability)

    # pos = rescale_layout(pos, scale=scale) + center
    return dict(zip(G, pos))


def _kamada_kawai_solve(dist_mtx, pos_arr, dim, tol=None, options=None, stability=None):

    import scipy as sp

    meanwt = 1e-3

    if stability is None:
        costargs = (np, 1 / (dist_mtx + np.eye(dist_mtx.shape[0]) * 1e-3), meanwt, dim)

        optresult = sp.optimize.minimize(
            _kamada_kawai_costfn,
            pos_arr.ravel(),
            method="L-BFGS-B",
            args=costargs,
            jac=True,
            tol=tol,
            options=options
        )
    else:
        costargs = (np, 1 / (dist_mtx + np.eye(dist_mtx.shape[0]) * 1e-3), meanwt, dim, stability["old_pos"], stability["norm_ratio"], stability["sta_alpha"], stability["keep"])
        optresult = sp.optimize.minimize(
            _kamada_kawai_costfn_stability,
            pos_arr.ravel(),
            method="L-BFGS-B",
            args=costargs,
            jac=True,
            tol=tol,
            options=options
        )

    # print("iters, cost", optresult.nit, optresult.fun)
    return optresult.x.reshape((-1, dim))

def get_kamada_kawai_costfn_stability(G, dist=None, pos=None, weight="weight", center=None, dim=2, old_pos=None, keep=None):
    G, center = _process_params(G, center, dim)
    nNodes = len(G)
    if nNodes == 0:
        return {}

    if dist is None:
        dist = dict(nx.shortest_path_length(G, weight=weight))
    dist_mtx = 1e6 * np.ones((nNodes, nNodes))
    for row, nr in enumerate(G):
        if nr not in dist:
            continue
        rdist = dist[nr]
        for col, nc in enumerate(G):
            if nc not in rdist:
                continue
            dist_mtx[row][col] = rdist[nc]

    invdist = 1 / (dist_mtx + np.eye(dist_mtx.shape[0]) * 1e-3)

    pos_arr = pos.reshape((nNodes, dim))
    old_pos_arr = old_pos.reshape((nNodes, dim))

    delta = pos_arr[:, np.newaxis, :] - pos_arr[np.newaxis, :, :]
    nodesep = np.linalg.norm(delta, axis=-1)
    direction = np.einsum("ijk,ij->ijk", delta, 1 / (nodesep + np.eye(nNodes) * 1e-3))

    offset = nodesep * invdist - 1.0
    offset[np.diag_indices(nNodes)] = 0

    cost1 = 0.5 * np.sum(offset ** 2)
    cost2 = np.sum(((pos_arr - old_pos_arr) ** 2)[keep])
    return cost1, cost2


def _kamada_kawai_costfn_stability(pos_vec, np, invdist, meanweight, dim, old_pos, norm_ratio, sta_alpha, keep):
    # Cost-function and gradient for Kamada-Kawai layout algorithm
    nNodes = invdist.shape[0]
    pos_arr = pos_vec.reshape((nNodes, dim))
    old_pos_arr = old_pos.reshape((nNodes, dim))

    delta = pos_arr[:, np.newaxis, :] - pos_arr[np.newaxis, :, :]
    nodesep = np.linalg.norm(delta, axis=-1)
    direction = np.einsum("ijk,ij->ijk", delta, 1 / (nodesep + np.eye(nNodes) * 1e-3))

    offset = nodesep * invdist - 1.0
    offset[np.diag_indices(nNodes)] = 0

    cost = 0.5 * np.sum(offset**2)
    cost = sta_alpha*cost + (1-sta_alpha)*np.sum(((pos_arr-old_pos_arr)**2)[keep])*norm_ratio
    grad = np.einsum("ij,ij,ijk->ik", invdist, offset, direction) - np.einsum(
        "ij,ij,ijk->jk", invdist, offset, direction
    )
    grad = sta_alpha*grad
    grad[keep] = grad[keep] + (1-sta_alpha)*2*(pos_arr-old_pos_arr)[keep]*norm_ratio

    # Additional parabolic term to encourage mean position to be near origin:
    sumpos = np.sum(pos_arr, axis=0)
    cost += 0.5 * meanweight * np.sum(sumpos**2)
    grad += meanweight * sumpos

    return (cost, grad.ravel())


def get_anchor_grids(feature):
    n = len(feature)
    nums = min(20, n)
    samples = np.random.choice(n, size=nums, replace=False)

    f_dist = cdist(feature[samples], feature[samples], 'euclidean')
    f_dist /= f_dist.max()

    # print(shld)
    f_dict = {}
    for i in range(nums):
        f_dict[i] = {}
        for j in range(nums):
            f_dict[i][j] = f_dist[i][j]

    G_full = nx.complete_graph(nums)
    pos = kamada_kawai_layout(G_full, dist=f_dict, tol=1e-3)
    result = np.zeros((nums, 2))
    for i in range(nums):
        result[i] = np.array(pos[i])

    father_dist = cdist(feature, feature[samples], 'euclidean')
    father = np.argmin(father_dist, axis=1)

    return result, father

if __name__=="__main__":
    grids = np.array([[0.1, 0.1]])
    sta_grids, father = get_anchor_grids(np.array([[0, 0, 0]]))
    sta_grids /= (sta_grids.max(axis=0) - sta_grids.min(axis=0))
    sta_grids *= (grids.max(axis=0) - grids.min(axis=0))
    sta_grids += grids.mean(axis=0) - sta_grids.mean(axis=0)
    # cost_matrix = np.zeros((n, n))
    # cost_matrix[selected_list, :] = np.power(cdist(sta_grids, grids, "euclidean"), 2)
    cost_matrix = np.power(cdist(sta_grids[father], grids, "euclidean"), 2)