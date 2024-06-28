import random

from scipy.optimize import quadratic_assignment
import numpy as np
import time
import math
from multiprocessing import dummy
from multiprocessing.pool import ThreadPool
from sklearn.manifold._t_sne import _joint_probabilities
from sklearn.manifold._t_sne import _joint_probabilities_nn
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from .qap import quadratic_assignment_faq
# from lapjv import lapjv as lsa
from lsa import linear_sum_assignment as lsa

import pickle

def save_pickle(data, file):
    with open(file, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

from sklearn.manifold._t_sne import _kl_divergence_bh, _kl_divergence, _gradient_descent
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix
from scipy.spatial.distance import squareform
from collections import Counter
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
import networkx as nx

def get_init_kmedoids(feature, c, grids):
    from .utils import kamada_kawai_layout
    n = len(feature)
    m = min(20, len(feature))

    dist = cdist(feature, feature, "eu") + ((1 - np.eye(n)) * 1e-5)
    kmedoids = KMedoids(n_clusters=m, random_state=42, metric='precomputed')
    father = kmedoids.fit_predict(dist)
    samples = kmedoids.medoid_indices_

    cost_matrix = np.zeros((n, n))
    for i in range(m):
        tmp_idx = (father == i)
        cost_matrix[tmp_idx] = c[tmp_idx].mean(axis=0)
    sol = lsa(cost_matrix)[0]
    centers = np.zeros((m, 2))
    grids = grids - grids.min(axis=0)
    grids = grids / (grids.max(axis=0)+1e-8)
    for i in range(m):
        tmp_idx = (father == i)
        centers[i] = grids[sol[tmp_idx]].mean(axis=0)
    # centers = centers + np.random.normal(0, 0.001, (m, 2))

    feature_mean = []
    for i in range(m):
        tmp_idx = (father == i)
        feature_mean.append(feature[tmp_idx].mean(axis=0))
    f_dist = cdist(feature_mean, feature_mean, 'eu')
    if m > 1:
        f_dist = squareform(f_dist) + 1e-6
        if f_dist.max() > f_dist.min() + 1e-6:
            f_dist = f_dist - f_dist.min()
        f_dist = squareform(f_dist)
        f_dist /= f_dist.max()
        tmp_min = 1 / np.sqrt(2) / (np.sqrt(m) - 0.999)
        tmp_delta = 1 - tmp_min
        f_dist = squareform(squareform(f_dist) * tmp_delta + tmp_min)
    f_dist = squareform(squareform(f_dist) + 1e-3)
    f_dict = {}
    for i in range(m):
        f_dict[i] = {}
        for j in range(m):
            f_dict[i][j] = f_dist[i][j]

    pos = {}
    for i in range(m):
        pos[i] = centers[i] - 0.5
    G_full = nx.complete_graph(m)
    # pos = nx.random_layout(G_full, seed=0)
    pos = kamada_kawai_layout(G_full, pos=pos, dist=f_dict, tol=1e-3)
    result = np.zeros((m, 2))
    for i in range(m):
        result[i] = np.array(pos[i])

    return result, father


def AssignQAP(grid_asses, labels, p=None, scale=20, grids=None, grid_asses_bf=None, selected=None, selected_bf=None,
              feature=None, confusion=None, with_score=False, best_w=None, small_labels=None, use_HV=False):
    random.seed(42)
    np.random.seed(42)

    start0 = time.time()
    new_grid_asses = grid_asses.copy()

    def get_layout_embedded_and_asses(grid_asses, square_len):
        grids = np.dstack(np.meshgrid(np.linspace(0, 1 - 1.0 / square_len, square_len),
                                      np.linspace(0, 1 - 1.0 / square_len, square_len))) \
            .reshape(-1, 2)
        tmp = grids[:, 0].copy()
        grids[:, 0] = grids[:, 1]
        grids[:, 1] = tmp
        ele_asses = np.zeros(grid_asses.shape[0], dtype='int')
        for i in range(grid_asses.shape[0]):
            ele_asses[grid_asses[i]] = i
        return grids[ele_asses], ele_asses

    N = grid_asses.shape[0]
    square_len = round(np.sqrt(N))
    maxLabel = labels.max() + 1
    # maxLabel = 1

    ele_asses = np.zeros(grid_asses.shape[0], dtype='int')
    for i in range(grid_asses.shape[0]):
        ele_asses[grid_asses[i]] = i
    for i in range(maxLabel):
        idx = ele_asses[np.arange(len(labels), dtype='int')[labels == i]]
        np.random.shuffle(grid_asses[idx])

    if grids is None:
        grid_embedded, ele_asses = get_layout_embedded_and_asses(grid_asses, square_len)
    else:
        ele_asses = np.zeros(N, dtype='int')
        for i in range(grid_asses.shape[0]):
            ele_asses[grid_asses[i]] = i
        grid_embedded = grids[ele_asses]

    if grid_asses_bf is not None:
        square_len_bf = round(np.sqrt(grid_asses_bf.shape[0]))
        grid_embedded_bf, _ = get_layout_embedded_and_asses(grid_asses_bf, square_len_bf)
        if_selected = np.ones(len(labels), dtype='int') * -1
        if_selected[selected] = np.arange(len(selected), dtype='int')

    full_D = cdist(grid_embedded, grid_embedded, "euclidean")
    # print("time dist", time.time() - start0)

    def get_P(lb, feature):
        perplexity = 30
        n = len(feature)
        n_neighbors = min(n - 1, int(3.0 * perplexity + 1))
        knn = NearestNeighbors(
            algorithm="auto",
            n_jobs=None,
            n_neighbors=n_neighbors,
            metric="euclidean",
        )
        knn.fit(feature)
        cost_matrix_nn = knn.kneighbors_graph(mode="distance")
        cost_matrix_nn.data **= 2
        P = _joint_probabilities_nn(cost_matrix_nn, perplexity, False)
        F = P.toarray()
        return F

    def solve_Assign(lb, info):
        start = time.time()

        F = info["F"]
        D = info["D"]
        addition_AB = info["addition_AB"]
        C = info["C"]
        P0 = info["P0"]
        solution0 = info["solution0"]
        n = info["n"]
        grids = info["grids"]
        sta_grids = info["sta_grids"]
        selected_list = info["selected_list"]
        father = info["father"]
        feature_now = info["feature"]
        confuse_id = info["confuse_id"]
        alpha = info["alpha"]
        idx = info["idx"]

        if "P_pre" not in info:
            if sta_grids is not None:
                delta = (sta_grids.max(axis=0) - sta_grids.min(axis=0))
                if delta[0] > 1e-8:
                    sta_grids[0] /= delta[0]
                if delta[1] > 1e-8:
                    sta_grids[1] /= delta[1]
                sta_grids *= (grids.max(axis=0) - grids.min(axis=0))
                sta_grids += grids.mean(axis=0) - sta_grids.mean(axis=0)
                # cost_matrix = np.zeros((n, n))
                # cost_matrix[selected_list, :] = np.power(cdist(sta_grids, grids, "euclidean"), 2)
                cost_matrix = np.power(cdist(sta_grids[father], grids, "euclidean"), 2)
                # cost_matrix += C[father] + C
                shuffle_col = np.arange(n, dtype='int')
                # np.random.shuffle(shuffle_col)
                solution2 = shuffle_col[lsa(cost_matrix[:, shuffle_col])[0]]
                P0 = np.eye(n)[solution2]
                solution0 = solution2
            else:
                # from .utils import get_anchor_grids
                # sta_grids, father = get_anchor_grids(feature_now)
                sta_grids, father = get_init_kmedoids(feature_now, C, grids)
                # print("achor time", time.time()-start)
                sta_grids /= (sta_grids.max(axis=0) - sta_grids.min(axis=0) + 1e-12)
                sta_grids *= (grids.max(axis=0) - grids.min(axis=0))
                sta_grids += grids.mean(axis=0) - sta_grids.mean(axis=0)
                # cost_matrix = np.zeros((n, n))
                # cost_matrix[selected_list, :] = np.power(cdist(sta_grids, grids, "euclidean"), 2)
                cost_matrix = np.power(cdist(sta_grids[father], grids, "euclidean"), 2)
                # cost_matrix += C[father] + C
                shuffle_col = np.arange(n, dtype='int')
                # np.random.shuffle(shuffle_col)
                solution1 = shuffle_col[lsa(cost_matrix[:, shuffle_col])[0]]
                P0 = np.eye(n)[solution1]
                solution0 = solution1

            info["P_pre"] = P0
            info["solution_pre"] = solution0

            c_norm = 0
            if confuse_id is not None:
                tmp = (F @ P0 @ D.T + F.T @ P0 @ D)
                mean1 = (tmp[confuse_id].max(axis=1) - tmp[confuse_id].min(axis=1)).mean()
                mean2 = (C[confuse_id].max(axis=1) - C[confuse_id].min(axis=1)).mean()
                # print(mean1)
                # print(mean2)

                if mean2 > 0:
                    c_norm = mean1 / mean2
                else:
                    c_norm = 0

            C = C * c_norm
            info["c_norm"] = c_norm
        else:
            P0 = info["P_pre"]
            solution0 = info["solution_pre"]
            C = C * info["c_norm"]

        old_C, old_D = C, D

        # print("pre time 1", lb, n, time.time() - start)

        D = D * alpha
        C = C * (1 - alpha)

        if small_labels is not None:
            now_small_labels = small_labels[idx]
            from collections import Counter
            counter = Counter(now_small_labels)
            if len(counter) > 1:
                centers = {}
                s_idx = {}
                for slabel in counter:
                    s_idx[slabel] = np.arange(n, dtype='int')[(now_small_labels == slabel)]
                    centers[slabel] = grids[solution0[s_idx[slabel]]].mean(axis=0)
                scenters = np.array(list(map(lambda x: centers[x], now_small_labels)))
                if not use_HV:
                    tmp_cost_matrix = np.power(cdist(scenters, grids, "euclidean"), 2)
                else:
                    axis = (grids.max(axis=0)-grids.min(axis=0)).argmax()
                    tmp_cost_matrix = np.power(cdist(scenters[:, [axis]], grids[:, [axis]], "euclidean"), 2)
                    tmp_cost_matrix += np.power(cdist(scenters, grids, "euclidean"), 2)*1
                tmp_cost_matrix *= 10
                tmp_cost_matrix += 0.001 * np.power(cdist(grids[solution0], grids, "euclidean"), 2)
                solution0 = lsa(tmp_cost_matrix)[0]
                P0 = np.eye(n)[solution0]
                for slabel in counter:
                    for slabel2 in counter:
                        if slabel != slabel2:
                            C[np.ix_(s_idx[slabel], solution0[s_idx[slabel2]])] += 10000
                            # F[np.ix_(s_idx[slabel], s_idx[slabel2])] = 0
                            if len(addition_AB) > 0:
                                addition_AB[0][0][np.ix_(s_idx[slabel], s_idx[slabel2])] = 0
                                addition_AB[1][0][np.ix_(s_idx[slabel], s_idx[slabel2])] = 0

        # print("pre time 2", lb, n, time.time() - start)

        sparse = False
        if (len(addition_AB) > 0) and (addition_AB[0][0].sum() + addition_AB[1][0].sum() < 4 * (n / 20) ** 2):
            sparse = True

        maxit = 20
        if info["N"] >= 3000:
            maxit = 15
        ans2 = quadratic_assignment_faq(F, D, addition_AB, C, P0=P0, addition_sparse=sparse, maxiter=maxit, tol=0)
        solution2 = ans2["col_ind"]
        # print(ans2)
        # if len(addition_AB)>0:
        #     tmp = addition_AB[0][0] * addition_AB[0][1][solution2][:, solution2] + addition_AB[1][0] * addition_AB[1][1][solution2][:, solution2]
        #     print("xy", tmp.sum())
        # solution2 = solution0

        info["score"] = np.sum(F * old_D[solution2][:, solution2])
        info["cscore"] = np.sum(old_C[np.arange(n), solution2])

        # print("qap time", lb, n, time.time() - start)

        # print(ans2)
        return solution2

    labels_idx = {}
    for i in range(maxLabel):
        labels_idx[i] = (labels == i)
        # print(labels_idx[i].sum())
        labels_idx[i] = np.arange(len(labels), dtype='int')[labels_idx[i]]

    start0 = time.time()
    time1 = 0
    time2 = 0
    time3 = 0
    time4 = 0
    FD_dict = {}

    if confusion is not None:
        if_confuse = confusion['if_confuse']
        confuse_class = confusion['confuse_class']
        conf = confusion['conf']
        confuse_dist = np.zeros((maxLabel, N))
        for lb in range(maxLabel):
            confuse_dist[lb] = full_D[labels_idx[lb]].min(axis=0)

        # print(time.time() - start0)

        near = np.zeros((maxLabel, maxLabel), dtype='bool')

        confuse_dist2 = np.zeros((maxLabel, N))
        for lb in range(maxLabel):
            idx = labels_idx[lb]
            tmp_D = full_D[np.ix_(idx, idx)]
            for lb2 in range(maxLabel):
                tmp = confuse_dist[lb2][idx]
                sort_id = np.argsort(tmp)
                tiny = 1e-8
                eps = 1 / square_len
                min_dist = tmp[sort_id[0]]
                if min_dist > eps + tiny:
                    min_dist += eps
                new_id = np.arange(len(idx), dtype='int')[tmp <= min_dist + tiny]
                # confuse_dist2[lb2][idx] = full_D[np.ix_(idx, idx[new_id])].min(axis=1)
                confuse_dist2[lb2][idx] = tmp_D[new_id].min(axis=0)
                if min_dist <= 2*eps + tiny:
                    near[lb][lb2] = True

    # if p is None:
    #     p = np.zeros((N, N))
    #     pool = ThreadPool(maxLabel)
    #     work_list = []
    #     for label in range(maxLabel):
    #         idx = labels_idx[label]
    #         now_feature = feature[idx]
    #         work_list.append((label, now_feature))
    #     result = pool.starmap_async(get_P, work_list)
    #     pool.close()
    #     pool.join()
    #     result = result.get()
    #     for label in range(maxLabel):
    #         idx = labels_idx[label]
    #         p[np.ix_(idx, idx)] = result[label]

    time0 = time.time() - start0


    if confusion is not None and maxLabel > 1:
        near_conf = 0
        full_conf = 0
        confuse_id = np.arange(len(if_confuse))[if_confuse]
        if len(confuse_id) > 0:
            t_lbs = confuse_class[confuse_id, 1]
            if maxLabel > 2:
                t_lbs2 = confuse_class[confuse_id, 2]
            else:
                t_lbs2 = t_lbs
            for i, id in enumerate(confuse_id):
                label = labels[id]
                t_lb = t_lbs[i]
                t_lb2 = t_lbs2[i]
                if t_lb == label: t_lb = confuse_class[id][0]
                if t_lb2 == label: t_lb2 = confuse_class[id][0]
                full_conf += conf[id][t_lb] + conf[id][t_lb2]
                if near[label][t_lb]:
                    near_conf += conf[id][t_lb]
                if near[label][t_lb2]:
                    near_conf += conf[id][t_lb2]
        # print("conf", near_conf, full_conf, near_conf/(full_conf+1e-12))
        # full_conf = N
        use_full = False
        if full_conf<N/100 or (grid_asses_bf is None and full_conf<N*0.03):
            use_full = True
        # print(use_full)


    for label in range(maxLabel):

        start1 = time.time()

        # idx = np.arange(len(labels), dtype='int')[labels_idx[label]]
        idx = labels_idx[label].copy()

        n = len(idx)

        if n == 1:
            F = np.array([[0]])
        elif p is not None:
            F = p[np.ix_(idx, idx)]
        else:
            perplexity = 30
            n_neighbors = min(n - 1, int(3.0 * perplexity + 1))

            # knn = NearestNeighbors(
            #     algorithm="auto",
            #     n_jobs=None,
            #     n_neighbors=n_neighbors,
            #     metric="euclidean",
            # )
            # knn.fit(feature[idx])
            # cost_matrix_nn = knn.kneighbors_graph(mode="distance")

            now_dist = cdist(feature[idx], feature[idx], 'eu')
            knn_matrix = np.argsort(now_dist, axis=1)[:, 1:n_neighbors + 1]
            knn_distances = np.take_along_axis(now_dist, knn_matrix, axis=1)
            rows = np.repeat(np.arange(n), n_neighbors)
            cols = knn_matrix.flatten()
            data = knn_distances.flatten()
            cost_matrix_nn = csr_matrix((data, (rows, cols)), shape=(n, n))

            cost_matrix_nn.data **= 2
            P = _joint_probabilities_nn(cost_matrix_nn, perplexity, False)
            F = P.toarray()

        D = cdist(grid_embedded[idx], grid_embedded[idx], "euclidean")
        D = np.power(D, 2) * scale * scale
        D = np.log(1 + D)

        time1 += time.time() - start1
        sta_grids = None
        selected_list = None
        father = None

        if grid_asses_bf is not None:

            start2 = time.time()

            sta_grids = []
            selected_list = []

            Fx = np.zeros((n, n))
            Dx = np.zeros((n, n))
            Fy = np.zeros((n, n))
            Dy = np.zeros((n, n))

            time2 += time.time() - start2

            selected_index = np.ones(len(selected), dtype='int') * -1
            for i in range(len(idx)):
                ii = idx[i]
                if if_selected[ii] >= 0:
                    selected_index[if_selected[ii]] = i
                    sta_grids.append(grid_embedded_bf[selected_bf[if_selected[ii]]])
                    selected_list.append(i)
            sta_grids = np.array(sta_grids)
            selected_list = np.array(selected_list)

            if len(selected_list) > 0:

                start3 = time.time()

                for i in range(len(selected)):
                    if selected_index[i] >= 0:
                        id1 = selected_index[i]
                    else:
                        continue
                    for j in range(len(selected)):
                        if selected_index[j] >= 0:
                            id2 = selected_index[j]
                        else:
                            continue
                        if grid_embedded_bf[selected_bf[i]][0] < grid_embedded_bf[selected_bf[j]][0]:
                            Fx[id1][id2] = 1
                        if grid_embedded_bf[selected_bf[i]][1] < grid_embedded_bf[selected_bf[j]][1]:
                            Fy[id1][id2] = 1

                time3 += time.time() - start3

                start4 = time.time()

                # min_gap = 1 / np.sqrt(len(labels))
                for i in range(len(idx)):
                    td_x = grid_embedded[idx, 0] <= grid_embedded[idx[i]][0]
                    Dx[i][td_x] = scale * scale
                    td_y = grid_embedded[idx, 1] <= grid_embedded[idx[i]][1]
                    Dy[i][td_y] = scale * scale

                addition_AB = [(Fx, Dx), (Fy, Dy)]
                # addition_AB = []

                father_dist = F[:, selected_list]
                father_dist[selected_list, np.arange(len(selected_list), dtype='int')] = 1
                father = np.argmax(father_dist, axis=1)

                time4 += time.time() - start4

            else:
                sta_grids = None
                selected_list = None
                father = None
                addition_AB = []
        else:
            addition_AB = []

        matrix0 = np.zeros((n, n))
        C = matrix0
        confuse_id = None
        if confusion is not None and maxLabel > 1:

            confuse_id = np.arange(n)[if_confuse[idx]]
            if len(confuse_id) > 0:
                # print(label, len(confuse_id))

                t_lbs = confuse_class[idx[confuse_id], 1]
                if maxLabel > 2:
                    t_lbs2 = confuse_class[idx[confuse_id], 2]
                else:
                    t_lbs2 = t_lbs

                for i, id in enumerate(confuse_id):
                    t_lb = t_lbs[i]
                    t_lb2 = t_lbs2[i]
                    if t_lb == label: t_lb = confuse_class[idx[id]][0]
                    if t_lb2 == label: t_lb2 = confuse_class[idx[id]][0]
                    dist1 = dist2 = 0
                    # if True:
                    if near[label][t_lb] or use_full:
                        dist1 = confuse_dist2[t_lb][idx]
                    if near[label][t_lb2] or use_full:
                        dist2 = confuse_dist2[t_lb2][idx]
                    # C[id] = dist1 * conf[idx[id], t_lb] + dist2 * conf[idx[id], t_lb2]
                    C[id] = np.power(dist1, 2) * conf[idx[id], t_lb] + np.power(dist2, 2) * conf[idx[id], t_lb2]
                    # C[id] = np.power(dist1-dist1.min(), 2)*conf[idx[id], t_lb]+np.power(dist2-dist2.min(), 2)*conf[idx[id], t_lb2]
                    # C[id] /= (conf[idx[id], t_lb] + conf[idx[id], t_lb2])

                    if len(addition_AB) > 0 and len(confuse_id)/n < 1/4:
                        addition_AB[0][0][:, id] = 0
                        addition_AB[0][0][id, :] = 0
                        addition_AB[1][0][:, id] = 0
                        addition_AB[1][0][id, :] = 0
            else:
                confuse_id = None

        solution0 = np.arange(n, dtype='int')
        # np.random.seed(100)
        # np.random.shuffle(solution0)
        P0 = np.eye(n)[solution0]

        FD_dict[label] = {"F": F, "D": D, "addition_AB": addition_AB, "C": C, "P0": P0, "solution0": solution0, "n": n, "N": N,
                          "idx": idx, "grids": grid_embedded[idx], "sta_grids": sta_grids,
                          "selected_list": selected_list, "father": father,
                          "feature": feature[idx], "confuse_id": confuse_id}

    # print("pre time", time.time() - start0, time0, time1, time2, time3, time4)

    def getQAP(alpha=0.5):
        for label in range(maxLabel):
            FD_dict[label]["alpha"] = alpha

        pool = ThreadPool(min(32, maxLabel))
        work_list = []
        for label in range(maxLabel):
            work_list.append((label, FD_dict[label]))
        result = pool.starmap_async(solve_Assign, work_list)
        pool.close()
        pool.join()
        result = result.get()

        # result = []
        # for label in range(maxLabel):
        # # for label in [4]:
        #     result.append(solve_Assign(label, FD_dict[label]))

        avg_score, avg_cscore = 0, 0
        for label in range(maxLabel):
            avg_score += FD_dict[label]["score"]
            avg_cscore += FD_dict[label]["cscore"]
        avg_score /= maxLabel
        avg_cscore /= maxLabel
        return result, avg_score, avg_cscore

    # ---------------use fixed weight parameter------------------
    if best_w is None:
        best_w = 0.15
        if confusion is not None and maxLabel > 1 and near_conf > N/10:
            best_w = 0.65
    # print("best w", best_w)
    result, result_score, result_cscore = getQAP(best_w)
    # -----------------------------------------------------------

    # # -------------use multi-task weight parameter------------------
    # goal = 2
    # pro_result, best_score, worst_cscore = getQAP(1)
    # conf_result, worst_score, best_cscore = getQAP(0.001)
    # left, right = 0, 1
    # best_solution = None
    # best_d = 0
    # for i in range(3):
    #     mid = (left + right) / 2
    #     new_result, new_score, new_cscore = getQAP(mid)
    #     d1 = (new_score - worst_score) / (best_score - worst_score)
    #     d2 = (new_cscore - worst_cscore) / (best_cscore - worst_cscore)
    #     if goal*d1 <= d2:
    #         left = mid
    #     else:
    #         right = mid
    #
    #     if best_solution is None or abs(goal * d1 - d2) / (d1 + d2) < best_d:
    #         best_d = abs(goal * d1 - d2) / (d1 + d2)
    #         best_solution = new_result.copy()
    #         result_score, result_cscore = new_score, new_cscore
    #
    # #     print("multitask", mid, d1, d2)
    #
    # result = best_solution
    # -----------------------------------------------------------

    for label in range(maxLabel):
        idx = np.arange(len(labels))[labels == label]
        solution2 = result[label]
        new_grid_asses[ele_asses[idx][solution2]] = idx

    new_grid_embedded, _ = get_layout_embedded_and_asses(new_grid_asses, square_len)

    if with_score:
        return new_grid_asses, new_grid_embedded, (result_score, result_cscore)
    return new_grid_asses, new_grid_embedded
