import os
import math
import random
import numpy as np
from scipy.spatial.distance import cdist
import threading
from collections import Counter

from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry import Point

from .gridOptimizer_clean import gridOptimizer
import application.grid.gridlayoutOpt as gridlayoutOpt
from .PowerDiagram import getPowerDiagramGrids
from .PowerDiagram import CentersAdjust
from .PowerDiagram import rotateEmbedded
from .colors import ColorScatter
from .AssignQAP import AssignQAP
from .utils import kamada_kawai_layout, get_kamada_kawai_costfn_stability

from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, DBSCAN, SpectralBiclustering
from sklearn_extra.cluster import KMedoids
from application.utils.pickle import *
import time
from sklearn.manifold import TSNE
import networkx as nx
from scipy.spatial.distance import squareform

import copy

case_id = 3
if case_id == 1:
    case_str = ""
else:
    case_str = str(case_id)

neighbor_type = "k"
max_e = 0.1


def get_layout_embedded(grid_asses, square_len):
    grids = np.dstack(np.meshgrid(np.linspace(0, 1 - 1.0 / square_len, square_len),
                                    np.linspace(0, 1 - 1.0 / square_len, square_len))) \
        .reshape(-1, 2)
    tmp = grids[:, 0].copy()
    grids[:, 0] = grids[:, 1]
    grids[:, 1] = tmp
    ele_asses = np.zeros(grid_asses.shape[0], dtype='int')
    for i in range(grid_asses.shape[0]):
        ele_asses[grid_asses[i]] = i
    return grids[ele_asses]

class GridLayout(object):
    def __init__(self, Ctrler=None):
        super().__init__()
        self.optimizer = gridOptimizer()
        self.cache_root = './cache'
        if not os.path.exists(self.cache_root):
            os.makedirs(self.cache_root)
        self.Ctrler = Ctrler

    def set_cache_path(self, dataset):
        self.cache_path = os.path.join(self.cache_root, dataset)
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

    def getInit(self, constrain_X, constrain_y, X):
        # 1. find closest point in constrain_X for each point in other_X
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(constrain_X)
        # 2. get the closest index
        indices = neigh.kneighbors(X, return_distance=False).reshape(-1)
        return constrain_y[indices]

    def get_knn_embedding(self, X, labels, selected, selected_embed, k=5, mode=0, target=0.8):
        # X: features
        # selected: selected points
        # selected_embed: selected points' embedding
        # k: knn
        # mode: 0: knn in extend, 1: knn in selected
        # target: target select points ratio
        # return constraint_x, constraint_y, constraint_labels, extend_selected
        dist = pairwise_distances(X, n_jobs=-1)
        refer_selected = selected[:]
        if k == -1:
            k = len(selected)
        else:
            k = min(k, len(selected))
        options = []
        for i in range(X.shape[0]):
            if i not in selected:
                options.append(i)
        res = selected[:]
        res_embed = selected_embed.copy()
        while len(res) < target * X.shape[0]:
            choice = random.choice(options)
            options.remove(choice)
            knns = np.argsort(dist[choice, refer_selected])[:k]
            # Weighted by distance
            weights = 1 / dist[choice, refer_selected][knns] ** 2
            weights = weights / np.sum(weights)
            res_embed = np.vstack([res_embed, np.sum(res_embed[knns] * weights.reshape(-1, 1), axis=0)])
            res.append(choice)
            if mode == 0:
                refer_selected.append(choice)
        return X[res], res_embed, labels[res], res

    def get_centers(self, X_embedded, labels):
        maxLabel = labels.max()+1
        centers = np.zeros((maxLabel, 2))
        for p in range(maxLabel):
            idx = (labels==p)
            if idx.sum()>0:
                centers[p] = X_embedded[idx].mean(axis=0)
        return centers

    def get_planar(self, f_dist, maxLabel):
        tmp_dist = squareform(f_dist)
        sort_id = np.argsort(tmp_dist)
        G = nx.empty_graph(maxLabel)
        cnt = 0
        map = {}
        for i in range(maxLabel):
            for j in range(i+1, maxLabel):
                map[cnt] = (i, j)
                cnt += 1
        edge_cnt = 0
        for id in sort_id:
            i, j = map[id]
            G.add_edge(i, j, weight=f_dist[i][j])
            if not nx.is_planar(G):
                G.remove_edge(i, j)
                # if nx.is_connected(G):
                #     break
            else:
                edge_cnt += 1
        print("planer edge", edge_cnt)
        return G

    def get_FD_layout_centers_stability(self, f_dict, old_pos, keep, goal=2, planer_G=None):

        maxLabel = len(old_pos)

        # import matplotlib.pyplot as plt
        # plt.switch_backend('agg')
        # for i in range(maxLabel):
        #     plt.scatter(old_pos[i][1], old_pos[i][0], color=plt.cm.tab20(i))
        # plt.savefig("ft0.png")
        # plt.clf()

        stability = {"old_pos": old_pos, "keep": keep}
        G_full = nx.complete_graph(maxLabel)
        pos_dict = {}
        for i in range(maxLabel):
            pos_dict[i] = old_pos[i]
        tmp_pos = pos_dict
        if planer_G is not None:
            tmp_pos = kamada_kawai_layout(planer_G, pos=pos_dict, options={"maxiter": 20})
        kk_pos_dict = kamada_kawai_layout(G_full, pos=tmp_pos, dist=f_dict, options={"maxiter": 20})
        kk_pos = np.zeros((maxLabel, 2))
        for i in range(maxLabel):
            kk_pos[i] = np.array(kk_pos_dict[i])

        # kk_scale = (kk_pos.max(axis=0)-kk_pos.min(axis=0)).sum()
        # old_scale = (old_pos.max(axis=0)-old_pos.min(axis=0)).sum()
        kk_scale = kk_pos.std(axis=0).sum()
        old_scale = old_pos.std(axis=0).sum()
        print("scale", old_scale, kk_scale)
        if old_scale > 0 and kk_scale > 0:
            old_pos = old_pos*kk_scale/old_scale      # scale normalize
            stability["old_pos"] = old_pos


        # import matplotlib.pyplot as plt
        # plt.switch_backend('agg')
        # for i in range(maxLabel):
        #     plt.scatter(kk_pos[i][1], kk_pos[i][0], color=plt.cm.tab20(i))
        # plt.savefig("ft_kk.png")
        # plt.clf()

        best_kk, worst_sta = get_kamada_kawai_costfn_stability(G_full, dist=f_dict, pos=kk_pos, old_pos=old_pos, keep=keep)
        worst_kk, best_sta = get_kamada_kawai_costfn_stability(G_full, dist=f_dict, pos=old_pos, old_pos=old_pos, keep=keep)

        norm_ratio = 1
        if worst_sta-best_sta > 1e-12:
            norm_ratio = (worst_kk-best_kk)/(worst_sta-best_sta)
        else:
            return kk_pos

        if worst_kk-best_kk <= 1e-12:
            return old_pos

        stability["norm_ratio"] = norm_ratio

        left, right = 0, 1
        best_pos = None
        best_d = 0
        for i in range(2):
            mid = (left + right) / 2
            stability["sta_alpha"] = mid
            tmp_pos = pos_dict
            if planer_G is not None:
                tmp_pos = kamada_kawai_layout(planer_G, pos=pos_dict, options={"maxiter": 20}, stability=stability)
            new_pos_dict = kamada_kawai_layout(G_full, pos=tmp_pos, dist=f_dict, options={"maxiter": 20}, stability=stability)
            new_pos = np.zeros((maxLabel, 2))
            for i in range(maxLabel):
                new_pos[i] = np.array(new_pos_dict[i])
            new_kk, new_sta = get_kamada_kawai_costfn_stability(G_full, dist=f_dict, pos=new_pos, old_pos=old_pos, keep=keep)

            d1 = (new_kk - worst_kk) / (best_kk - worst_kk)
            d2 = (new_sta - worst_sta) / (best_sta - worst_sta)
            if goal*d1 <= d2:
                left = mid
            else:
                right = mid

            if best_pos is None or abs(goal * d1 - d2) / (d1 + d2) < best_d:
                best_d = abs(goal * d1 - d2) / (d1 + d2)
                best_pos = new_pos.copy()

            print("FD multitask", mid, d1, d2)

        # import matplotlib.pyplot as plt
        # plt.switch_backend('agg')
        # for i in range(maxLabel):
        #     plt.scatter(best_pos[i][1], best_pos[i][0], color=plt.cm.tab20(i))
        # plt.savefig("ft1.png")
        # plt.clf()

        return best_pos


    def get_FD_layout_centers(self, X_feature, top_partition, labels, info_before, conf=None, alpha=0.5):
        from scipy.stats import entropy
        N = len(labels)
        maxLabel = labels.max()+1

        if maxLabel == 1:
            return np.array([[0.5, 0.5]]), np.array([[False]])

        feature_mean = []
        for lb in range(maxLabel):
            feature_mean.append(X_feature[labels==lb].mean(axis=0))
        feature_mean = np.array(feature_mean)

        top_count = np.zeros(top_partition.max()+1)
        for lb in range(maxLabel):
            top_count[top_partition[lb]] += (labels==lb).sum()

        f_dist = cdist(feature_mean, feature_mean, "euclidean")

        if maxLabel > 3:
            f_dist = squareform(f_dist)
            if f_dist.max() > f_dist.min() + 1e-6:
                f_dist = f_dist-f_dist.min()
            f_dist = squareform(f_dist)
        f_dist /= f_dist.max()

        if conf is not None:
            labels_idx = {}
            for i in range(maxLabel):
                labels_idx[i] = (labels == i)
            c_list = []
            for i in range(maxLabel):
                c_list.append(conf[labels_idx[i]].mean(axis=0))
            c_list = np.array(c_list)
            for i in range(maxLabel):
                c_list[i] = np.maximum(c_list[i], 1e-6)
                c_list[i] /= c_list[i].sum()
            c_dist = np.zeros((maxLabel, maxLabel))
            for i in range(maxLabel):
                for j in range(maxLabel):
                    c_dist[i][j] = entropy(c_list[i], c_list[j])
            c_dist = (c_dist+c_dist.T)/2

        if conf is not None:
            if maxLabel > 3:
                c_dist = squareform(c_dist)
                if c_dist.max() > c_dist.min() + 1e-6:
                    c_dist = c_dist - c_dist.min()
                c_dist = squareform(c_dist)
            c_dist /= c_dist.max()

        if maxLabel > 3:
            tmp_min = 1/np.sqrt(2)/(np.sqrt(maxLabel)-1)
            tmp_delta = 1-tmp_min

            f_dist = squareform(squareform(f_dist)*tmp_delta+tmp_min)
            if conf is not None:
                c_dist = squareform(squareform(c_dist)*tmp_delta+tmp_min)

        if conf is not None:
            for i in range(maxLabel):
                for j in range(maxLabel):
                    f_dist[i][j] = alpha*f_dist[i][j]+(1-alpha)*c_dist[i][j]

        init = None
        caled = np.zeros(maxLabel, dtype='bool')

        if info_before is not None:
            N2 = info_before['grid_asses'].shape[0]
            tmp_embedded2 = get_layout_embedded(info_before['grid_asses'], round(np.sqrt(N2)))
            tmp_embedded2[info_before['selected_bf']] -= tmp_embedded2[info_before['selected_bf']].min(axis=0)-1/2/np.sqrt(N2)
            tmp_embedded2[info_before['selected_bf']] /= tmp_embedded2[info_before['selected_bf']].max(axis=0)+1/2/np.sqrt(N2)

            centers = np.zeros((maxLabel, 2))
            for p in range(maxLabel):
                idx = (labels[info_before['selected']]==p)
                # print(p, tmp_embedded2[info_before['selected_bf']][idx])
                if idx.sum()>0:
                    centers[p] = tmp_embedded2[info_before['selected_bf']][idx].mean(axis=0)
                    caled[p] = True

            caled_list = np.arange(maxLabel, dtype='int')[caled]
            # print("caled", caled)

            np.random.seed(0)
            if len(caled_list)>0:
                # neigh = NearestNeighbors(n_neighbors=min(len(caled_list), 2))
                # neigh.fit(feature_mean[caled_list])
                # for p in range(maxLabel):
                #     idx = (labels[info_before['selected']]==p)
                #     if idx.sum()==0:
                #         indices = neigh.kneighbors(feature_mean[p].reshape(1, -1), return_distance=False).reshape(-1)
                #         centers[p] = (centers[caled_list[indices[0]]]+centers[caled_list[indices[1]]])/2
                dist_matrix = cdist(feature_mean, feature_mean[caled_list])
                for p in range(maxLabel):
                    if not caled[p]:
                        indices = dist_matrix[p].argsort()
                        if len(indices)>1:
                            centers[p] = (centers[caled_list[indices[0]]] + centers[caled_list[indices[1]]]) / 2 + np.random.normal(0, 0.05, 2)
                        else:
                            centers[p] = centers[caled_list[indices[0]]] + np.random.normal(0, 0.1, 2)
                init = centers
                init += np.random.normal(0, 0.001, (maxLabel, 2))
            else:
                init = None

        # print("f_dist", f_dist)

        f_dict = {}
        for i in range(maxLabel):
            f_dict[i] = {}
            for j in range(maxLabel):
                f_dict[i][j] = f_dist[i][j]

        G = self.get_planar(f_dist, maxLabel)

        edge_matrix = np.zeros((maxLabel, maxLabel), dtype='bool')
        for e in G.edges():
            edge_matrix[e[0]][e[1]] = edge_matrix[e[1]][e[0]] = True

        if init is None:
            # pos = nx.random_layout(G)
            # pos = None
            pos = nx.kamada_kawai_layout(G)
        else:
            # pos = {}
            # for i in range(maxLabel):
            #     pos[i] = init[i]-0.5
            # print("pos", pos)
            # import matplotlib.pyplot as plt
            # plt.switch_backend('agg')
            # for i in range(maxLabel):
            #     plt.scatter(init[i][1], init[i][0], color=plt.cm.tab20(i))
            # plt.savefig("ft0.png")
            # plt.clf()

            # o_dist = cdist(init, init, "euclidean")
            # o_dist /= o_dist.max()
            #
            # for p in range(top_partition.max()+1):
            #     l_id = (top_partition == p)
            #     if f_dist[np.ix_(l_id, l_id)].max() > 0:
            #         f_dist[np.ix_(l_id, l_id)] /= f_dist[np.ix_(l_id, l_id)].max()
            #
            # G = nx.empty_graph(maxLabel)
            # for i in range(maxLabel):
            #     for j in range(i+1, maxLabel):
            #         if top_partition[i] == top_partition[j]:
            #             # G.add_edge(i, j, weight=f_dist[i][j])
            #             # G.add_edge(i, j, weight=f_dist[i][j]*np.sqrt(top_count[top_partition[i]]/N))
            #             if caled[i] and caled[j]:
            #                 G.add_edge(i, j, weight=(f_dist[i][j]*np.sqrt(top_count[top_partition[i]]/N)+o_dist[i][j])/2)
            #             else:
            #                 G.add_edge(i, j, weight=f_dist[i][j]*np.sqrt(top_count[top_partition[i]]/N))
            #         # else:
            #         #     # G.add_edge(i, j, weight=f_dist[i][j]*2)
            #         #     G.add_edge(i, j, weight=f_dist[i][j])
            # # pos = nx.kamada_kawai_layout(G, pos=pos)
            # pos = kamada_kawai_layout(G, pos=pos, options={"maxiter": 20})
            # result = np.zeros((maxLabel, 2))
            # for i in range(maxLabel):
            #     result[i] = np.array(pos[i])

            result = self.get_FD_layout_centers_stability(f_dict, init-init.mean(axis=0), caled, planer_G=G)

            result -= result.min(axis=0)
            result /= result.max(axis=0)
            # import matplotlib.pyplot as plt
            # plt.switch_backend('agg')
            # for i in range(maxLabel):
            #     plt.scatter(result[i][1], result[i][0], color=plt.cm.tab20(i))
            # plt.savefig("ft1.png")
            # plt.clf()
            return result, edge_matrix


        # np.savez("KD.npz", f_dist=f_dist, init=init)

        G_full = nx.complete_graph(maxLabel)
        pos = nx.kamada_kawai_layout(G_full, pos=pos, dist=f_dict)
        result = np.zeros((maxLabel, 2))
        for i in range(maxLabel):
            result[i] = np.array(pos[i])

        if info_before is None:
            best_result = None
            best_a = None
            for rid in range(16):
                r = rid * 2 * math.pi / 16
                tmp_result = np.zeros((maxLabel, 2))
                for lb in range(maxLabel):
                    tmp_x = result[lb][0]
                    tmp_y = result[lb][1]
                    tmp_result[lb][0] = tmp_x * math.cos(r) - tmp_y * math.sin(r)
                    tmp_result[lb][1] = tmp_x * math.sin(r) + tmp_y * math.cos(r)
                bound = tmp_result.max(axis=0)-tmp_result.min(axis=0)
                if best_a is None or bound[0]*bound[1] < best_a:
                    best_a = bound[0]*bound[1]
                    best_result = tmp_result
            result = best_result

        result -= result.min(axis=0)
        result /= result.max(axis=0)
        # import matplotlib.pyplot as plt
        # plt.switch_backend('agg')
        # for i in range(maxLabel):
        #     plt.scatter(result[i][1], result[i][0], color=plt.cm.tab20(i))
        # plt.savefig("ft1.png")
        # plt.clf()

        return result, edge_matrix

    def adjust_FD_layout_centers(self, top_partition, labels, partition_centers, top_labels):
        maxTop = top_partition.max() + 1
        maxLabel = len(top_partition)
        belong = np.zeros(maxLabel, dtype='int')
        for lb in range(maxLabel):
            tmp = Counter(top_labels[labels == lb])
            belong[lb] = max(tmp, key=lambda x: tmp[x])

        for p in range(maxTop):
            tmp_idx = np.arange(maxLabel, dtype='int')[top_partition==p]

            belong_list = {}
            for lb in tmp_idx:
                if belong[lb] not in belong_list:
                    belong_list[belong[lb]] = []
                belong_list[belong[lb]].append(lb)

            main_top = max(belong_list, key=lambda x: len(belong_list[x]))
            main_bound_min = partition_centers[belong_list[main_top]].min(axis=0)
            main_bound_max = partition_centers[belong_list[main_top]].max(axis=0)
            main_bound_center = partition_centers[belong_list[main_top]].mean(axis=0)

            for top in belong_list:
                if top == main_top:
                    continue
                bound_min = partition_centers[belong_list[top]].min(axis=0)
                bound_max = partition_centers[belong_list[top]].max(axis=0)
                bound_center = partition_centers[belong_list[top]].mean(axis=0)
                move_d = bound_center-main_bound_center
                move1 = (main_bound_max+(bound_center-bound_min)/2-bound_center)/(move_d+1e-12)
                move2 = (main_bound_min+(bound_center-bound_max)/2-bound_center)/(move_d+1e-12)
                move = min(max(move1[0], move2[0], 0), max(move1[1], move2[1], 0))*1.2
                # move = move + np.abs((bound_max-bound_min)/(move_d+1e-12)).min()
                for lb in belong_list[top]:
                    partition_centers[lb] = (partition_centers[lb]-bound_center)/2 + bound_center + move*move_d

        return partition_centers


    def get_FD_layout_top(self, top_partition, labels, partition_centers):
        maxTop = top_partition.max()+1
        maxLabel = len(top_partition)
        count = np.zeros(maxLabel)
        for lb in range(maxLabel):
            count[lb] = (labels==lb).sum()
        centers = np.zeros((maxTop, 2))
        for p in range(maxTop):
            tmp_idx = (top_partition==p)
            tmp_centers = partition_centers[tmp_idx]
            tmp_count = count[tmp_idx].reshape((1, -1))
            centers[p] = (tmp_count@tmp_centers)/tmp_count.sum()

        # import matplotlib.pyplot as plt
        # plt.switch_backend('agg')
        # for i in range(maxTop):
        #     plt.scatter(centers[i][1], centers[i][0], color=plt.cm.tab20(i))
        # plt.savefig("ft2.png")
        # plt.clf()

        return centers

    def get_label_partition_feature(self, X_feature, labels, top_labels, filter_labels, cluster_ratio=0.2, original_top_labels=None):
        # 输出label与partition的对应关系，以及每个partition对应的重心位置（顶层使用tsne计算，非顶层使用上层父亲元素的layout位置计算）
        start = time.time()
        unique_labels = np.unique(labels)
        assert np.array_equal(unique_labels, np.arange(unique_labels.shape[0])) # labels should be normalized

        label_embeds = {}
        label_lists = {}
        filtered_labels = []

        map2top = {}

        for i in range(X_feature.shape[0]):
            if labels[i] not in label_embeds:
                label_embeds[labels[i]] = []
            label_embeds[labels[i]].append(X_feature[i])
            if top_labels[i] not in label_lists:
                label_lists[top_labels[i]] = []

            map2top[labels[i]] = top_labels[i]

            if labels[i] in filter_labels[1]:
                if labels[i] not in filtered_labels:
                    filtered_labels.append(labels[i])
            else:
                if labels[i] not in label_lists[top_labels[i]]:
                    label_lists[top_labels[i]].append(labels[i])

        # print("time label partition 0", time.time()-start)

        # print(filter_labels)
        # print(label_lists)
        # print(filtered_labels)

        top_partition = []
        partitions = {}
        cur_partition = 0
        cluster_size = int(X_feature.shape[0] * cluster_ratio)
        tcnt = 0
        for tlabel in label_lists:
            if tlabel in filter_labels[0]:
                continue
            tcnt += 1
        for tlabel in label_lists:
            if tlabel in filter_labels[0]:
                filtered_labels.extend(label_lists[tlabel])
                continue
            if len(label_lists[tlabel])==0:
                continue

            for i in range(len(label_lists[tlabel])):
                partitions[label_lists[tlabel][i]] = i + cur_partition

            cur_partition += len(label_lists[tlabel])

            for i in range(len(label_lists[tlabel])):
                top_partition.append(tlabel)

        top_partition = np.array(top_partition)

        from collections import Counter
        print(top_partition)
        print(Counter(labels))

        print("time label partition 1", time.time()-start)
        print(filtered_labels)

        for label in filtered_labels:
            partitions[label] = -1
        label_partition = np.zeros((unique_labels.shape[0], ))
        for i in range(unique_labels.shape[0]):
            label_partition[i] = partitions[i]
        label_partition = label_partition.astype(np.int32)
        partition_num = cur_partition
        partition_centers = np.zeros((partition_num, X_feature.shape[1]))

        # print(Counter(label_partition[labels]))

        for i in range(partition_num):
            partition_centers[i] = (X_feature[(label_partition[labels] == i)]).sum(axis=0)/(X_feature[(label_partition[labels] == i)]).shape[0]

        # print(partition_num)

        if original_top_labels is not None:
            original_label_lists = {}
            original_map2top = {}
            for i in range(X_feature.shape[0]):
                if original_top_labels[i] not in original_label_lists:
                    original_label_lists[original_top_labels[i]] = []
                original_map2top[labels[i]] = original_top_labels[i]

                if labels[i] not in filtered_labels:
                    if labels[i] not in original_label_lists[original_top_labels[i]]:
                        original_label_lists[original_top_labels[i]].append(labels[i])

        if len(filtered_labels) > 0:
            # set filter labels to nearest partition

            # neigh = NearestNeighbors(n_neighbors=1)
            # neigh.fit(partition_centers)

            filtered_center = []
            for label in filtered_labels:
                label_center = np.mean(np.array(label_embeds[label]), axis=0)
                filtered_center.append(label_center)
            filtered_center = np.array(filtered_center)
            dist_matrix = cdist(filtered_center, partition_centers)

            order = 0
            for label in filtered_labels:
                # label_center = np.mean(np.array(label_embeds[label]), axis=0)
                tlabel = map2top[label]
                if original_top_labels is not None and (len(original_label_lists[original_map2top[label]]) > 0):
                    tmp_list = original_label_lists[original_map2top[label]]
                    nn = label_partition[tmp_list][dist_matrix[order][label_partition[tmp_list]].argsort()[0]]
                    label_partition[label] = nn
                elif tlabel not in filter_labels[0] and (len(label_lists[tlabel]) > 0):
                    # neigh2 = NearestNeighbors(n_neighbors=1)
                    # neigh2.fit(partition_centers[label_partition[label_lists[tlabel]]])
                    # indices = neigh2.kneighbors(label_center.reshape(1, -1), return_distance=False).reshape(-1)
                    # label_partition[label] = label_partition[label_lists[tlabel]][indices[0]]
                    nn = label_partition[label_lists[tlabel]][dist_matrix[order][label_partition[label_lists[tlabel]]].argsort()[0]]
                    label_partition[label] = nn
                else:
                    # indices = neigh.kneighbors(label_center.reshape(1, -1), return_distance=False).reshape(-1)
                    # label_partition[label] = indices[0]
                    nn = dist_matrix[order].argsort()[0]
                    label_partition[label] = nn
                order += 1

        for i in range(partition_num):
            partition_centers[i] = (X_feature[(label_partition[labels] == i)]).sum(axis=0)/(X_feature[(label_partition[labels] == i)]).shape[0]
        partition_labels = np.array(list(map(lambda x: label_partition[x], labels)))

        print("time label partition 2", time.time()-start)

        # from IPython import embed; embed()
        return label_partition, partition_labels, top_partition

    def _tree_partition(self, label_list, label_embeds, cur_idx=0, csize=40, tcnt=1, tot_num=-1)->dict:
        sample_num = 0
        label_num = 0
        avg_embed = []
        for label in label_list:
            sample_num += len(label_embeds[label])
            avg_embed.append(np.mean(np.array(label_embeds[label]), axis=0))
            label_num += 1
        avg_embed = np.array(avg_embed)

        k = max(min(label_num, round(sample_num / csize)), 1)
        # if (tcnt <= 2) or (sample_num > tot_num * 0.5):
        #     k = max(min(label_num, round(sample_num / csize)), 1)
        # else:
        #     k = 1
        # k = max(min(label_num, round(sample_num / csize) - 1), 1)

        start = time.time()
        print("cluster", k, avg_embed.shape[0])
        clusters = KMeans(n_clusters=k, random_state=8).fit_predict(avg_embed)
        print("time cluster", time.time()-start)

        partition = {}
        for i in range(label_num):
            partition[label_list[i]] = clusters[i] + cur_idx
        # from IPython import embed; embed()

        # import matplotlib.pyplot as plt
        # plt.switch_backend('agg')
        # plt.scatter(avg_embed[:, 1], -avg_embed[:, 0], c=clusters)
        # plt.legend()
        # plt.savefig("tmp.png")
        # from IPython import embed; embed()
        info = {}
        return partition, k, info

    def copyPartTree(self, item, item2, tree_list):
        item['id'] = item2['id']
        item['axis'] = item2['axis']

        if item2['child'] is None:
            item['part_id'] = item2['part_id']
            item['size'] = item2['size']
            return
        child1, child2 = item2['child']
        new_item1 = {'id': None, 'child': None, 'axis': None}
        new_item2 = {'id': None, 'child': None, 'axis': None}
        tree_list.append(new_item1)
        tree_list.append(new_item2)
        item['child'] = (new_item1, new_item2)
        self.copyPartTree(new_item1, child1, tree_list)
        self.copyPartTree(new_item2, child2, tree_list)


    def dfsPartTree(self, item, tree_list):
        tree_list.append(item)
        if item['child'] is None:
            return
        child1, child2 = item['child']
        self.dfsPartTree(child1, tree_list)
        self.dfsPartTree(child2, tree_list)


    def reducePartTreeItem(self, item2, label_list):
        # print('reduce item', item2, label_list)
        item = {'id': None, 'child': None, 'axis': None}
        item['id'] = item2['id']
        item['axis'] = item2['axis']

        if item2['child'] is None:
            item['part_id'] = item2['part_id']
            item['size'] = item2['size']
            if item['part_id'] in label_list:
                return item
            return None

        child1, child2 = item2['child']
        new_item1 = self.reducePartTreeItem(child1, label_list)
        new_item2 = self.reducePartTreeItem(child2, label_list)

        if new_item1 is not None and new_item2 is not None:
            item['child'] = (new_item1, new_item2)
            return item
        elif new_item1 is not None:
            return new_item1
        else:
            return new_item2

    def _tree_partition2(self, label_list, label_embeds, cur_idx=0, csize=40, tcnt=1, score_standard=1, tot_num=-1):
        # partition tree by enum grid division
        # score_standard: 0 Difference in length and width of rects
        #                 1 Intersect areas of partitions rects
        sample_num = 0
        label_num = 0
        avg_embed = []
        avg_dists = []
        avg_dists_xy = []
        nums_label = []

        for label in label_list:
            sample_num += len(label_embeds[label])
            nums_label.append(len(label_embeds[label]))
            avg_embed.append(np.mean(np.array(label_embeds[label]), axis=0))
            label_num += 1
        avg_embed = np.array(avg_embed)
        for i, label in enumerate(label_list):
            avg_dists.append(np.mean(cdist(np.array(label_embeds[label]), avg_embed[i].reshape((1, 2)))))
            avg_dists_xy.append(np.mean(np.abs(np.array(label_embeds[label]) - avg_embed[i].reshape((1, 2))), axis=0))
        avg_dists_xy = np.array(avg_dists_xy) * 2

        # import matplotlib.pyplot as plt
        # plt.switch_backend('agg')
        # plt.figure(figsize=(6, 6))
        # plt.clf()
        # for i in range(len(label_list)):
        #     x = [avg_embed[i][0] - avg_dists_xy[i][0], avg_embed[i][0] - avg_dists_xy[i][0],
        #         avg_embed[i][0] + avg_dists_xy[i][0], avg_embed[i][0] + avg_dists_xy[i][0],
        #         avg_embed[i][0] - avg_dists_xy[i][0]]
        #     y = [avg_embed[i][1] - avg_dists_xy[i][1], avg_embed[i][1] + avg_dists_xy[i][1],
        #         avg_embed[i][1] + avg_dists_xy[i][1], avg_embed[i][1] - avg_dists_xy[i][1],
        #         avg_embed[i][1] - avg_dists_xy[i][1]]
        #     plt.plot(y, x, color=plt.cm.tab20(i))
        #     plt.scatter(avg_embed[i][1], avg_embed[i][0], color=plt.cm.tab20(i))
        # plt.show()
        # plt.savefig("1.png")

        nums_label = np.array(nums_label)

        k = max(min(label_num, round(sample_num / csize)), 1)
        # if (tcnt <= 2) or (sample_num > tot_num * 0.5):
        #     k = max(min(label_num, round(sample_num / csize)), 1)
        # else:
        #     k = 1

        if k == 1:
            partition = {}
            for i in range(label_num):
                partition[label_list[i]] = cur_idx
            info = {}
            # info['axis'] = 'x'
            # info['divide'] = [[label_list]]
            # info['ways'] = [[cur_idx]]
            info['axis'] = 'tree'
            part_tree = [{'id': 0, 'labels': np.arange(len(label_list), dtype='int'), 'size': sample_num, 'child': None,
                        'axis': None,
                        'range': np.array([1.0, 1.0]), 'part_id': cur_idx}]
            info['ways'] = part_tree

            return partition, k, info

        def get_cut_cost(now_labels, sorted_id, cut, axis):
            left = -1000
            for i in range(cut + 1):
                left = max(left,
                        avg_embed[now_labels[sorted_id[i]]][axis] + avg_dists_xy[now_labels[sorted_id[i]]][axis])
            right = 1000
            for i in range(cut + 1, len(now_labels)):
                right = min(right,
                            avg_embed[now_labels[sorted_id[i]]][axis] - avg_dists_xy[now_labels[sorted_id[i]]][axis])
            return left - right

        def part_two(now_labels, tot_num, now_range=None):
            # plt.figure(figsize=(6, 6))
            # plt.clf()
            # for ii in range(len(now_labels)):
            #     i = now_labels[ii]
            #     x = [avg_embed[i][0] - avg_dists_xy[i][0], avg_embed[i][0] - avg_dists_xy[i][0], avg_embed[i][0] + avg_dists_xy[i][0], avg_embed[i][0] + avg_dists_xy[i][0], avg_embed[i][0] - avg_dists_xy[i][0]]
            #     y = [avg_embed[i][1] - avg_dists_xy[i][1], avg_embed[i][1] + avg_dists_xy[i][1], avg_embed[i][1] + avg_dists_xy[i][1], avg_embed[i][1] - avg_dists_xy[i][1], avg_embed[i][1] - avg_dists_xy[i][1]]
            #     plt.plot(y, x, color=plt.cm.tab20(i))
            # plt.show()
            # plt.savefig("1.png")
            now_k = max(min(len(now_labels), round(tot_num / csize)), 1)
            divides = [[1 / max(2, min(now_k, 4)), 1 - 1 / max(2, min(now_k, 4))]]
            # divides = [[1/3, 2/3]]
            best_cost = -1
            best_part1 = None
            best_part2 = None
            best_axis = 0
            for i in range(2):
                size = (avg_embed[now_labels] - avg_dists_xy[now_labels]).max(axis=0) - (
                            avg_embed[now_labels] - avg_dists_xy[now_labels]).min(axis=0)
                if now_range is not None and 1 / 2 < size.min() / size.max():
                    size = now_range
                if size[i] < size[1 - i] / 2:
                    continue
                # if now_range is not None and now_range[i] < now_range[1-i]/2:
                #     continue
                pos = avg_embed[now_labels, i]
                sorted_pos = np.sort(pos)
                sorted_id = np.argsort(pos)
                for divide in divides:
                    count = 0
                    bf = 0
                    cur_divide = 0
                    cut_left = 0
                    cut_right = len(now_labels) - 2
                    for y in range(len(now_labels)):
                        count += nums_label[now_labels[sorted_id[y]]]
                        while count >= divide[cur_divide] * tot_num:
                            if count - divide[cur_divide] * tot_num <= divide[cur_divide] * tot_num - count + \
                                    nums_label[
                                        now_labels[sorted_id[y]]] or bf == y:
                                # if cur_divide == 1:
                                gap = y
                                bf = gap + 1
                                # count = 0
                            else:
                                gap = y - 1
                                bf = gap + 1
                                # count = nums_label[now_labels[sorted_id[y]]]

                            if cur_divide == 0:
                                cut_left = min(len(now_labels) - 2, max(0, gap))
                            else:
                                cut_right = max(0, min(len(now_labels) - 2, gap))

                            cur_divide += 1
                            if cur_divide == 2:
                                break

                        if cur_divide == 2:
                            break
                    if cut_left > cut_right:
                        cut_right, cut_left = cut_left, cut_right

                    best_cut = -1
                    best_cut_cost = 0

                    best_cut2 = -1
                    best_cut_cost2 = 0

                    for cut in range(cut_left, cut_right + 1):
                        cut_cost = get_cut_cost(now_labels, sorted_id, cut, i)
                        c_flag = True
                        if cut == 0 or cut == len(now_labels)-2:
                            now_range1 = now_range.copy()
                            now_range2 = now_range.copy()
                            if i == 0:
                                now_range1[0] = now_range1[0] * nums_label[now_labels[sorted_id[0:cut + 1]]].sum() / tot_num
                                now_range2[0] = now_range2[0] * (1 - nums_label[now_labels[sorted_id[0:cut + 1]]].sum() / tot_num)
                            else:
                                now_range1[1] = now_range1[1] * nums_label[now_labels[sorted_id[0:cut + 1]]].sum() / tot_num
                                now_range2[1] = now_range2[1] * (1 - nums_label[now_labels[sorted_id[0:cut + 1]]].sum() / tot_num)
                            if cut==0 and now_range1.min()*2<now_range1.max():
                                c_flag = False
                            if cut==len(now_labels)-2 and now_range2.min()*2<now_range2.max():
                                c_flag = False

                        if c_flag:
                        # if True:
                            if best_cut == -1 or cut_cost < best_cut_cost:
                                best_cut_cost = cut_cost
                                best_cut = cut
                        else:
                            if best_cut2 == -1 or cut_cost < best_cut_cost2:
                                best_cut_cost2 = cut_cost
                                best_cut2 = cut

                    if best_cut == -1:
                        best_cut = best_cut2
                        best_cut_cost = best_cut_cost2

                    if best_part1 is None or best_cut_cost < best_cost:
                        best_cost = best_cut_cost
                        best_part1 = now_labels[sorted_id[0:best_cut + 1]]
                        best_part2 = now_labels[sorted_id[best_cut + 1:]]
                        best_axis = i
            if best_axis == 0:
                best_axis = 'x'
            else:
                best_axis = 'y'
            return best_part1, best_part2, best_axis

        part_tree = [
            {'id': 0, 'labels': np.arange(len(label_list), dtype='int'), 'size': sample_num, 'child': None,
            'axis': None,
            'range': np.array([1.0, 1.0])}]
        part = 1
        id_cnt = 1
        while True:
            id = -1
            chosen = None
            for item in part_tree:
                if item['child'] is None:
                    if len(item['labels']) > 1 and (chosen is None or item['size'] > chosen['size']):
                        chosen = item
                        id = item['id']
            if chosen is None:
                break
            if chosen['size'] < csize:
                break
            part1, part2, axis = part_two(chosen['labels'], chosen['size'], chosen['range'])
            chosen['axis'] = axis
            range1 = chosen['range'].copy()
            range2 = chosen['range'].copy()
            if axis == 'x':
                range1[0] = range1[0] * nums_label[part1].sum() / chosen['size']
                range2[0] = range2[0] * (1 - nums_label[part1].sum() / chosen['size'])
            else:
                range1[1] = range1[1] * nums_label[part1].sum() / chosen['size']
                range2[1] = range2[1] * (1 - nums_label[part1].sum() / chosen['size'])
            new_item1 = {'id': id_cnt, 'labels': part1, 'size': nums_label[part1].sum(), 'child': None, 'axis': None,
                        'range': range1}
            id_cnt += 1
            new_item2 = {'id': id_cnt, 'labels': part2, 'size': nums_label[part2].sum(), 'child': None, 'axis': None,
                        'range': range2}
            id_cnt += 1
            chosen['child'] = (new_item1, new_item2)
            part_tree.append(new_item1)
            part_tree.append(new_item2)

        partition = {}
        start_idx = 0

        for item in part_tree:
            if item["child"] is None:
                for label in item['labels']:
                    partition[label_list[label]] = start_idx + cur_idx
                item["part_id"] = start_idx + cur_idx
                start_idx += 1

        info = {}
        info['axis'] = 'tree'
        info['ways'] = part_tree

        return partition, start_idx, info

    def _tree_partition3(self, label_list, label_embeds, cur_idx=0, csize=40, tcnt=1, score_standard=1, tot_num=-1):
        if len(label_list) > 40:
            return self._tree_partition2(label_list, label_embeds, cur_idx, csize, tcnt, score_standard, tot_num)

        sample_num = 0
        label_num = 0
        avg_embed = []
        avg_dists = []
        avg_dists_xy = []
        nums_label = []

        for label in label_list:
            sample_num += len(label_embeds[label])
            nums_label.append(len(label_embeds[label]))
            avg_embed.append(np.mean(np.array(label_embeds[label]), axis=0))
            label_num += 1
        avg_embed = np.array(avg_embed)
        for i, label in enumerate(label_list):
            avg_dists.append(np.mean(cdist(np.array(label_embeds[label]), avg_embed[i].reshape((1, 2)))))
            avg_dists_xy.append(np.mean(np.abs(np.array(label_embeds[label]) - avg_embed[i].reshape((1, 2))), axis=0))
        avg_dists_xy = np.array(avg_dists_xy) * 2

        sort_x = np.argsort(avg_embed[:, 0])
        sort_y = np.argsort(avg_embed[:, 1])
        rank_x = np.arange(len(avg_embed))
        rank_y = np.arange(len(avg_embed))
        rank_x[sort_x] = np.arange(len(avg_embed))
        rank_y[sort_y] = np.arange(len(avg_embed))
        # for i in avg_embed:
        #     print("tmp_xy = {", i[0], ",", i[1], "};")
        #     print("xy.push_back(tmp_xy);")
        # for i in avg_dists_xy:
        #     print("tmp_xy = {", i[0]/2, ",", i[1]/2, "};")
        #     print("sxy.push_back(tmp_xy);")
        # for i in rank_x:
        #     print("rank_x.push_back(", i, ");")
        # for i in rank_y:
        #     print("rank_y.push_back(", i, ");")
        # for i in nums_label:
        #     print("weight.push_back(", i, ");")

        start = time.time()
        print("treemap size", len(avg_embed))
        tree = gridlayoutOpt.SearchForTree(avg_embed, avg_dists_xy/2, rank_x, rank_y, nums_label)
        print("c search time", time.time()-start)

        nums_label = np.array(nums_label)

        k = max(min(label_num, round(sample_num / csize)), 1)

        if k == 1:
            partition = {}
            for i in range(label_num):
                partition[label_list[i]] = cur_idx
            info = {}
            # info['axis'] = 'x'
            # info['divide'] = [[label_list]]
            # info['ways'] = [[cur_idx]]
            info['axis'] = 'tree'
            part_tree = [{'id': 0, 'labels': np.arange(len(label_list), dtype='int'), 'size': sample_num, 'child': None,
                        'axis': None,
                        'range': np.array([1.0, 1.0]), 'part_id': cur_idx}]
            info['ways'] = part_tree

            return partition, k, info

        def part_two(now_labels, tree_cut):
            # plt.figure(figsize=(6, 6))
            # plt.clf()
            # for ii in range(len(now_labels)):
            #     i = now_labels[ii]
            #     x = [avg_embed[i][0] - avg_dists_xy[i][0], avg_embed[i][0] - avg_dists_xy[i][0], avg_embed[i][0] + avg_dists_xy[i][0], avg_embed[i][0] + avg_dists_xy[i][0], avg_embed[i][0] - avg_dists_xy[i][0]]
            #     y = [avg_embed[i][1] - avg_dists_xy[i][1], avg_embed[i][1] + avg_dists_xy[i][1], avg_embed[i][1] + avg_dists_xy[i][1], avg_embed[i][1] - avg_dists_xy[i][1], avg_embed[i][1] - avg_dists_xy[i][1]]
            #     plt.plot(y, x, color=plt.cm.tab20(i))
            # plt.show()
            # plt.savefig("1.png")

            i = tree_cut[0]
            if i == 0:
                sorted_id = np.argsort(rank_x[now_labels])
            else:
                sorted_id = np.argsort(rank_y[now_labels])

            best_cut = tree_cut[1]

            best_part1 = now_labels[sorted_id[0:best_cut + 1]]
            best_part2 = now_labels[sorted_id[best_cut + 1:]]
            best_axis = i

            if best_axis == 0:
                best_axis = 'x'
            else:
                best_axis = 'y'

            return best_part1, best_part2, best_axis

        part_tree = [
            {'id': 0, 'labels': np.arange(len(label_list), dtype='int'), 'size': sample_num, 'child': None,
            'axis': None,
            'range': np.array([1.0, 1.0])}]

        part = 1
        id_cnt = [1]
        cut_id = [0]

        def dfs_part_two(chosen):
            if len(chosen['labels']) <= 1:
                return
            part1, part2, axis = part_two(chosen['labels'], tree[cut_id[0]])
            cut_id[0] += 1

            chosen['axis'] = axis
            range1 = chosen['range'].copy()
            range2 = chosen['range'].copy()
            if axis == 'x':
                range1[0] = range1[0] * nums_label[part1].sum() / chosen['size']
                range2[0] = range2[0] * (1 - nums_label[part1].sum() / chosen['size'])
            else:
                range1[1] = range1[1] * nums_label[part1].sum() / chosen['size']
                range2[1] = range2[1] * (1 - nums_label[part1].sum() / chosen['size'])
            new_item1 = {'id': id_cnt[0], 'labels': part1, 'size': nums_label[part1].sum(), 'child': None, 'axis': None,
                        'range': range1}
            id_cnt[0] += 1
            new_item2 = {'id': id_cnt[0], 'labels': part2, 'size': nums_label[part2].sum(), 'child': None, 'axis': None,
                        'range': range2}
            id_cnt[0] += 1
            chosen['child'] = (new_item1, new_item2)
            part_tree.append(new_item1)
            dfs_part_two(new_item1)
            part_tree.append(new_item2)
            dfs_part_two(new_item2)

        dfs_part_two(part_tree[0])

        partition = {}
        start_idx = 0

        for item in part_tree:
            if item["child"] is None:
                for label in item['labels']:
                    partition[label_list[label]] = start_idx + cur_idx
                item["part_id"] = start_idx + cur_idx
                start_idx += 1

        info = {}
        info['axis'] = 'tree'
        info['ways'] = part_tree

        return partition, start_idx, info


    def get_sub_partition(self, p, grid_partition, cell, X_embedded, top_partition, labels, partition_center, square_len, use_boundary=False):

        num = labels.shape[0]
        N = square_len * square_len
        now_grids = np.zeros(N, dtype='bool')
        for gid in range(N):
            id = grid_partition[gid]
            if (id<num)and(top_partition[labels[id]]==p):
                now_grids[gid] = True


        idx = (top_partition[labels] == p)
        idx_list = np.arange(num, dtype='int')[idx]

        tmp_embedded = X_embedded[idx].copy()
        tmp_labels = labels[idx].copy()
        label_cnt = Counter(tmp_labels)
        labelmap = {}
        label_list = []
        cur_idx = 0
        for label in label_cnt:
            if label not in labelmap:
                labelmap[label] = cur_idx
                cur_idx += 1
                label_list.append(label)
        tmp_labels = np.array(list(map(lambda x: labelmap[x], tmp_labels))).astype(np.int32)
        label_list = np.array(label_list)
        maxLabel = tmp_labels.max()+1
        tmp_center = partition_center[label_list].copy()

        for i in range(maxLabel):
            tmp_center[i] = tmp_embedded[tmp_labels==i].mean(axis=0)

        c = tmp_embedded.mean(axis=0)
        c2 = np.array([cell.centroid.x, cell.centroid.y])
        tmp_center += c2-c

        while True:
            flag = True
            for i in range(maxLabel):
                if not cell.covers(Point(tmp_center[i])):
                    flag = False
                    break
            if flag:
                break
            tmp_center = (tmp_center-c2)*2/3+c2

        if (maxLabel>1)and(use_boundary):
            import time
            start = time.time()
            new_centers = CentersAdjust(tmp_embedded, tmp_labels, tmp_center, now_hull=cell)
            for i in range(tmp_center.shape[0]):
                tmp_center[i] = np.array(new_centers[i])
            print("time adjust", time.time() - start)
        print('start power')
        try:
            row_asses, cells = getPowerDiagramGrids(tmp_labels, tmp_center, square_len, now_hull=cell, now_grids=now_grids)
        except:
            from IPython import embed; embed(); exit()

        ret_asses = grid_partition.copy()
        ret_asses[now_grids] = idx_list[row_asses[now_grids]]

        cells_dict = {}
        for lb in label_list:
            cells_dict[lb] = cells[labelmap[lb]]

        # 输出grid与partition的对应关系
        return ret_asses, cells_dict

    def get_sub_partition_HV(self, p, grid_partition, cell, X_embedded, top_partition, labels, partition_center, square_len, cut_ways):

        num = labels.shape[0]
        N = square_len * square_len
        from .PowerDiagram import getCutGrids

        now_grids = np.zeros(N, dtype='bool')
        for gid in range(N):
            id = grid_partition[gid]
            if (id<num)and(top_partition[labels[id]]==p):
                now_grids[gid] = True

        idx = (top_partition[labels] == p)
        idx_list = np.arange(num, dtype='int')[idx]

        tmp_embedded = X_embedded[idx].copy()
        tmp_labels = labels[idx].copy()
        label_cnt = Counter(tmp_labels)
        labelmap = {}
        label_list = []
        cur_idx = 0
        for label in label_cnt:
            if label not in labelmap:
                labelmap[label] = cur_idx
                cur_idx += 1
                label_list.append(label)
        tmp_labels = np.array(list(map(lambda x: labelmap[x], tmp_labels))).astype(np.int32)
        label_list = np.array(label_list)
        maxLabel = tmp_labels.max()+1
        tmp_center = partition_center[label_list].copy()

        if cut_ways['axis'] == 'tree':
            tmp_cut = cut_ways
            for item in tmp_cut['ways']:
                if item['child'] is None:
                    item['part_id'] = labelmap[item['part_id']]
        else:
            tmp_cut = cut_ways.copy()
            for i in range(len(tmp_cut['ways'])):
                for j in range(len(tmp_cut['ways'][i])):
                    tmp_cut['ways'][i][j] = labelmap[tmp_cut['ways'][i][j]]

        c = tmp_embedded.mean(axis=0)
        c2 = np.array([cell.centroid.x, cell.centroid.y])
        tmp_center += c2-c

        while True:
            flag = True
            for i in range(maxLabel):
                if not cell.covers(Point(tmp_center[i])):
                    flag = False
                    break
            if flag:
                break
            tmp_center = (tmp_center-c2)*2/3+c2

        row_asses, cells = getCutGrids(tmp_labels, tmp_center, square_len, tmp_cut, now_hull=cell, now_grids=now_grids)

        ret_asses = grid_partition.copy()
        ret_asses[now_grids] = idx_list[row_asses[now_grids]]

        cells_dict = {}
        for lb in label_list:
            cells_dict[lb] = cells[labelmap[lb]]

        if tmp_cut['axis'] == 'tree':
            for item in tmp_cut['ways']:
                if item['child'] is None:
                    item['part_id'] = label_list[item['part_id']]

        # 输出grid与partition的对应关系
        return ret_asses, cells_dict, tmp_cut

    def get_power_partition(self, X_embedded, labels, partition_center, square_len, use_boundary=False, reduce=None, major_coords=None):

        num = labels.shape[0]
        N = square_len * square_len
        maxLabel = labels.max()+1

        if (maxLabel>1)and(use_boundary):
            import time
            start = time.time()
            new_centers = CentersAdjust(X_embedded, labels, partition_center, reduce, major_coords)
            for i in range(partition_center.shape[0]):
                partition_center[i] = np.array(new_centers[i])
            print("time adjust", time.time() - start)
        row_asses, cells = getPowerDiagramGrids(labels, partition_center, square_len)

        # 输出grid与partition的对应关系
        return row_asses, cells

    def get_power_partition_zoom(self, labels, zoom_partition_map, zoom_min, zoom_max, all_cells_bf, partition_center, square_len):

        num = labels.shape[0]
        N = square_len * square_len
        maxLabel = labels.max()+1

        zoom_size = zoom_max-zoom_min
        # zoom_max = zoom_max + 0.1*zoom_size
        # zoom_min = zoom_min - 0.1*zoom_size

        import time
        start = time.time()
        from .PowerDiagram import CentersAdjustZoom
        new_centers = CentersAdjustZoom(zoom_partition_map, zoom_min, zoom_max, all_cells_bf, partition_center)
        for i in range(partition_center.shape[0]):
            partition_center[i] = np.array(new_centers[i])
        print("time adjust", time.time() - start)

        row_asses, cells = getPowerDiagramGrids(labels, partition_center, square_len, compact=False)

        # 输出grid与partition的对应关系
        return row_asses, cells

    def get_partition_HV(self, X_embedded, labels, partition_center, square_len, reduce=None):

        num = labels.shape[0]
        N = square_len * square_len
        from .PowerDiagram import getCutGrids
        maxLabel = labels.max()+1

        label_list = np.arange(maxLabel, dtype='int')

        tmp_embedded = X_embedded
        tmp_labels = labels
        if reduce is not None:
            tmp_embedded = tmp_embedded[reduce]
            tmp_labels = tmp_labels[reduce]
        label_embeds = {}
        for lb in label_list:
            label_embeds[lb] = tmp_embedded[(tmp_labels==lb)]
        partition, k, cut_ways = self._tree_partition3(label_list, label_embeds, 0, 1)

        labelmap = {}
        for lb in partition:
            labelmap[partition[lb]] = lb

        tmp_cut = cut_ways
        for item in tmp_cut['ways']:
            if item['child'] is None:
                item['part_id'] = labelmap[item['part_id']]
                item['size'] = (labels==item['part_id']).sum()
        for item in reversed(tmp_cut['ways']):
            if item['child'] is not None:
                child1, child2 = item['child']
                item['size'] = child1['size'] + child2['size']

        row_asses, cells = getCutGrids(labels, partition_center, square_len, tmp_cut)

        for item in tmp_cut['ways']:
            if item['child'] is None:
                item['part_id'] = str(item['part_id'])+"-top"

        # 输出grid与partition的对应关系
        return row_asses, cells, tmp_cut

    def get_partition_HV_zoom(self, labels, zoom_partition_map, all_tree_bf, partition_center, square_len):

        num = labels.shape[0]
        N = square_len * square_len
        from .PowerDiagram import getCutGrids
        maxLabel = labels.max()+1

        label_list = np.arange(maxLabel, dtype='int')

        tmp_cut = {}
        tmp_cut['axis'] = 'tree'
        tmp_cut['ways'] = []
        part_list = np.array(list(map(lambda x: zoom_partition_map[x], label_list)))
        top_item = self.reducePartTreeItem(all_tree_bf[0], part_list)
        self.dfsPartTree(top_item, tmp_cut['ways'])

        labelmap = {}
        for lb in zoom_partition_map:
            labelmap[zoom_partition_map[lb]] = lb

        for item in tmp_cut['ways']:
            if item['child'] is None:
                item['part_id'] = labelmap[item['part_id']]
                item['size'] = (labels==item['part_id']).sum()
        for item in reversed(tmp_cut['ways']):
            if item['child'] is not None:
                child1, child2 = item['child']
                item['size'] = child1['size'] + child2['size']

        row_asses, cells = getCutGrids(labels, partition_center, square_len, tmp_cut)

        for item in tmp_cut['ways']:
            if item['child'] is None:
                item['part_id'] = str(item['part_id'])+"-top"

        # 输出grid与partition的对应关系
        return row_asses, cells, tmp_cut

    def get_foldline_partition(self, x_bf, y_bf, grid_bf, label_bf, labels, partition_center, square_len):

        num = labels.shape[0]
        N = square_len * square_len
        from .PowerDiagram import getFoldlineGrids
        grid_label = np.full((x_bf, y_bf), fill_value=-1, dtype='int')
        for i in range(grid_bf.shape[0]):
            x = round(x_bf*grid_bf[i][0]-1/2)
            y = round(y_bf*grid_bf[i][1]-1/2)
            grid_label[x][y] = label_bf[i]
        row_asses, cells = getFoldlineGrids(x_bf, y_bf, grid_label, labels, partition_center, square_len)

        # 输出grid与partition的对应关系
        return row_asses, cells

    def grid(self, labels, X_feature, true_id, hierarchy, top_labels, filter_labels=None, info_before=None, P=None, scale=1/2, confs_hierarchy=None, shres=0.8):
        random.seed(42)
        np.random.seed(42)

        import matplotlib.pyplot as plt
        plt.switch_backend('agg')
        from application.grid.colors import MyColorMap
        cm = MyColorMap()
        plt.figure(figsize=(6, 6))

        use_HV = False
        use_conf = True

        if self.Ctrler is not None and self.Ctrler.use_HV is not None:
            use_HV = self.Ctrler.use_HV
        if self.Ctrler is not None and self.Ctrler.use_conf is not None:
            use_conf = self.Ctrler.use_conf


        num = labels.shape[0]
        square_len = math.ceil(np.sqrt(num))
        N = square_len * square_len

        original_top_labels = top_labels.copy()
        original_labels = labels.copy()
        original_filter = copy.deepcopy(filter_labels)

        # from collections import Counter
        # print("top labels: ", Counter(top_labels))
        # print("filter labels: ", filter_labels[0])
        # print("labels: ", Counter(labels))

        #----------------------------------将top_label更改为上层分块----------------------------------------

        if info_before is not None:
            for i in range(num):
                if top_labels[i] in filter_labels[0]:
                    if labels[i] not in filter_labels[1]:
                        filter_labels[1].append(labels[i])

            filter_labels[0] = []
            selected_bf = info_before['selected_bf']
            selected_now = info_before['selected']
            partition_info_bf = info_before['partition_info_bf']
            partition_labels_bf = partition_info_bf['partition_labels']

            cnt = 0
            partition_bf_id = {}
            partition_bf_list = []
            map_bf = {}
            for i in range(len(selected_now)):
                # print(i)
                # print(selected_now[i])
                # print(selected_bf[i])
                if partition_labels_bf[selected_bf[i]] not in partition_bf_id:
                    partition_bf_id[partition_labels_bf[selected_bf[i]]] = cnt
                    partition_bf_list.append(partition_labels_bf[selected_bf[i]])
                    cnt += 1
                map_bf[top_labels[selected_now[i]]] = partition_bf_id[partition_labels_bf[selected_bf[i]]]

            top_list = []
            for i in range(cnt):
                top_list.append([])
            cnt2 = 0
            top_list2 = []
            top_id2 = []
            map_bf2 = {}
            for i in range(num):
                if top_labels[i] in map_bf:
                    top_list[map_bf[top_labels[i]]].append(X_feature[i])
                else:
                    if top_labels[i] not in map_bf2:
                        map_bf2[top_labels[i]] = cnt2
                        cnt2 += 1
                        top_list2.append([])
                        top_id2.append(top_labels[i])
                    top_list2[map_bf2[top_labels[i]]].append(X_feature[i])
            if cnt2 > 0:
                for i in range(cnt):
                    top_list[i] = np.array(top_list[i]).mean(axis=0)
                for i in range(cnt2):
                    top_list2[i] = np.array(top_list2[i]).mean(axis=0)
                top_dist = cdist(top_list2, top_list, "eu")
                for i in range(cnt2):
                    map_bf[top_id2[i]] = top_dist[i].argsort()[0]

            # for i in range(num):
            #     if top_labels[i] not in map_bf:
            #         if labels[i] not in filter_labels[1]:
            #             print("???", filter_labels[1], (labels==labels[i]).sum(), labels[i], top_labels[i])
            #         map_bf[top_labels[i]] = cnt
            #         cnt += 1

            top_labels = np.array(list(map(lambda x: map_bf[x], top_labels))).astype(np.int32)

        #----------------------------------START LABEL PARTITION----------------------------------------

        import time
        start = time.time()

        # label normalization
        def norm_labels(labels):
            labelmap = {}
            cur_idx = 0
            for label in labels:
                label = int(label)
                if label not in labelmap:
                    labelmap[label] = cur_idx
                    cur_idx += 1
            return labelmap

        labelmap = norm_labels(labels)
        labels = np.array(list(map(lambda x: labelmap[x], labels))).astype(np.int32)
        filter_labels[1] = np.array(list(map(lambda x: labelmap[x], filter_labels[1]))).astype(np.int32)
        label_partition, partition_labels, top_partition = self.get_label_partition_feature(X_feature, labels, top_labels, filter_labels, original_top_labels=original_top_labels)

        tlabelmap = norm_labels(top_partition)
        top_partition = np.array(list(map(lambda x: tlabelmap[x], top_partition))).astype(np.int32)

        print("time label partition", time.time()-start)

        # print(filter_labels)
        # print(label_partition)
        # print(top_partition)

        partitions = label_partition.max()+1

        #----------------------------------END LABEL PARTITION----------------------------------------

        #----------------------------------START CONF CALCULATE----------------------------------------

        now_conf = None
        conf_vis = None
        if confs_hierarchy is not None:
            now_conf = np.zeros((num, partitions))
            conf_label_list = {}
            for i in range(num):
                o_label = original_labels[i]
                p_label = label_partition[labels[i]]
                if p_label not in conf_label_list:
                    conf_label_list[p_label] = []
                if o_label not in original_filter[1]:
                    if o_label not in conf_label_list[p_label]:
                        conf_label_list[p_label].append(o_label)
            print("origin label", conf_label_list)
            for p_label in conf_label_list:
                conf_label_list[p_label] = np.array(list(map(lambda x: confs_hierarchy['id_map'][x], conf_label_list[p_label])))
            conf_labels = np.array(list(map(lambda x: confs_hierarchy['id_map'][x], original_labels)))
            tmp_conf = confs_hierarchy['confs'][true_id]
            for j in range(partitions):
                now_conf[:, j] = tmp_conf[:, conf_label_list[j]].sum(axis=1)
                tmp_idx = label_partition[labels] == j
                now_conf[tmp_idx, j] = tmp_conf[tmp_idx, conf_labels[tmp_idx]]

            labelmap = norm_labels(original_labels)
            tmp_labels = np.array(list(map(lambda x: labelmap[x], original_labels))).astype(np.int32)
            conf_vis = {"labelmap": labelmap}
            conf_vis["confs"] = np.zeros((num, tmp_labels.max()+1))
            for j in labelmap:
                conf_vis["confs"][:, labelmap[j]] = tmp_conf[:, confs_hierarchy['id_map'][j]]
            conf_vis["confs"] = conf_vis["confs"].tolist()

        #----------------------------------END CONF CALCULATE----------------------------------------

        #----------------------------------START CENTER CALCULATE----------------------------------------

        print("time start FD layout", time.time()-start)

        partition_center, edge_matrix = self.get_FD_layout_centers(X_feature, top_partition, label_partition[labels], info_before, now_conf)
        partition_center = self.adjust_FD_layout_centers(top_partition, label_partition[labels], partition_center, original_top_labels)
        top_center = self.get_FD_layout_top(top_partition, label_partition[labels], partition_center)

        # print("top center", top_center)

        print("time FD layout", time.time()-start)
        time_dec = time.time()-start

        # print(label_partition, partition_center)

        #----------------------------------END CENTER CALCULATE----------------------------------------

        #----------------------------------START GRID PARTITION----------------------------------------

        # plt.figure(figsize=(6, 6))


        use_boundary = False    # 基于边界划分
        if info_before is not None:
            use_grid_embedded = True    # 使用grid划分
        else:
            use_grid_embedded = False

        tmp_embedded = partition_center[label_partition[labels]].copy()
        tmp_labels = labels
        partition_center2 = partition_center.copy()
        top_center2 = top_center.copy()
        selected_e = None
        x_bf, y_bf = 0, 0
        grid_zoom_flag = False
        zoom_partition_map = {}
        zoom_max = None
        zoom_min = None

        if use_grid_embedded:
            new_flag = False
            if info_before is not None:

                N2 = info_before['grid_asses'].shape[0]
                tmp_embedded2 = get_layout_embedded(info_before['grid_asses'], round(np.sqrt(N2)))
                tmp_embedded[info_before['selected']] = tmp_embedded2[info_before['selected_bf']].copy()

                tmp_xy = (tmp_embedded[info_before['selected']].max(axis=0)-tmp_embedded[info_before['selected']].min(axis=0))*round(np.sqrt(N2))
                x_bf = round(tmp_xy[0])+1
                y_bf = round(tmp_xy[1])+1

                tmp_min = tmp_embedded[info_before['selected']].min(axis=0)
                tmp_max = tmp_embedded[info_before['selected']].max(axis=0)+1/np.sqrt(N2)
                tmp_embedded[info_before['selected']] -= tmp_embedded[info_before['selected']].min(axis=0)-1/2/np.sqrt(N2)
                tmp_embedded[info_before['selected']] /= tmp_embedded[info_before['selected']].max(axis=0)+1/2/np.sqrt(N2)
                tmp_labels2 = np.ones(N2, dtype='int')*(-1)
                tmp_labels2[info_before['selected_bf']] = top_partition[label_partition[labels[info_before['selected']]]]
                is_selected = np.zeros(N2, dtype='bool')
                is_selected[info_before['selected_bf']] = True

                selected_e = np.zeros(tmp_embedded.shape[0], dtype='bool')
                selected_e[info_before['selected']] = True

                for p in top_partition:
                    sum = (top_partition[label_partition[labels[info_before['selected']]]]==p).sum()
                    # print('partition sum', p, sum)
                    if sum == 0:
                        new_flag = True
                        use_grid_embedded = False
                        break

                if not new_flag:
                    grid_zoom_flag = True
                    grid_bf = tmp_embedded[info_before['selected']]
                    label_bf = top_partition[label_partition[labels[info_before['selected']]]]

                    for p in top_partition:
                        idx = (top_partition[label_partition[labels[info_before['selected']]]]==p)
                        count = Counter(partition_labels_bf[np.array(info_before['selected_bf'])[idx]])
                        zoom_partition_map[p] = max(count, key=lambda x:count[x])
                    zoom_max = tmp_max
                    zoom_min = tmp_min

                    top_center2 = self.get_centers(tmp_embedded[info_before['selected']], top_partition[label_partition[labels[info_before['selected']]]])
                    # print("top_center2", top_center2)

        if not use_grid_embedded:
            tmp_embedded = partition_center[label_partition[labels]].copy()
            tmp_labels = labels
            partition_center2 = partition_center.copy()
            top_center2 = top_center.copy()
            selected_e = None

        print("time partition pre", time.time()-start)

        if use_HV:
            if grid_zoom_flag:
                all_tree_bf = partition_info_bf['part_way']
                grid_partition, cells, top_way = self.get_partition_HV_zoom(top_partition[label_partition[tmp_labels]], zoom_partition_map, all_tree_bf, top_center2, square_len)
            else:
                grid_partition, cells, top_way = self.get_partition_HV(tmp_embedded, top_partition[label_partition[tmp_labels]], top_center2, square_len, selected_e)
        else:
            if grid_zoom_flag:
                all_cells_bf = partition_info_bf['part_way']
                grid_partition, cells = self.get_power_partition_zoom(top_partition[label_partition[tmp_labels]], zoom_partition_map, zoom_min, zoom_max, all_cells_bf, top_center2, square_len)
            else:
                grid_partition, cells = self.get_power_partition(tmp_embedded, top_partition[label_partition[tmp_labels]], top_center2, square_len, False)

        print("time partition top grid", time.time()-start)

        # self.optimizer.show_grid(grid_partition, top_partition[label_partition[labels]], square_len, 'partition_top.png', just_save=True)

        tmp_embedded = partition_center[label_partition[labels]].copy()

        all_cells = {}
        all_tree = []
        if use_HV:
            all_tree = top_way['ways']

        for p in range(top_partition.max()+1):
            idx = (top_partition[label_partition[tmp_labels]]==p)
            sub_cnt = Counter(label_partition[tmp_labels[idx]])
            # print("sub partition", p, sub_cnt)
            tmp_label_list = np.arange(len(top_partition), dtype='int')[top_partition==p]
            tmp_label_embeds = {}
            for lb in tmp_label_list:
                tmp_label_embeds[lb] = tmp_embedded[(label_partition[tmp_labels]==lb)]
            if len(sub_cnt)>1:
                if use_HV:
                    tmp_part, _, tmp_cut = self._tree_partition3(tmp_label_list, tmp_label_embeds, 0, 1)
                    tmp_labelmap = {}
                    for lb in tmp_part:
                        tmp_labelmap[tmp_part[lb]] = lb
                    for item in tmp_cut['ways']:
                        if item['child'] is None:
                            item['part_id'] = tmp_labelmap[item['part_id']]
                    # print("ways:", tmp_cut)
                    grid_partition, _, tmp_cut = self.get_sub_partition_HV(p, grid_partition, cells[p], tmp_embedded, top_partition, label_partition[tmp_labels], partition_center2, square_len, tmp_cut)
                    for item in all_tree:
                        if 'part_id' in item and item['part_id'] == str(p)+"-top":
                            item['part_id'] = None
                            self.copyPartTree(item, tmp_cut['ways'][0], all_tree)
                            break
                else:
                    grid_partition, cells_dict = self.get_sub_partition(p, grid_partition, cells[p], tmp_embedded, top_partition, label_partition[tmp_labels], partition_center2, square_len, False)
                    all_cells.update(cells_dict)
            else:
                if use_HV:
                    for item in all_tree:
                        if 'part_id' in item and item['part_id'] == str(p)+"-top":
                            for key in sub_cnt:
                                item['part_id'] = key
                                item['size'] = sub_cnt[key]
                else:
                    for key in sub_cnt:
                        all_cells[key] = cells[p]


        print("time partition grid", time.time()-start)

        # self.optimizer.show_grid(grid_partition, label_partition[labels], square_len, 'partition_sub.png', just_save=True)

        # plt.figure(figsize=(6, 6))
        # for i in range(X_embedded.shape[0]):
        #     plt.scatter(X_embedded[i][1], X_embedded[i][0], color=plt.cm.tab20(labels[i]))
        # plt.savefig("tmp_old.png")
        # plt.show()

        # start0 = time.time()
        # grid_asses_ori, _, _, _, _ = self.optimizer.grid(X_embedded, labels, type='Triple', maxit=5, maxit2=0,
        #                                                  use_global=False,
        #                                                  use_local=False, only_compact=False, swap_cnt=2147483647,
        #                                                  pred_labels=labels, swap_op_order=False,
        #                                                  choose_k=1)
        # time_ori = time.time()-start0
        # time_dec += time_ori
        # print("ori done", time.time()-start0)

        # start0 = time.time()
        # grid_asses_g, _, _, _, _ = self.optimizer.grid(X_embedded, labels, type='PerimeterRatio', maxit=0, maxit2=5,
        #                                                  use_global=True,
        #                                                  use_local=False, only_compact=False, swap_cnt=2147483647,
        #                                                  pred_labels=labels, swap_op_order=False,
        #                                                  choose_k=1)
        # print("global done", time.time()-start0)

        # start0 = time.time()
        # # np.savez("whole.npz", X=X_embedded, labels=labels)
        # grid_asses_old, _, _, _, _ = self.optimizer.grid(X_embedded, labels, type='PerimeterRatio', maxit=0, maxit2=8,
        #                                                  use_global=True,
        #                                                  use_local=True, only_compact=False, swap_cnt=2147483647,
        #                                                  pred_labels=labels, swap_op_order=False,
        #                                                  choose_k=1)
        # print("whole done", time.time()-start0)
        # time_old = time.time()-start0
        # time_dec += time_old

        # print("time partition rotate", time.time()-start)

        #----------------------------------END GRID PARTITION----------------------------------------

        #----------------------------------START GRID ASSIGN----------------------------------------

        top_partition_full = np.concatenate([top_partition, [-1]], axis=0)
        label_partition_full = np.concatenate([label_partition, [top_partition_full.shape[0]-1]], axis=0)
        # label_partition_full = np.concatenate([label_partition, [-1]], axis=0)
        labels_full = np.concatenate([labels, np.full(N-num, fill_value=label_partition_full.shape[0]-1, dtype='int')], axis=0)

        _partition = label_partition_full[labels_full[grid_partition]]

        end = time.time()
        print("time partition finish", end-start)

        grid_asses = grid_partition.copy()

        confusion = None
        if_confuse = None
        if use_conf and confs_hierarchy is not None:
            conf = now_conf.copy()
            conf_max = conf[np.arange(len(labels)), label_partition[labels]]
            sheld = 0.8
            conf_argsort = np.argsort(-conf_max)
            # sheld = min(0.8, conf_max[conf_argsort[int(len(conf_max)*3/4)]])
            sheld = shres
            if_confuse = conf_max<sheld
            confuse_idx = np.arange(len(labels))[if_confuse]
            print(if_confuse.sum())
            confuse_class = np.argsort(-conf, axis=1)
            confusion = {"confuse_class": confuse_class, "if_confuse": if_confuse, "conf": now_conf, "conf_vis": conf_vis}

        alpha = scale
        scale = max(square_len*alpha, 0)

        if info_before is not None:
            # save_pickle({"grid_asses": grid_asses, "labels": label_partition[labels], "info_before": info_before, "scale": scale}, "stability2.pkl")
            grid_asses, grid_new = AssignQAP(grid_asses, label_partition[labels], P, scale, grid_asses_bf=info_before['grid_asses'], selected=info_before['selected'], selected_bf=info_before['selected_bf'], feature=X_feature, confusion=confusion, small_labels=labels, use_HV=use_HV)
        else:
            # save_pickle({"grid_asses": grid_asses, "labels": label_partition[labels], "feature": X_feature}, "top.pkl")
            grid_asses, grid_new = AssignQAP(grid_asses, label_partition[labels], P, scale, feature=X_feature, confusion=confusion, small_labels=labels, use_HV=use_HV)


        # for i in confuse_idx:
        #     print(true_id[i], now_conf[i])

        # embedded = np.array(get_layout_embedded(grid_asses, square_len))[:len(labels)]
        # fig = plt.figure(figsize=(6, 6))
        # plt.clf()
        # for i in range(label_partition[labels].max()+1):
        #     print(i)
        #     plt.scatter(embedded[label_partition[labels]==i, 1], embedded[label_partition[labels]==i, 0], color=plt.cm.tab20(i))
        # plt.scatter(embedded[confuse_idx, 1], embedded[confuse_idx, 0], color='black')
        # plt.savefig("tmp_conf.png")
        # plt.clf()
        # plt.scatter(embedded[confuse_idx, 1], embedded[confuse_idx, 0], color=plt.cm.tab20(confuse_class[confuse_idx, 0]))
        # plt.savefig("tmp_conf1.png")
        # plt.clf()
        # plt.scatter(embedded[confuse_idx, 1], embedded[confuse_idx, 0], color=plt.cm.tab20(confuse_class[confuse_idx, 1]))
        # plt.savefig("tmp_conf2.png")
        # plt.clf()
        # plt.scatter(embedded[confuse_idx, 1], embedded[confuse_idx, 0], color=plt.cm.tab20(confuse_class[confuse_idx, 2]))
        # plt.savefig("tmp_conf3.png")
        # plt.clf()

        # tmp_confs = confs_hierarchy['confs'][true_id]
        # tmp_confs2 = np.zeros((num, 398))
        # for i in range(398):
        #     tmp_confs2[:, i] = tmp_confs[:, confs_hierarchy['id_map'][i]]
        # print(tmp_confs2.sum())
        # plt.clf()
        # for i in range(label_partition[labels].max()+1):
        #     plt.scatter(embedded[label_partition[labels]==i, 1], embedded[label_partition[labels]==i, 0], color=plt.cm.tab20(i))
        # tmp_idx = (tmp_confs2.max(axis=1)<0.8)
        # plt.scatter(embedded[tmp_idx, 1], embedded[tmp_idx, 0], color='black')
        # plt.savefig("tmp_conf0.png")

        end = time.time()
        time_ours = end-start-time_dec

        print('done')
        print("time", end-start)
        # ----------------------------------END GRID ASSIGN----------------------------------------

        partition_info = {}
        partition_info['partition_labels'] = label_partition[labels]
        if not use_HV:
            partition_info['part_way'] = all_cells
        else:
            for item in reversed(all_tree):
                if item['child'] is not None:
                    child1, child2 = item['child']
                    # print(child1['size'], child2['size'])
                    item['size'] = child1['size'] + child2['size']
            partition_info['part_way'] = all_tree

        # -------------------------------  draw grid  -----------------------------
        #
        # otlabelmap = norm_labels(original_top_labels)
        # tmp_partition = np.array(list(map(lambda x: otlabelmap[x], original_top_labels))).astype(np.int32)
        # self.optimizer.show_grid(grid_asses, tmp_partition, square_len, 'partition_old.png', just_save=True)
        #
        # # self.optimizer.show_grid(grid_asses, labels_full, square_len, 'final0.png', showNum=True, scatter=X_embedded_a)
        # if info_before is not None:
        #     self.optimizer.show_grid(grid_asses, labels, square_len, 'final1.png', just_save=True)
        # else:
        #     self.optimizer.show_grid(grid_asses, labels, square_len, 'final0.png', just_save=True)
        # self.optimizer.show_grid(grid_asses_ori, labels, square_len, 'final2.png')
        # self.optimizer.show_grid(grid_asses_old, labels, square_len, 'final3.png')
        # self.optimizer.show_grid(grid_asses_g, labels, square_len, 'final4.png')
        #
        # plt.clf()
        # for i in range(X_embedded.shape[0]):
        #     plt.scatter(X_embedded[i][1], X_embedded[i][0], color=plt.cm.tab20(labels[i]))
        # plt.savefig("tsne.png")
        # plt.show()

        # # -------------------------------  proximity  -----------------------------
        #
        # def testNeighbor(a, b, maxk=50, labels=None, type='all'):
        #     start = time.time()
        #     order = np.arange(a.shape[0], dtype='int')
        #     np.random.seed(5)
        #     np.random.shuffle(order)
        #     dist_a = cdist(a, a[order], "euclidean")
        #     dist_b = cdist(b, b[order], "euclidean")
        #     arg_a = order[np.argsort(dist_a, axis=1)]
        #     arg_b = order[np.argsort(dist_b, axis=1)]
        #
        #     # print("dist time", time.time()-start)
        #     nn = len(a)
        #     p1 = np.zeros(maxk)
        #     p2 = np.zeros(maxk)
        #     if type == 'cross':
        #         for k in range(maxk):
        #             for i in range(nn):
        #                 diff = labels[arg_a[i]] != labels[i]
        #                 diff2 = labels[arg_b[i]] != labels[i]
        #                 # p1[k] += len(set(arg_a[i][diff][:k+1]).intersection(set(arg_b[i][diff2][:k+1])))
        #                 p1[k] += (labels[np.array(list(set(arg_a[i][:k + 2]).intersection(set(arg_b[i][:k + 2]))))] !=
        #                           labels[i]).sum()
        #                 p2[k] += 1
        #     else:
        #         for k in range(maxk):
        #             for i in range(nn):
        #                 p1[k] += len(set(arg_a[i][:k + 2]).intersection(set(arg_b[i][:k + 2]))) - 1
        #                 p2[k] += 1
        #     ret = p1 / p2
        #
        #     if labels is not None:
        #         cnt = 0
        #         for i in range(nn):
        #             cnt += (labels[arg_a[i][:maxk]] == labels[i]).sum()
        #         print(cnt, maxk * nn - cnt)
        #
        #     return ret
        #
        # def AUC(y):
        #     cnt = 0
        #     a_20 = 0
        #     a_full = 0
        #     for i in range(len(y)):
        #         cnt += y[i]
        #         if i == 19:
        #             a_20 = cnt
        #     a_full = cnt
        #     return a_full, a_20
        #
        # grid_new = get_layout_embedded(grid_asses, square_len)
        #
        # # avg_knnp3 = 0
        # # for label in range(labels.max() + 1):
        # #     idx = np.arange(len(labels))[labels == label]
        # #     grids3 = grid_new[idx]
        # #     knnp3 = testNeighbor(X_feature[idx], grids3)
        # #     avg_knnp3 += knnp3 * len(idx)
        # # avg_knnp3 /= len(labels)
        #
        # avg_knnp3 = testNeighbor(X_feature, grid_new)
        #
        # x = (np.arange(avg_knnp3.shape[0], dtype='int') + 1)
        # plt.clf()
        # auc50, auc20 = AUC(avg_knnp3)
        # # auc50 = auc20 = 0
        # print("auc20", auc20, "auc50", auc50)
        # # plt.plot(x, avg_knnp3, label='qap' + ", auc20=%d" % auc20 + ", auc50=%d" % auc50, linewidth=0.5)
        # # plt.legend()
        # # plt.ylim(0, 40)
        # # if info_before is None:
        # #     plt.savefig("qap_knn_top.png", dpi=300)
        # # else:
        # #     plt.savefig("qap_knn_zoom.png", dpi=300)
        # # np.save("qap_knnp_"+str(alpha), avg_knnp3)
        #
        # # -------------------------------  stability  -----------------------------
        #
        # if info_before is not None:
        #
        #     N2 = info_before['grid_asses'].shape[0]
        #     tmp_embedded2 = get_layout_embedded(info_before['grid_asses'], round(np.sqrt(N2)))
        #     square_len2 = round(np.sqrt(N2))
        #
        #     tmp_min = tmp_embedded2[info_before['selected_bf']].min(axis=0)
        #     tmp_max = tmp_embedded2[info_before['selected_bf']].max(axis=0) + 1 / np.sqrt(N2)
        #     tmp_labels2 = np.ones(N2, dtype='int') * (-1)
        #     tmp_labels2[info_before['selected_bf']] = top_partition[label_partition[labels[info_before['selected']]]]
        #     is_selected = np.zeros(N2, dtype='bool')
        #     is_selected[info_before['selected_bf']] = True
        #
        #     tmp_min = np.array([1, 1])
        #     tmp_max = np.array([0, 0])
        #     major_coords = gridlayoutOpt.getConnectShape(info_before['grid_asses'], tmp_labels2, is_selected)
        #     for i in range(len(major_coords)):
        #         for j in range(len(major_coords[i])):
        #             tmp_min = np.minimum(tmp_min, np.array(major_coords[i][j]).min(axis=0))
        #             tmp_max = np.maximum(tmp_max, np.array(major_coords[i][j]).max(axis=0))
        #     for i in range(len(major_coords)):
        #         for j in range(len(major_coords[i])):
        #             major_coords[i][j] = (np.array(major_coords[i][j]) - tmp_min) / (tmp_max - tmp_min)
        #
        #     zoom_partition_map = {}
        #     for p in top_partition:
        #         idx = (top_partition[label_partition[labels[info_before['selected']]]]==p)
        #         if idx.sum() > 0:
        #             count = Counter(partition_labels_bf[np.array(info_before['selected_bf'])[idx]])
        #             zoom_partition_map[p] = max(count, key=lambda x:count[x])
        #     tmp_min2 = np.array([square_len2, square_len2])
        #     tmp_max2 = np.array([0, 0])
        #     major_points = {}
        #     for i in range(len(info_before['grid_asses'])):
        #         id = info_before['grid_asses'][i]
        #         if is_selected[id]:
        #             lb = tmp_labels2[id]
        #             if zoom_partition_map[lb] != partition_labels_bf[id]:
        #                 continue
        #             if lb not in major_points:
        #                 major_points[lb] = []
        #             tmp_min2 = np.minimum(tmp_min2, [i // square_len2, i % square_len2])
        #             tmp_max2 = np.maximum(tmp_max2, [i // square_len2 + 0.5, i % square_len2 + 0.5])
        #             major_points[lb].append([i // square_len2, i % square_len2])
        #             major_points[lb].append([i // square_len2, i % square_len2 + 0.5])
        #             major_points[lb].append([i // square_len2 + 0.5, i % square_len2])
        #             major_points[lb].append([i // square_len2 + 0.5, i % square_len2 + 0.5])
        #     for lb in major_points:
        #         major_points[lb] = (np.array(major_points[lb]) - tmp_min2) / (tmp_max2 - tmp_min2)
        #
        #     from .PowerDiagram import get_graph_from_coords
        #     from .testMeasure import checkShape, checkShapeAndPosition, checkPosition2
        #     if use_HV and self.Ctrler.scenario == "ans":
        #         shapes = get_graph_from_coords(major_coords, graph_type="origin")
        #     else:
        #         shapes = get_graph_from_coords(major_coords, graph_type="hull", major_points=major_points)
        #
        #     IoU_ours, _ = checkShapeAndPosition(grid_asses, top_partition[label_partition[labels]], square_len, shapes)
        #     print("IoU", IoU_ours)
        #     # dist_ours = checkShape(grid_asses, top_partition[label_partition[labels]], square_len, shapes, "dist")
        #     # print("dist", dist_ours)
        #     relative = checkPosition2(grid_asses, label_partition[labels], square_len, info_before)
        #     print('stab-position', relative)
        #
        #     from.testMeasure import checkXYOrder
        #     order_score, order_cnt = checkXYOrder(get_layout_embedded(grid_asses, square_len), labels, grid_asses_bf=info_before['grid_asses'], selected=info_before['selected'], selected_bf=info_before['selected_bf'], if_confuse=if_confuse)
        #     print("order score", order_score, order_score/(order_cnt+1e-12))
        #
        # # -------------------------------  ambiguity  -----------------------------
        #
        # if confusion is not None:
        #     from .testMeasure import checkConfusion
        #     confusion_score = checkConfusion(get_layout_embedded(grid_asses, square_len), label_partition[labels], confusion)
        #     print("confusion score", confusion_score)
        #
        # # -------------------------------  compactness convexity  -----------------------------
        #
        # from .testMeasure import check_cost_type
        # consider = np.zeros(grid_asses.shape[0], dtype='bool')
        # for i in range(len(grid_asses)):
        #     if grid_asses[i] < len(labels):
        #         consider[i] = True
        # if use_HV:
        #     cost = check_cost_type(np.zeros((len(grid_asses), 2)), grid_asses, label_partition[labels], "PerimeterRatio", consider)
        # else:
        #     cost = check_cost_type(np.zeros((len(grid_asses), 2)), grid_asses, label_partition[labels], "Triple", consider)
        # compactness = np.exp(-cost[1]/len(grid_asses))
        # convexity = 1-cost[2]/len(grid_asses)
        # print('comp', compactness, 'conv', convexity)
        #
        # # -------------------------------  all measures  -----------------------------
        #
        # score_dict = {'prox-auc20': auc20, 'prox-auc50': auc50}
        #
        # score_dict['time'] = end-start
        #
        # score_dict.update({'comp': compactness, 'conv': convexity})
        #
        # if info_before is not None:
        #     score_dict.update({'stab-shape': IoU_ours, 'stab-position': relative, 'stab-order': order_score, 'order_ratio': order_score/(order_cnt+1e-12)})
        # if confusion is not None:
        #     score_dict.update({'ambi': confusion_score})
        #
        # score_dict.update({"layout": {"asses": grid_asses, "part_labels": label_partition[labels], "confusion": confusion, "labels": labels, "info_before": info_before, "top_labels": top_partition[label_partition[labels]]}})
        #
        # name = "qap" + "_" + self.Ctrler.scenario + "_" + self.Ctrler.dataset
        # if self.Ctrler.scenario == "ans":
        #     name = name + "_" + str(self.Ctrler.sample_num)
        # if self.Ctrler.select_method != "square":
        #     name = name + "_" + self.Ctrler.select_method
        # if use_HV:
        #     name = name + "_HV"
        #
        # if self.Ctrler.scenario == "dendroans":
        #     name = name + "_" + str(self.Ctrler.px)+"px.pkl"
        # else:
        #     name = name + ".pkl"
        #
        # if self.Ctrler.scenario == "ans":
        #     if not os.path.exists(str(self.Ctrler.sample_num)):
        #         os.makedirs(str(self.Ctrler.sample_num))
        #     name = str(self.Ctrler.sample_num) + "/" + name
        #
        # if os.path.exists(name):
        #     ans = load_pickle(name)
        # else:
        #     ans = {'top': [], 'zoom': []}
        # if info_before is None:
        #     ans['top'].append(score_dict)
        # else:
        #     ans['zoom'].append(score_dict)
        # save_pickle(ans, name)
        # print("save", name, N)

        return grid_asses, square_len, partition_labels, partition_info, top_partition, confusion

    # X_feature: [num X f]
    # labels: [num]
    # true_id: [num], 元素的真实id
    # info_before: dict, 上层元素信息
    #   ["grid_asses"]: [N_before], 上一层的layout
    #   ["X_embedded"]: [num_before X 2], 上一层元素的tsne
    #   ["selected"]: [N_s X 2], zoom时选中的元素在当前层与上层的序号(0..num-1与0..num_before-1)
    #   ["sampled_id"]: [num_before], 上层元素的真实id
    # tsne_total: [np.ndarray N X 2] 预处理得到的全部数据的tsne，方便全局第一步快速生成
    # top_labels: [num], tree cut 更高一层的 label
    # filter_labels: sample较少的label，作为散点
    def fit(self, X_feature, labels, true_id, hierarchy, info_before=None, tsne_total=None, top_labels=None, filter_labels=None, confs_hierarchy=None, scale=1/2, shres=0.8):
        if top_labels is None:
            top_labels = labels
        time1 = time.time()
        time2 = time.time()
        grid_asses, grid_size, partition, partition_info, top_part, confusion = self.grid(labels, X_feature, true_id, hierarchy, top_labels, filter_labels, info_before, None, scale, confs_hierarchy=confs_hierarchy, shres=shres)
        time3 = time.time()
        # print("tsne time: ", time2 - time1)
        print("grid time: ", time3 - time2)

        num = labels.shape[0]
        square_len = math.ceil(np.sqrt(num))
        X_embedded = get_layout_embedded(grid_asses, square_len)

        return X_embedded, grid_asses, grid_size, partition, partition_info, top_part, confusion

# if __name__ == "__main__":
#     X = np.random.rand(500, 128)
#     labels = np.random.randint(10, size=500)
