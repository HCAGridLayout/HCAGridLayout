import numpy as np
import time
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment
from scipy.stats import entropy
import matplotlib.pyplot as plt

def checkConfusion(full_grids, labels, confusion):
    maxLabel = labels.max()+1
    full_D = cdist(full_grids, full_grids, "euclidean")
    N = len(full_grids)
    num = len(labels)
    square_len = round(np.sqrt(N))

    labels_idx = {}
    for i in range(maxLabel):
        labels_idx[i] = (labels == i)
        # print(labels_idx[i].sum())
        labels_idx[i] = np.arange(len(labels), dtype='int')[labels_idx[i]]

    confuse_dist = np.zeros((maxLabel, N))
    for lb in range(maxLabel):
        confuse_dist[lb] = full_D[:, labels_idx[lb]].min(axis=1)
    confuse_dist2 = np.zeros((maxLabel, N))
    for lb in range(maxLabel):
        idx = labels_idx[lb]
        for lb2 in range(maxLabel):
            tmp = confuse_dist[lb2][idx]
            sort_id = np.argsort(tmp)
            tiny = 1e-8
            eps = 1 / square_len
            min_dist = tmp[sort_id[0]]
            if min_dist > eps + tiny:
                min_dist += eps
            new_id = (tmp <= min_dist + tiny)
            confuse_dist2[lb2][idx] = full_D[np.ix_(idx, idx[new_id])].min(axis=1)

    conf = confusion['conf']
    confuse_class = confusion['confuse_class']
    score = 0
    for now_id in range(len(labels)):
        for j in range(min(3, maxLabel)):
            # if near[labels[now_id]][confuse_class[now_id][j]]:
            if True:
                score += conf[now_id][confuse_class[now_id][j]] * confuse_dist2[confuse_class[now_id][j]][now_id] ** 2

    return score