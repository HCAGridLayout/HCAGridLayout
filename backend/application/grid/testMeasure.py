import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import application.grid.gridlayoutOpt as gridlayoutOpt
from shapely import Point, distance
import time

def testNeighbor(a, b, check=None, kind="all", same_kind=False, k=20, arg_a=None, arg_b=None, return_arg=False):
    start=time.time()
    if check is None:
        check = np.ones(a.shape[0], dtype='bool')
    if arg_a is None or arg_b is None:
        order = np.arange(a.shape[0], dtype='int')
        np.random.seed(5)
        np.random.shuffle(order)
        dist_a = cdist(a, a[order], "euclidean")
        dist_b = cdist(b, b[order], "euclidean")
        arg_a = order[np.argsort(dist_a, axis=1)]
        arg_b = order[np.argsort(dist_b, axis=1)]
        # print(arg_a)
        # print(arg_b)
    # print("dist time", time.time()-start)
    ret = np.array(gridlayoutOpt.checkNeighbor(arg_a, arg_b, k, check, kind, same_kind))

    # print("neighbor time", time.time()-start)
    
    # print(ret)
    if return_arg:
        return ret, arg_a, arg_b
    return ret


def testEpsilonNeighbor(a, b, check=None, kind="all", same_kind=False, max_e=0.1, k=20):
    # print("get Epsilon Neighbor")
    if check is None:
        check = np.ones(a.shape[0], dtype='bool')
    order = np.arange(a.shape[0], dtype='int')
    np.random.seed(5)
    np.random.shuffle(order)
    dist_a = cdist(a, a[order], "euclidean")
    dist_b = cdist(b, b[order], "euclidean")
    arg_a = order[np.argsort(dist_a, axis=1)]
    arg_b = order[np.argsort(dist_b, axis=1)]
    real_dist_a = dist_a.copy()
    cv_order = order.copy()
    for i in range(order.shape[0]):
        cv_order[order[i]] = i
    for i in range(real_dist_a.shape[0]):
        real_dist_a[i] = real_dist_a[i][cv_order]
    # print(real_dist_a)
    # print(arg_a)
    # print(arg_b)
    ret0 = gridlayoutOpt.checkENeighbor(arg_a, real_dist_a, arg_b, max_e, k, check, kind, same_kind)
    ret1 = []
    ret2 = []
    for i in range(int(len(ret0)/2)):
        ret1.append(ret0[i*2])
        ret2.append(ret0[i*2+1])
    # print(ret1, ret2)
    return ret1, ret2

def checkXYOrder(full_grids, labels, grid_asses_bf, selected, selected_bf, if_confuse=None):
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

    square_len_bf = round(np.sqrt(grid_asses_bf.shape[0]))
    grid_embedded_bf = get_layout_embedded(grid_asses_bf, square_len_bf)

    order_score = 0
    order_cnt = 0
    for label in range(labels.max()+1):
        for i in range(len(selected)):
            if if_confuse is not None and if_confuse[selected[i]]:
                continue
            if labels[selected[i]] != label:
                continue
            for j in range(len(selected)):
                if if_confuse is not None and if_confuse[selected[j]]:
                    continue
                if labels[selected[i]] != labels[selected[j]]:
                    continue
                if grid_embedded_bf[selected_bf[i]][0] < grid_embedded_bf[selected_bf[j]][0]:
                    order_cnt += 1
                    if full_grids[selected[i]][0] >= full_grids[selected[j]][0]:
                        order_score += 1
                if grid_embedded_bf[selected_bf[i]][1] < grid_embedded_bf[selected_bf[j]][1]:
                    order_cnt += 1
                    if full_grids[selected[i]][1] >= full_grids[selected[j]][1]:
                        order_score += 1
        print(label, order_score)


    return order_score, order_cnt

def checkConfusion(full_grids, labels, confusion):
    import math
    maxLabel = labels.max() + 1
    full_D = cdist(full_grids, full_grids, "euclidean")
    N = len(full_grids)
    num = len(labels)
    square_len = round(np.sqrt(N))
    near = np.zeros((maxLabel, maxLabel), dtype='bool')

    labels_idx = {}
    for i in range(maxLabel):
        labels_idx[i] = (labels == i)
        # print(labels_idx[i].sum())
        labels_idx[i] = np.arange(len(labels), dtype='int')[labels_idx[i]]

    confuse_dist_max = np.zeros(maxLabel)
    for lb in range(maxLabel):
        idx = labels_idx[lb]
        confuse_dist_max[lb] = full_D[np.ix_(idx, idx)].max()

    confuse_dist = np.zeros((maxLabel, N))
    for lb in range(maxLabel):
        confuse_dist[lb] = full_D[:, labels_idx[lb]].min(axis=1)
    confuse_dist2 = np.zeros((maxLabel, N))
    confuse_dist3 = confuse_dist.copy()
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
            if min_dist > eps + tiny:
                # confuse_dist3[lb2][idx] = np.sqrt(len(labels_idx[lb])/math.pi)/square_len
                confuse_dist3[lb2][idx] = confuse_dist_max[lb]
            else:
                near[lb][lb2] = True

    conf = confusion['conf']
    confuse_class = confusion['confuse_class']
    score = 0
    score_near = 0
    for now_id in range(num):
        for j in range(min(3, maxLabel)):
            if near[labels[now_id]][confuse_class[now_id][j]]:
                score_near += conf[now_id][confuse_class[now_id][j]] * confuse_dist3[confuse_class[now_id][j]][now_id] ** 2
            if True:
                # score += conf[now_id][confuse_class[now_id][j]] * confuse_dist[confuse_class[now_id][j]][now_id] ** 2
                # score += conf[now_id][confuse_class[now_id][j]] * confuse_dist2[confuse_class[now_id][j]][now_id] ** 2
                score += conf[now_id][confuse_class[now_id][j]] * confuse_dist3[confuse_class[now_id][j]][now_id] ** 2

    print('score near', score_near)
    return score

def checkShape(grid_asses, labels, square_len, shapes, dtype="IoU"):
    checkP = 1
    info = {}
    num = len(labels)
    for i in range(square_len):
        for j in range(square_len):
            gid = i*square_len+j
            if grid_asses[gid] >= num or labels[grid_asses[gid]] < 0:
                continue
            lb = labels[grid_asses[gid]]
            if lb not in info:
                info[lb] = {"TP": 0, "FP": 0, "dist": 0}
            if lb not in shapes or shapes[lb].is_empty:
                continue
            for ii in range(checkP):
                for jj in range(checkP):
                    x = (i+1/checkP/2+ii/checkP)/square_len
                    y = (j+1/checkP/2+jj/checkP)/square_len
                    p = Point([x, y])
                    if shapes[lb].covers(p):
                        info[lb]["TP"] += 1/square_len/square_len/checkP/checkP
                    else:
                        # print(lb, x, y, i, j, ii, jj)
                        info[lb]["FP"] += 1/square_len/square_len/checkP/checkP
                        info[lb]["dist"] += distance(shapes[lb], p)/square_len/square_len/checkP/checkP
    FP = 0
    TP = 0
    TN = 0
    dist = 0
    IoU = 0
    for lb in info:
        FP += info[lb]["FP"]
        TP += info[lb]["TP"]
        if lb in shapes:
            TN += shapes[lb].area-info[lb]["TP"]
        dist += info[lb]["dist"]
    print(info)
    IoU = TP/(FP+TP+TN)
    if dtype == "IoU":
        return IoU
    if dtype == "dist":
        return dist
    if dtype == "TP":
        return TP
    if dtype == "FP":
        return FP
    if dtype == "TN":
        return TN
    return 0

def checkShapeAndPosition(grid_asses, labels, square_len, shapes):
    checkP = 1
    info = {}
    grid_shapes = {}
    num = len(labels)

    for i in range(square_len):
        for j in range(square_len):
            gid = i*square_len+j
            if grid_asses[gid] >= num or labels[grid_asses[gid]] < 0:
                continue
            lb = labels[grid_asses[gid]]
            if lb not in grid_shapes:
                grid_shapes[lb] = []
            grid_shapes[lb].append([i, j])

    from shapely.affinity import translate, affine_transform

    for lb in grid_shapes:
        if lb not in shapes or shapes[lb].is_empty:
            continue
        if lb not in info:
            info[lb] = {"TP": 0, "FP": 0, "dist": 0}
        centroid = ((np.array(grid_shapes[lb])+0.5)/square_len).mean(axis=0)
        old_centroid = np.array([shapes[lb].centroid.x, shapes[lb].centroid.y])
        info[lb]['centroid'] = centroid
        info[lb]['old_centroid'] = old_centroid

        ratio = np.sqrt(len(grid_shapes[lb])/(shapes[lb].area+1e-12))/square_len
        shape = affine_transform(shapes[lb], [ratio, 0, 0, ratio, 0, 0])
        shape = translate(shape, xoff=centroid[0]-shape.centroid.x, yoff=centroid[1]-shape.centroid.y)

        for g in grid_shapes[lb]:
            i, j = g[0], g[1]
            for ii in range(checkP):
                for jj in range(checkP):
                    x = (i + 1 / checkP / 2 + ii / checkP) / square_len
                    y = (j + 1 / checkP / 2 + jj / checkP) / square_len
                    p = Point([x, y])
                    if shape.covers(p):
                        info[lb]["TP"] += 1 / square_len / square_len / checkP / checkP
                    else:
                        # print(lb, x, y, i, j, ii, jj)
                        info[lb]["FP"] += 1 / square_len / square_len / checkP / checkP
                        info[lb]["dist"] += distance(shape, p) / square_len / square_len / checkP / checkP

    FP = 0
    TP = 0
    TN = 0
    for lb in info:
        FP += info[lb]["FP"]
        TP += info[lb]["TP"]
        TN += info[lb]["FP"]
    print(info)
    IoU = TP/(FP+TP+TN)

    rela = 0
    for lb in info:
        for lb2 in info:
            if lb<lb2:
                r1 = info[lb]['centroid']-info[lb2]['centroid']
                r2 = info[lb]['old_centroid']-info[lb2]['old_centroid']
                r = r1 - r2
                rela += np.sqrt(r[0]**2+r[1]**2)*len(grid_shapes[lb])*len(grid_shapes[lb2]) / (square_len**4)

    return IoU, rela


# a = np.array([[0, 1, 3], [1, 0, 4], [3, 4, 0]])
# b = np.array([[0, 1, 4], [1, 0, 3], [4, 3, 0]])
# print(testNeighboor(a, b))