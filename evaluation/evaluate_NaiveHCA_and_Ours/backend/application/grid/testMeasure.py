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

def checkShapeAndPosition(grid_asses, labels, square_len, shapes, with_info=False):
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

    if with_info:
        return IoU, rela, info

    return IoU, rela


# a = np.array([[0, 1, 3], [1, 0, 4], [3, 4, 0]])
# b = np.array([[0, 1, 4], [1, 0, 3], [4, 3, 0]])
# print(testNeighboor(a, b))

def checkPosition2(grid_asses, part_labels, square_len, info_before):
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

    if info_before is not None:

        N2 = info_before['grid_asses'].shape[0]
        tmp_embedded2 = get_layout_embedded(info_before['grid_asses'], round(np.sqrt(N2)))
        square_len2 = round(np.sqrt(N2))

        maxLabel = part_labels.max()+1
        caled = np.zeros(maxLabel, dtype='bool')

        tmp_embedded2 = get_layout_embedded(info_before['grid_asses'], round(np.sqrt(N2)))
        tmp_embedded2[info_before['selected_bf']] -= tmp_embedded2[info_before['selected_bf']].min(axis=0) - 1 / 2 / np.sqrt(N2)
        tmp_embedded2[info_before['selected_bf']] /= tmp_embedded2[info_before['selected_bf']].max(axis=0) + 1 / 2 / np.sqrt(N2)

        centers = np.zeros((maxLabel, 2))
        for p in range(maxLabel):
            idx = (part_labels[info_before['selected']] == p)
            # print(p, tmp_embedded2[info_before['selected_bf']][idx])
            if idx.sum() > 0:
                centers[p] = tmp_embedded2[info_before['selected_bf']][idx].mean(axis=0)
                caled[p] = True

        caled_list = np.arange(maxLabel, dtype='int')[caled]
        grid_embedded = get_layout_embedded(grid_asses, square_len)

        centers2 = np.zeros((maxLabel, 2))

        p_size = np.zeros(maxLabel)
        for p in caled_list:
            idx = np.arange(len(part_labels), dtype='int')[(part_labels == p)]
            p_size[p] = len(idx)
            centers2[p] = (np.array(grid_embedded[idx])+0.5/square_len).mean(axis=0)

        rela2 = 0
        for p1 in caled_list:
            for p2 in caled_list:
                # if p1<p2:
                    r1 = centers[p1]-centers[p2]
                    r2 = centers2[p1]-centers2[p2]
                    r = r1 - r2
                    rela2 += np.sqrt(r[0]**2+r[1]**2)*p_size[p1]*p_size[p2] / (square_len**4)

        return rela2

    return 0

def checkConfusionForDisconnectedShapes(full_grids, labels, confusion, min_dist):

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

    def dfs(node, graph, visited, connected_component):
        visited[node] = True
        connected_component.append(node)

        for neighbor in graph[node]:
            if not visited[neighbor]:
                dfs(neighbor, graph, visited, connected_component)

    def find_connected_components(edge_list):
        graph = {}
        # 构建无向图
        for i in range(len(labels)):
            graph[i] = []
        for edge in edge_list:
            node1, node2 = edge
            if node1 not in graph:
                graph[node1] = []
            if node2 not in graph:
                graph[node2] = []
            graph[node1].append(node2)
            graph[node2].append(node1)
        visited = {node: False for node in graph}
        connected_components = []
        for node in graph:
            if not visited[node]:
                connected_component = []
                dfs(node, graph, visited, connected_component)
                connected_components.append(connected_component)
        return connected_components

    edge_list = []
    for i in range(len(labels)):
        ls = np.arange(N, dtype='int')[full_D[i]<min_dist+1e-6]
        for j in ls:
            if j <= i or j >= len(labels):
                continue
            if labels[i] == labels[j] and (full_grids[i][0]-full_grids[j][0])**2+(full_grids[i][1]-full_grids[j][1])**2<min_dist**2+1e-12:
                edge_list.append([i, j])

    connected_components = find_connected_components(edge_list)
    if_connected = np.zeros(len(labels), dtype='bool')
    conn_dict = {}

    # for conn in connected_components:
    #     # if len(conn) <= 5:
    #     #     continue
    #     lb = labels[conn[0]]
    #     if lb not in conn_dict or len(conn_dict[lb]) < len(conn):
    #         conn_dict[lb] = conn
    # for lb in conn_dict:
    #     conn = conn_dict[lb]
    #     for id in conn:
    #         if_connected[id] = True

    for conn in connected_components:
        if len(conn) >= len(labels_idx[labels[conn[0]]])/3:
            for id in conn:
                if_connected[id] = True
        lb = labels[conn[0]]
        if lb not in conn_dict or len(conn_dict[lb]) < len(conn):
            conn_dict[lb] = conn
    for lb in conn_dict:
        conn = conn_dict[lb]
        for id in conn:
            if_connected[id] = True

    confuse_dist_max = np.zeros(maxLabel)
    for lb in range(maxLabel):
        idx = labels_idx[lb]
        idx = idx[if_connected[idx]]
        confuse_dist_max[lb] = full_D[np.ix_(idx, idx)].max()

    confuse_dist = np.zeros((maxLabel, N))
    for lb in range(maxLabel):
        idx = labels_idx[lb]
        if if_connected[idx].sum()!=0:
            tmp_idx = idx[if_connected[idx]]
        else:
            tmp_idx = idx
            print("???")
            exit(0)
        confuse_dist[lb] = full_D[:, tmp_idx].min(axis=1)

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
    score2 = 0
    for now_id in range(num):
        for j in range(min(3, maxLabel)):
            # if near[labels[now_id]][confuse_class[now_id][j]]:
            if True:
                # score += conf[now_id][confuse_class[now_id][j]] * confuse_dist2[confuse_class[now_id][j]][now_id] ** 2
                score2 += conf[now_id][confuse_class[now_id][j]] * confuse_dist3[confuse_class[now_id][j]][now_id] ** 2

    return score2

def check_cost_type(ori_embedded, row_asses, labels, type, consider, disconn=False):
    N = row_asses.shape[0]
    tmp_row_asses = np.array(gridlayoutOpt.Optimizer(0).checkCostForAll(ori_embedded, row_asses, labels, type, 1, 0, consider))
    if disconn and type == 'PerimeterRatio':
        def get_grids_and_labels(grid_asses, square_len, consider):
            ls = np.arange(square_len, dtype='int')
            grids = np.array([(i, j) for i in ls for j in ls])
            return grids[consider], labels[grid_asses[consider]]
        new_grids, new_labels = get_grids_and_labels(row_asses, np.sqrt(N), consider)
        tmp_row_asses[N + 2] = gridlayoutOpt.getConvexityForPerimeterFree(new_grids, new_labels, 1) * len(row_asses)

    new_cost = np.array([tmp_row_asses[N], tmp_row_asses[N + 1], tmp_row_asses[N + 2], tmp_row_asses[N + 3]])

    return new_cost