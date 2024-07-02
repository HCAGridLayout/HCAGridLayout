from application.data.port import Port
import numpy as np
import random
import os
import json
from scipy.spatial.distance import cdist
import time
import application.grid.gridlayoutOpt as gridlayoutOpt
from application.utils.pickle import *
from collections import Counter

def getCompactness(embedded, labels, width):
    import math
    from collections import Counter
    labels = np.array(labels)
    square_len = math.ceil(np.sqrt(len(embedded)))
    N = square_len*square_len
    labels_cnt = Counter(labels)
    ans = 0
    for label in labels_cnt:
        idx = np.array(labels == label)
        center = embedded[idx].mean(axis=0)
        ans += np.power(cdist([center], embedded[idx], "eu"), 2).sum()
    return ans*(1/square_len/width)**2/N

def testNeighbor(a, b, maxk=50, labels=None, type='all'):
    start = time.time()
    order = np.arange(a.shape[0], dtype='int')
    np.random.seed(5)
    np.random.shuffle(order)
    dist_a = cdist(a, a[order], "euclidean")
    dist_b = cdist(b, b[order], "euclidean")
    arg_a = order[np.argsort(dist_a, axis=1)]
    arg_b = order[np.argsort(dist_b, axis=1)]

    # print("dist time", time.time()-start)
    nn = len(a)
    p1 = np.zeros(maxk)
    p2 = np.zeros(maxk)
    if type == 'cross':
        for k in range(maxk):
            for i in range(nn):
                diff = labels[arg_a[i]] != labels[i]
                diff2 = labels[arg_b[i]] != labels[i]
                # p1[k] += len(set(arg_a[i][diff][:k+1]).intersection(set(arg_b[i][diff2][:k+1])))
                p1[k] += (labels[np.array(list(set(arg_a[i][:k + 2]).intersection(set(arg_b[i][:k + 2]))))] !=
                          labels[i]).sum()
                p2[k] += 1
    else:
        for k in range(maxk):
            for i in range(nn):
                p1[k] += len(set(arg_a[i][:k + 2]).intersection(set(arg_b[i][:k + 2]))) - 1
                p2[k] += 1
    ret = p1 / p2

    if labels is not None:
        cnt = 0
        for i in range(nn):
            cnt += (labels[arg_a[i][:maxk]] == labels[i]).sum()
        print(cnt, maxk * nn - cnt)

    return ret

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

def testNeighbor_full(a, b, labels):
    avg_knnp = testNeighbor(a, b)

    # from collections import Counter
    # avg_knnp = 0
    # labels = np.array(labels)
    # label_list = Counter(labels)
    # for label in label_list:
    #     idx = np.arange(len(labels))[labels == label]
    #     knnp = testNeighbor(a[idx], b[idx])
    #     avg_knnp += knnp * len(idx)
    # avg_knnp /= len(labels)

    auc50, auc20 = AUC(avg_knnp)
    return avg_knnp, auc50, auc20

def checkShapeAndPositionFree(embedded, labels, shapes, width):
    import math
    from shapely import Point, distance
    checkP = 1
    info = {}
    grid_shapes = {}
    num = len(labels)

    square_len = math.ceil(np.sqrt(len(embedded)))

    for id in range(len(embedded)):
        lb = labels[id]
        if lb not in grid_shapes:
            grid_shapes[lb] = []
        grid_shapes[lb].append(embedded[id])

    from shapely.affinity import translate, affine_transform

    for lb in grid_shapes:
        if lb not in shapes or shapes[lb].is_empty:
            continue
        if lb not in info:
            info[lb] = {"TP": 0, "FP": 0, "dist": 0}
        centroid = ((np.array(grid_shapes[lb])+0.5*width)/width/square_len).mean(axis=0)
        old_centroid = np.array([shapes[lb].centroid.x, shapes[lb].centroid.y])
        info[lb]['centroid'] = centroid
        info[lb]['old_centroid'] = old_centroid

        ratio = np.sqrt(len(grid_shapes[lb])/(shapes[lb].area+1e-12))/square_len
        shape = affine_transform(shapes[lb], [ratio, 0, 0, ratio, 0, 0])
        shape = translate(shape, xoff=centroid[0]-shape.centroid.x, yoff=centroid[1]-shape.centroid.y)

        for g in grid_shapes[lb]:
            i, j = g[0]/width, g[1]/width
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

def checkXYOrderFree(embedded, labels, embedded_bf, selected, selected_bf, if_confuse=None):

    from collections import Counter
    labels = np.array(labels)
    labels_cnt = Counter(labels)

    order_score = 0
    order_cnt = 0
    for label in labels_cnt:
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
                if embedded_bf[selected_bf[i]][0] < embedded_bf[selected_bf[j]][0]:
                    order_cnt += 1
                    if embedded[selected[i]][0] >= embedded[selected[j]][0]:
                        order_score += 1
                if embedded_bf[selected_bf[i]][1] < embedded_bf[selected_bf[j]][1]:
                    order_cnt += 1
                    if embedded[selected[i]][1] >= embedded[selected[j]][1]:
                        order_score += 1
        # print(label, order_score)

        # plt.clf()
        # for i in range(len(selected)):
        #     if if_confuse is not None and if_confuse[selected[i]]:
        #         continue
        #     if labels[selected[i]] != label:
        #         continue
        #     plt.text(embedded_bf[selected_bf[i]][1], embedded_bf[selected_bf[i]][0], str(i), color="b")
        # plt.savefig("order0.png")
        #
        # plt.clf()
        # for i in range(len(selected)):
        #     if if_confuse is not None and if_confuse[selected[i]]:
        #         continue
        #     if labels[selected[i]] != label:
        #         continue
        #     plt.text(embedded[selected[i]][1], embedded[selected[i]][0], str(i), color="b")
        # plt.savefig("order1.png")

    return order_score, order_cnt

def checkBetweenClusterStability(dendro_embedded, top_part, part_labels, dendro_before, info_before, width):
    N2 = dendro_before['embedded'].shape[0]
    tmp_labels2 = np.ones(N2, dtype='int') * (-1)
    tmp_labels2[info_before['selected_bf']] = top_part[part_labels[info_before['selected']]]
    is_selected = np.zeros(N2, dtype='bool')
    is_selected[info_before['selected_bf']] = True

    zoom_partition_map = {}
    for p in top_part:
        idx = (top_part[part_labels[info_before['selected']]] == p)
        if idx.sum() > 0:
            count = Counter(dendro_before['part_labels'][np.array(info_before['selected_bf'])[idx]])
            zoom_partition_map[p] = max(count, key=lambda x: count[x])
    tmp_min2 = dendro_before['embedded'].max(axis=0)
    tmp_max2 = dendro_before['embedded'].min(axis=0)
    major_points = {}
    for id in range(len(dendro_before['embedded'])):
        if is_selected[id]:
            lb = tmp_labels2[id]
            if zoom_partition_map[lb] != dendro_before['part_labels'][id]:
                continue
            if lb not in major_points:
                major_points[lb] = []
            now_embedded = dendro_before['embedded'][id]
            tmp_min2 = np.minimum(tmp_min2, now_embedded)
            tmp_max2 = np.maximum(tmp_max2, now_embedded+width)
            major_points[lb].append(now_embedded)
            major_points[lb].append(now_embedded+np.array([0, width]))
            major_points[lb].append(now_embedded+np.array([width, 0]))
            major_points[lb].append(now_embedded+width)
    for lb in major_points:
        major_points[lb] = (np.array(major_points[lb]) - tmp_min2) / (tmp_max2 - tmp_min2)

    from application.grid.PowerDiagram import get_graph_from_coords

    shapes = get_graph_from_coords(np.zeros((len(major_points), 0, 2, 2)), graph_type="hull", major_points=major_points)

    IoU_ours, relative = checkShapeAndPositionFree(dendro_embedded, top_part[part_labels], shapes, width)
    print("IoU", IoU_ours)
    print('stab-position', relative)

    tmp_embedded2 = np.array(dendro_before['embedded']).astype('float64')
    tmp_embedded2[info_before['selected_bf']] -= tmp_embedded2[info_before['selected_bf']].min(axis=0) - 1 / 2 * width
    tmp_embedded2[info_before['selected_bf']] /= tmp_embedded2[info_before['selected_bf']].max(axis=0) + 1 / 2 * width

    maxLabel = part_labels.max()+1
    caled = np.zeros(maxLabel, dtype='bool')

    centers = np.zeros((maxLabel, 2))
    for p in range(maxLabel):
        idx = (part_labels[info_before['selected']] == p)
        # print(p, tmp_embedded2[info_before['selected_bf']][idx])
        if idx.sum() > 0:
            centers[p] = tmp_embedded2[info_before['selected_bf']][idx].mean(axis=0)
            caled[p] = True

    caled_list = np.arange(maxLabel, dtype='int')[caled]
    grid_embedded = np.array(dendro_embedded).astype('float64')

    grid_embedded -= grid_embedded.min(axis=0) - 1 / 2 * width
    grid_embedded /= grid_embedded.max(axis=0) + 1 / 2 * width

    centers2 = np.zeros((maxLabel, 2))

    p_size = np.zeros(maxLabel)
    for p in caled_list:
        idx = (part_labels == p)
        p_size[p] = idx.sum()
        centers2[p] = np.array(grid_embedded[idx]).mean(axis=0)

    rela2 = 0
    for p1 in caled_list:
        for p2 in caled_list:
            # if p1<p2:
            r1 = centers[p1] - centers[p2]
            r2 = centers2[p1] - centers2[p2]
            r = r1 - r2
            rela2 += np.sqrt(r[0] ** 2 + r[1] ** 2) * p_size[p1] * p_size[p2] / (len(part_labels) ** 2)


    return IoU_ours, rela2

def checkConfusionFree(full_grids, labels, confusion, min_dist):

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
        for j in range(1, min(3, maxLabel)):
            # if near[labels[now_id]][confuse_class[now_id][j]]:
            if True:
                # score += conf[now_id][confuse_class[now_id][j]] * confuse_dist2[confuse_class[now_id][j]][now_id] ** 2
                score2 += conf[now_id][confuse_class[now_id][j]] * confuse_dist3[confuse_class[now_id][j]][now_id] ** 2

    return score2

for dataset_name in ["cifar100", "imagenet1k", "inat2021"]:
    dataset = dataset_name
    if dataset_name == 'cifar100':
        dataset = 'cifar100_for_dendromap'
    for image_size in [30, 20, 16]:
        if os.path.exists('dendromap_dendroans_' + dataset + '_' + str(image_size) + 'px.pkl'):
            continue
        port = Port(1600, {"use_HV": True, "use_conf": True, "scenario": "dendroans", "select_method": "square", "method": "qap"})
        port.load_dataset(dataset) #'MNIST''cifar'
        select_type = 'square'

        if dataset_name == "cifar100":
            mini_samples = None
        elif dataset_name == "imagenet1k":
            mini_samples = np.load("datasets/imagenet1k/imagenet1k_minisamples.npy")
        elif dataset_name == "inat2021":
            mini_samples = np.load("datasets/inat2021/inat2021_minisamples.npy")

        with open("dendromap_step_"+dataset_name+"_"+str(image_size)+"px.json", 'r') as f:
            dendromap_step = json.load(f)

        # print(len(dendromap_step))

        gridlayout_stack = []
        dendromap_stack = []
        for i in range(len(dendromap_step)):
            dendro = dendromap_step[i]
            if dendro["method"] == "top":
                gridlayout_stack = []
                sampled_id = []
                dendro_embedded_map = {}
                for item in dendro["samples"]:
                    id = int(item["id"])
                    if mini_samples is not None:
                        id = mini_samples[id]
                    sampled_id.append(id)
                    dendro_embedded_map[id] = [int(item["x"]), int(item["y"])]
                sampled_id = np.array(sampled_id)

                new_gridlayout = port.top_gridlayout(pre_sampled_id=sampled_id)
                gridlayout_stack.append(new_gridlayout)

                dendro_embedded = []
                for id in new_gridlayout["sample_ids"]:
                    if id in dendro_embedded_map:
                        dendro_embedded.append(dendro_embedded_map[id])
                dendro_embedded = np.array(dendro_embedded)
                avg_knnp, auc50, auc20 = testNeighbor_full(new_gridlayout['feature'], dendro_embedded, new_gridlayout['labels'])
                print("auc20", auc20, "auc50", auc50)
                compactness = np.exp(-getCompactness(dendro_embedded, new_gridlayout['part_labels'], image_size))
                # convexity = 1-gridlayoutOpt.getConvexityForTriplesFree(dendro_embedded, new_gridlayout['part_labels'], image_size)
                convexity = 1-gridlayoutOpt.getConvexityForPerimeterFree(dendro_embedded, new_gridlayout['part_labels'], image_size)
                print("compactness", compactness)
                print("convexity", convexity)
                if_confuse = None
                if new_gridlayout["confusion"] is not None:
                    if_confuse = new_gridlayout["confusion"]["if_confuse"]
                    confuse_idx = np.arange(len(if_confuse))[if_confuse]
                    confuse_class = new_gridlayout["confusion"]["confuse_class"]
                if new_gridlayout["confusion"] is not None:
                    # from application.grid.testMeasure import checkConfusion
                    import math
                    tmp_square_len = math.ceil(np.sqrt(len(dendro_embedded)))
                    confusion_score = checkConfusionFree(dendro_embedded/tmp_square_len/image_size, new_gridlayout['part_labels'], new_gridlayout["confusion"], 1/tmp_square_len)
                    print("confusion score", confusion_score)

                dendromap_stack.append({'embedded': dendro_embedded, 'labels': new_gridlayout['labels'],
                                        'part_labels': new_gridlayout['part_labels'], "top_part": new_gridlayout['top_part'],
                                        'info_before': new_gridlayout['info_before']})

                score_dict = {'prox-auc20': auc20, 'prox-auc50': auc50}
                score_dict.update({'comp': compactness, 'conv': convexity})
                if new_gridlayout["confusion"] is not None:
                    score_dict.update({'ambi': confusion_score})
                if os.path.exists("dendromap_dendroans_"+dataset+"_"+str(image_size)+"px.pkl"):
                    ans = load_pickle("dendromap_dendroans_"+dataset+"_"+str(image_size)+"px.pkl")
                else:
                    ans = {'top': [], 'zoom': []}
                ans['top'].append(score_dict)
                save_pickle(ans, "dendromap_dendroans_"+dataset+"_"+str(image_size)+"px.pkl")


            elif dendro["method"] == "zoomin":
                gridlayout_bf = gridlayout_stack[-1]

                index_dict = {}
                for index in range(len(gridlayout_stack[-1]["sample_ids"])):
                    index_dict[gridlayout_stack[-1]["sample_ids"][index]] = index
                sampled_id = []
                dendro_embedded_map = {}
                selected = []
                for item in dendro["samples"]:
                    id = int(item["id"])
                    if mini_samples is not None:
                        id = mini_samples[id]
                    sampled_id.append(id)
                    dendro_embedded_map[id] = [int(item["x"]), int(item["y"])]
                    if id in index_dict:
                        selected.append(index_dict[id])
                sampled_id = np.array(sampled_id)
                selected = np.array(selected)

                new_gridlayout = port.layer_gridlayout(gridlayout_bf['index'], selected, pre_sampled_id=sampled_id)
                gridlayout_stack.append(new_gridlayout)

                dendro_embedded = []
                for id in new_gridlayout["sample_ids"]:
                    if id in dendro_embedded_map:
                        dendro_embedded.append(dendro_embedded_map[id])
                dendro_embedded = np.array(dendro_embedded)
                avg_knnp, auc50, auc20 = testNeighbor_full(new_gridlayout['feature'], dendro_embedded, new_gridlayout['labels'])
                print("auc20", auc20, "auc50", auc50)
                compactness = np.exp(-getCompactness(dendro_embedded, new_gridlayout['part_labels'], image_size))
                # convexity = 1-gridlayoutOpt.getConvexityForTriplesFree(dendro_embedded, new_gridlayout['part_labels'], image_size)
                convexity = 1-gridlayoutOpt.getConvexityForPerimeterFree(dendro_embedded, new_gridlayout['part_labels'], image_size)
                print("compactness", compactness)
                print("convexity", convexity)
                IoU, rela = checkBetweenClusterStability(dendro_embedded, new_gridlayout['top_part'], new_gridlayout['part_labels'], dendromap_stack[-1], new_gridlayout['info_before'], image_size)
                if_confuse = None
                if new_gridlayout["confusion"] is not None:
                    if_confuse = new_gridlayout["confusion"]["if_confuse"]
                    confuse_idx = np.arange(len(if_confuse))[if_confuse]
                    confuse_class = new_gridlayout["confusion"]["confuse_class"]
                order_score, order_cnt = checkXYOrderFree(dendro_embedded, new_gridlayout['labels'], dendromap_stack[-1]["embedded"], new_gridlayout['info_before']['selected'], new_gridlayout['info_before']['selected_bf'], if_confuse)
                print("order", order_score, order_score/order_cnt)
                if new_gridlayout["confusion"] is not None:
                    # from application.grid.testMeasure import checkConfusion
                    import math
                    tmp_square_len = math.ceil(np.sqrt(len(dendro_embedded)))
                    confusion_score = checkConfusionFree(dendro_embedded/tmp_square_len/image_size, new_gridlayout['part_labels'], new_gridlayout["confusion"], 1/tmp_square_len)
                    print("confusion score", confusion_score)

                dendromap_stack.append({'embedded': dendro_embedded, 'labels': new_gridlayout['labels'],
                                        'part_labels': new_gridlayout['part_labels'], "top_part": new_gridlayout['top_part'],
                                        'info_before': new_gridlayout['info_before']})

                score_dict = {'prox-auc20': auc20, 'prox-auc50': auc50}
                score_dict.update({'comp': compactness, 'conv': convexity})
                score_dict.update({'stab-shape': IoU, 'stab-position': rela, 'stab-order': order_score, 'order_ratio': order_score/order_cnt})
                if new_gridlayout["confusion"] is not None:
                    score_dict.update({'ambi': confusion_score})
                if os.path.exists("dendromap_dendroans_"+dataset+"_"+str(image_size)+"px.pkl"):
                    ans = load_pickle("dendromap_dendroans_"+dataset+"_"+str(image_size)+"px.pkl")
                else:
                    ans = {'top': [], 'zoom': []}
                ans['zoom'].append(score_dict)
                save_pickle(ans, "dendromap_dendroans_"+dataset+"_"+str(image_size)+"px.pkl")

            else:
                gridlayout_bf = gridlayout_stack[-1]
                _ = port.zoom_out_gridlayout(gridlayout_bf['index'])
                gridlayout_stack.pop()
                dendromap_stack.pop()
