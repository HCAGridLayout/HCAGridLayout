import math
import os
import numpy as np
from scipy.spatial import Delaunay, ConvexHull
from shapely.geometry import Polygon, MultiPolygon, MultiPoint
from shapely.geometry import Point
import time
import pickle
from multiprocessing import Pool
from CGAL.CGAL_Kernel import *
from CGAL.CGAL_Triangulation_2 import *
from CGAL import CGAL_Convex_hull_2
from scipy.spatial.distance import cdist
# import gridlayoutOpt as gridlayoutOpt
import application.grid.gridlayoutOpt as gridlayoutOpt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor
import torch
import random
import matplotlib.pyplot as plt
import alphashape
# from colors import MyColorMap
from application.grid.colors import MyColorMap


def convexhull_area(positions, radii, selected_indices=None):
    if selected_indices is None:
        positions = np.array(positions)
        radii = np.array(radii)
    else:
        positions = np.array(positions)[selected_indices]
        radii = np.array(radii)[selected_indices]
    N = len(radii)
    APPROX_NUM = 50
    approx_points = []
    for i in range(N):
        pos = positions[i]
        r = radii[i]
        for i in range(APPROX_NUM):
            approx_points.append(
                [pos[0] + r * np.cos(np.pi * 2 * i / APPROX_NUM), pos[1] + r * np.sin(np.pi * 2 * i / APPROX_NUM)])
    convex_hull = ConvexHull(approx_points)
    # get the area of the convex_hull
    return convex_hull.volume


def modularity(vec):
    return np.sqrt(np.sum(vec ** 2))


def cellArea(cell):
    if isinstance(cell, Polygon):
        try:
            return cell.area
        except:
            return 0

    area = 0
    for k in range(len(cell)):
        p1 = cell[k]
        p2 = cell[(k + 1) % len(cell)]
        area += p1[0] * p2[1] - p2[0] * p1[1]
    area /= 2
    return abs(area)


def cellCentroid(cell):
    try:
        return np.array([cell.centroid.x, cell.centroid.y])
    except:
        x, y = 0, 0
        area = 0
        for k in range(len(cell)):
            p1 = cell[k]
            p2 = cell[(k + 1) % len(cell)]
            v = p1[0] * p2[1] - p2[0] * p1[1]
            area += v
            x += (p1[0] + p2[0]) * v
            y += (p1[1] + p2[1]) * v
        area *= 3
        if area == 0:
            return None
        return np.array([x / area, y / area])


def computePowerDiagramByCGAL(positions, weights=None, hull=None, withEdge=False):
    start = time.time()
    if weights is None:
        nonneg_weights = np.zeros(len(positions))
    else:
        nonneg_weights = weights - np.min(weights)

    rt = Regular_triangulation_2()

    v_handles = []
    for pos, w in zip(positions, nonneg_weights):
        v_handle = rt.insert(Weighted_point_2(Point_2(float(pos[0]), float(pos[1])), float(w)))
        v_handles.append(v_handle)

    control_point_set = [
        Weighted_point_2(Point_2(-10, -10), 0),
        Weighted_point_2(Point_2(10, -10), 0),
        Weighted_point_2(Point_2(10, 10), 0),
        Weighted_point_2(Point_2(-10, 10), 0)
    ]

    for cwp in control_point_set:
        rt.insert(cwp)

    # print("insert time", time.time()-start)

    cells = []
    edges = []

    hash = {}

    def point_same(x1, y1, x2, y2):
        tiny = 1e-10
        if (x1-x2)*(x1-x2)+(y1-y2)*(y1-y2) < tiny:
            return True
        return False

    def check_hash(x1, y1):
        index1 = (round(x1*300), round(y1*300))
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                index2 = (index1[0]+x, index1[1]+y)
                if index2 in hash:
                    for i in hash[index2]:
                        if point_same(positions[i, 0], positions[i, 1], x1, y1):
                            return i
        return -1

    def add_hash(x1, y1, id):
        index1 = (round(x1*300), round(y1*300))
        if index1 not in hash:
            hash.update({index1: []})
        hash[index1].append(id)

    for i in range(len(positions)):
        add_hash(positions[i, 0], positions[i, 1], i)

    def get_id(pt):
        # for i in range(len(positions)):
        #     if (pt.x() - positions[i, 0]) ** 2 + (pt.y() - positions[i, 1]) ** 2 <= 1e-10:
        #         return i
        # return -1
        return check_hash(pt.x(), pt.y())


    # print(dir(hull))
    # for i, handle in enumerate(v_handles):
    i = 0
    for handle in rt.finite_vertices():
        non_hidden_point = handle.point()
        while i < len(positions) and (non_hidden_point.x() - positions[i, 0]) ** 2 + (
                non_hidden_point.y() - positions[i, 1]) ** 2 > 1e-10:
            i += 1
            cells.append(Polygon([]))
        if i >= len(positions):
            break

        f = rt.incident_faces(handle)
        # print("handle time 0", time.time()-start)

        done = f.next()
        cell = []
        while True:
            face_circulator = f.next()
            wc = rt.weighted_circumcenter(face_circulator)
            x, y = wc.x(), wc.y()
            if (len(cell) == 0) or (((x - cell[-1][0]) ** 2 + (y - cell[-1][1]) ** 2 > 1e-10) and (
                    (x - cell[0][0]) ** 2 + (y - cell[0][1]) ** 2 > 1e-10)):
                cell.append((wc.x(), wc.y()))
            if face_circulator == done:
                break

        # print("handle time 1", time.time()-start)

        if withEdge:
            # v = rt.incident_vertices(handle)
            # done = v.next()
            # # print("------------------")
            # # print(handle.point().x(), handle.point().x())
            # while True:
            #     vertice_circulator = v.next()
            #     # print(vertice_circulator.point().x(), vertice_circulator.point().y())
            #     p = vertice_circulator.point()
            #     if vertice_circulator == done:
            #         break

            e = rt.incident_edges(handle)
            done = e.next()
            # print("handle time 2", time.time()-start)
            # print("------------------")
            # print(handle.point().x(), handle.point().x())
            cnt = 0
            while True:
                edge_circulator = e.next()
                seg = rt.segment(edge_circulator[0], edge_circulator[1])
                # print("handle time 2", cnt, time.time()-start)
                # seg2 = rt.dual(edge_circulator).get_Segment_2()

                # print(seg.source(), seg.target())
                # print(seg2.source(), seg2.target())
                a = get_id(seg.source())
                b = get_id(seg.target())
                if (a >= 0) and (b > a):
                    # x = seg2.target().x() - seg2.source().x()
                    # y = seg2.target().y() - seg2.source().y()
                    # z = math.sqrt(x*x+y*y)
                    # x /= z
                    # y /= z
                    edges.append((a, b))
                    # print(a, b)
                # print(seg2)
                # print("handle time 2", cnt, time.time()-start)
                cnt += 1
                if edge_circulator == done:
                    break
            # print("------------------")

        # print("handle time 3", time.time()-start)

        # ***************************************
        poly_cell = Polygon(cell)
        if hull is not None and not hull.contains(poly_cell):
            poly_cell = hull.intersection(poly_cell)
        cells.append(poly_cell)
        # **************************************
        i += 1
        # print("handle time 4", time.time()-start)

    if withEdge:
        # print(edges)
        return cells, edges
    return cells


def computeConvexHull(positions):
    point_set = []
    for pos in positions:
        point_set.append(Point_2(float(pos[0]), float(pos[1])))
    convex_hull = []
    CGAL_Convex_hull_2.convex_hull_2(point_set, convex_hull)
    cvp = [(v.x(), v.y()) for v in convex_hull]
    poly_hull = Polygon(cvp)
    return poly_hull


def grad_W(cells, caps):
    grad = []
    for cell, cap in zip(cells, caps):
        area = cellArea(cell)
        grad.append(cap - area)
    return np.array(grad)


def F(cells, sites, caps, weights):
    ret = 0
    for i, cell in enumerate(cells):
        area = cellArea(cell)
        cx, cy = sites[i, 0], sites[i, 1]
        if area > 0:
            centroid = cellCentroid(cell)
            ret += -2 * (cx * centroid[0] + cy * centroid[1]) * area + area * (cx ** 2 + cy ** 2)
        ret -= weights[i] * (area - caps[i])
    return ret


def find_W(positions, w0, caps, hull):
    # Maximize F
    half = 0.5
    # For L-BFGS
    m = 5

    cells = computePowerDiagramByCGAL(positions, w0, hull=hull)

    H0 = np.eye(len(w0))
    s, y, rho = [], [], []
    k = 1
    gk = -grad_W(cells, caps)
    dk = -H0.dot(gk)
    current_F = F(cells, positions, caps, w0)
    count = 0
    while True:
        count += 1
        n = 0
        mk = -1
        gk = -grad_W(cells, caps)
        if modularity(gk) < 1e-12:
            break
        while n < 20:
            new_w = w0 + (half ** n) * dk
            new_w = new_w - new_w.mean()
            new_cells = computePowerDiagramByCGAL(positions, new_w, hull=hull)
            new_F = F(new_cells, positions, caps, new_w)

            if not np.isnan(new_F) and new_F - current_F > 1e-12:
                mk = n
                break
            n += 1
        if mk < 0:
            break
        w = new_w
        cells = new_cells
        current_F = new_F

        sk = w - w0
        qk = -grad_W(cells, caps)
        yk = qk - gk
        s.append(sk)
        y.append(yk)
        rho.append(1 / (yk.T.dot(sk) + 1e-5))

        a = []
        for i in range(max(k - m, 0), k):
            alpha = rho[k - i - 1] * s[k - i - 1].T.dot(qk)
            qk = qk - alpha * y[k - i - 1]
            a.append(alpha)
        r = H0.dot(qk)
        for i in range(max(k - m, 0), k):
            beta = rho[i] * y[i].T.dot(r)
            r = r + s[i] * (a[k - i - 1] - beta)

        if rho[-1] > 0:
            dk = -r

        k += 1
        w0 = w

    return w0


def show_grid_tmp(row_asses, grid_labels, square_len, path='new.png', showNum=True, just_save=False):
    import matplotlib.pyplot as plt
    def highlight_cell(x, y, ax=None, **kwargs):
        rect = plt.Rectangle((x - .5, y - .5), 1, 1, fill=False, **kwargs)
        ax = ax or plt.gca()
        ax.add_patch(rect)
        return rect

    from .colors import MyColorMap
    cm = MyColorMap()

    data = []
    num = math.ceil(square_len)
    for i in range(num - 1, -1, -1):
        row = []
        for j in range(num):
            if row_asses[num * i + j]>=len(grid_labels) or grid_labels[row_asses[num * i + j]]==-1:
                row.append((1, 1, 1, 1))
            else:
                row.append(plt.cm.tab20(grid_labels[row_asses[num * i + j]]))
            # row.append(cm.color(grid_labels[row_asses[num * i + j]]))
        data.append(row)
    plt.cla()
    plt.imshow(data)
    for i in range(num - 1, -1, -1):
        for j in range(num):
            highlight_cell(i, j, color="white", linewidth=1)
    if showNum:
        for i in range(num):
            for j in range(num):
                text = plt.text(j, num - i - 1, row_asses[num * i + j], fontsize=7, ha="center", va="center",
                                color="w")
                # text = plt.text(j, num - i - 1, grid_labels[row_asses[num * i + j]], fontsize=7, ha="center", va="center",
                #                 color="w")
    plt.axis('off')
    plt.savefig(path)
    if not just_save:
        plt.show()


def merge_coords(coords_list):
    hash = {}
    edges = []

    def point_same(x1, y1, x2, y2, tiny=1e-10):
        if (x1-x2)*(x1-x2)+(y1-y2)*(y1-y2) < tiny:
            return True
        return False

    def id_edge_has_point(i, x1, y1, tiny=1e-10):
        x3, y3, x4, y4 = edges[i]['edge']
        if point_same(x1, y1, x3, y3, tiny):
            return True
        if point_same(x1, y1, x4, y4, tiny):
            return True
        return False

    def id_edge_same(i, x1, y1, x2, y2):
        x3, y3, x4, y4 = edges[i]['edge']
        tiny = 1e-10
        if (x1-x3)*(x1-x3)+(y1-y3)*(y1-y3)+(x2-x4)*(x2-x4)+(y2-y4)*(y2-y4) < tiny:
            return True
        if (x2-x3)*(x2-x3)+(y2-y3)*(y2-y3)+(x1-x4)*(x1-x4)+(y1-y4)*(y1-y4) < tiny:
            return True
        return False

    def check_hash(x1, y1, x2, y2):
        index1 = (round(x1*300), round(y1*300))
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                index2 = (index1[0]+x, index1[1]+y)
                if index2 in hash:
                    for i in hash[index2]:
                        if id_edge_same(i, x1, y1, x2, y2):
                            return i
        return -1

    def check_hash_not_same(x1, y1, x2, y2, tiny=1e-10):
        index1 = (round(x1*300), round(y1*300))
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                index2 = (index1[0]+x, index1[1]+y)
                if index2 in hash:
                    for i in hash[index2]:
                        if id_edge_has_point(i, x1, y1, tiny) and not id_edge_same(i, x1, y1, x2, y2) and not edges[i]['check']:
                            return i
        return -1

    def add_hash(x1, y1, x2, y2, id):
        index1 = (round(x1*300), round(y1*300))
        if index1 not in hash:
            hash.update({index1: []})
        hash[index1].append(id)
        index2 = (round(x2*300), round(y2*300))
        if index2 not in hash:
            hash.update({index2: []})
        hash[index2].append(id)

    for ce in range(len(coords_list)):
        coords = coords_list[ce]
        for i in range(coords.shape[0]-1):
            e = check_hash(coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1])
            if e != -1:
                edges[e]['cnt'] += 1
            else:
                edges.append({'edge': (coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1]), 'cnt': 1, 'check': True})
                id = len(edges) - 1
                add_hash(coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1], id)

    hash = {}
    for id in range(len(edges)):
        edge = edges[id]
        x1, y1, x2, y2 = edge['edge']
        if (edge['cnt'] == 1) and not point_same(x1, y1, x2, y2):
            edge['check'] = False
            add_hash(x1, y1, x2, y2, id)

    ret_coords = []
    ret_area = 0
    for id in range(len(edges)):
        if edges[id]['check']:
            continue
        first_edge = edges[id]['edge']
        first_id = id
        coords = []
        coords.append([first_edge[0], first_edge[1]])
        coords.append([first_edge[2], first_edge[3]])
        edges[first_id]['check'] = True
        circle_flag = True
        while not point_same(coords[0][0], coords[0][1], coords[-1][0], coords[-1][1]):
            id = check_hash_not_same(coords[-1][0], coords[-1][1], coords[-2][0], coords[-2][1])
            if id == -1:
                id = check_hash_not_same(coords[-1][0], coords[-1][1], coords[-2][0], coords[-2][1], 1e-9)
            if id == -1:
                circle_flag = False
                break
            x1, y1, x2, y2 = edges[id]['edge']
            if point_same(coords[-1][0], coords[-1][1], x1, y1):
                coords.append([x2, y2])
            else:
                coords.append([x1, y1])
            edges[id]['check'] = True
        if circle_flag and (len(coords)>2):
            area = Polygon(coords).area
            if area > ret_area:
                ret_area = area
                ret_coords = coords

    coords = np.array(ret_coords)
    if len(coords) == 0:
        coords = np.zeros((0, 2))
    return coords


def get_skeleton(label_coords, hull=None, withEdges=False):
    hash = {}
    points = []

    def point_same(x1, y1, x2, y2):
        tiny = 1e-10
        if (x1-x2)*(x1-x2)+(y1-y2)*(y1-y2) < tiny:
            return True
        return False

    def check_hash(x1, y1):
        index1 = (round(x1*300), round(y1*300))
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                index2 = (index1[0]+x, index1[1]+y)
                if index2 in hash:
                    for i in hash[index2]:
                        x2, y2 = points[i]['coord']
                        if point_same(x1, y1, x2, y2):
                            return i
        return -1

    def add_hash(x1, y1, id):
        index1 = (round(x1*300), round(y1*300))
        if index1 not in hash:
            hash.update({index1: []})
        hash[index1].append(id)

    label_coords_id = {}

    for lb in label_coords:
        coords = label_coords[lb]
        coords_id = []
        for i in range(coords.shape[0]-1):
            point = coords[i]
            x, y = point[0], point[1]
            p = check_hash(x, y)
            if p != -1:
                points[p]['cells'].append(lb)
            else:
                points.append({'coord': (x, y), 'cells': [lb]})
                id = len(points) - 1
                add_hash(x, y, id)
                p = id
            coords_id.append(p)
        if coords.shape[0] > 1:
            coords_id.append(coords_id[0])
        label_coords_id.update({lb: coords_id})

    tiny = 1e-10
    label_skeleton = {}

    check_edges = []
    for lb in label_coords_id:
        skeleton = []
        intersec = set([])
        start_coord = (0, 0)
        first_i = -1
        for i in range(len(label_coords_id[lb])-1):
            id = label_coords_id[lb][i]
            x, y = points[id]['coord']
            intersec = intersec.intersection(set(points[id]['cells']))
            next_id = label_coords_id[lb][i+1]
            last_id = label_coords_id[lb][i-1]
            if i==0:
                last_id = label_coords_id[lb][-2]
            flag = False
            if len(set(points[last_id]['cells']).intersection(set(points[id]['cells']))) < len(set(points[id]['cells'])):
                flag = True
            if len(set(points[next_id]['cells']).intersection(set(points[id]['cells']))) < len(set(points[id]['cells'])):
                flag = True
            if flag or (len(points[id]['cells']) > 2) or (len(points[id]['cells']) == 1) or (x > 1-tiny) or (x < tiny) or (y > 1-tiny) or (y < tiny) or ((hull is not None) and not hull.covers(Point(x, y))):
                if first_i == -1:
                    first_i = i
                skeleton.append([x, y])
                if len(intersec) == 2:
                    pair = []
                    for i in intersec:
                        pair.append(i)
                    # print(pair[0], pair[1], start_coord[0], start_coord[1], x, y)
                    dx = x - start_coord[0]
                    dy = y - start_coord[1]
                    dz = np.sqrt(dx*dx+dy*dy)
                    if dz > tiny:
                        dx /= dz
                        dy /= dz
                        check_edges.append((pair[0], pair[1], dx, dy, dz))
                start_coord = (x, y)
                intersec = set(points[id]['cells'])

        if first_i != -1 and len(intersec) > 0:
            for i in range(first_i+1):
                id = label_coords_id[lb][i]
                intersec = intersec.intersection(set(points[id]['cells']))
            id = label_coords_id[lb][first_i]
            x, y = points[id]['coord']
            if len(intersec) == 2:
                pair = []
                for i in intersec:
                    pair.append(i)
                # print(pair[0], pair[1], start_coord[0], start_coord[1], x, y)
                dx = x - start_coord[0]
                dy = y - start_coord[1]
                dz = np.sqrt(dx*dx+dy*dy)
                if dz > tiny:
                    dx /= dz
                    dy /= dz
                    check_edges.append((pair[0], pair[1], dx, dy, dz))

        if len(skeleton) > 1:
            skeleton.append(skeleton[0])
        skeleton = np.array(skeleton)
        if len(skeleton) == 0:
            skeleton = np.zeros((0, 2))
        label_skeleton.update({lb: skeleton})

    if withEdges:
        return label_skeleton, check_edges
    return label_skeleton


def check_merge_coords(coords_list):
    hash = {}
    edges = []

    def point_same(x1, y1, x2, y2):
        tiny = 1e-10
        if (x1-x2)*(x1-x2)+(y1-y2)*(y1-y2) < tiny:
            return True
        return False

    def id_edge_has_point(i, x1, y1):
        x3, y3, x4, y4 = edges[i]['edge']
        if point_same(x1, y1, x3, y3):
            return True
        if point_same(x1, y1, x4, y4):
            return True
        return False

    def id_edge_same(i, x1, y1, x2, y2):
        x3, y3, x4, y4 = edges[i]['edge']
        tiny = 1e-10
        if (x1-x3)*(x1-x3)+(y1-y3)*(y1-y3)+(x2-x4)*(x2-x4)+(y2-y4)*(y2-y4) < tiny:
            return True
        if (x2-x3)*(x2-x3)+(y2-y3)*(y2-y3)+(x1-x4)*(x1-x4)+(y1-y4)*(y1-y4) < tiny:
            return True
        return False

    def check_hash(x1, y1, x2, y2):
        index1 = (round(x1*300), round(y1*300))
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                index2 = (index1[0]+x, index1[1]+y)
                if index2 in hash:
                    for i in hash[index2]:
                        if id_edge_same(i, x1, y1, x2, y2):
                            return i
        return -1

    def add_hash(x1, y1, x2, y2, id):
        index1 = (round(x1*300), round(y1*300))
        if index1 not in hash:
            hash.update({index1: []})
        hash[index1].append(id)
        index2 = (round(x2*300), round(y2*300))
        if index2 not in hash:
            hash.update({index2: []})
        hash[index2].append(id)

    fa = np.arange(len(coords_list), dtype='int')

    def get_fa(u):
        if fa[u] != u:
            fa[u] = get_fa(fa[u])
        return fa[u]

    for ce in range(len(coords_list)):
        coords = coords_list[ce]
        for i in range(coords.shape[0]-1):
            e = check_hash(coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1])
            if e != -1:
                edges[e]['cnt'] += 1
                fa1 = get_fa(ce)
                fa2 = get_fa(edges[e]['cid'])
                if fa1 != fa2:
                    fa[fa1] = fa2
            else:
                edges.append({'edge': (coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1]), 'cnt': 1, 'cid': ce})
                id = len(edges) - 1
                add_hash(coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1], id)

    area_list = {}
    for ce in range(len(coords_list)):
        coords = coords_list[ce]
        fa0 = get_fa(ce)
        if fa0 in area_list:
            area_list[fa0]['area'] += Polygon(coords).area
            area_list[fa0]['list'].append(ce)
        else:
            info = {'area': Polygon(coords).area, 'list': [ce]}
            area_list.update({fa0: info})
    tot_area = 0
    for fa0 in area_list:
        tot_area += area_list[fa0]['area']
    for fa0 in area_list:
        # if area_list[fa0]['area'] >= tot_area*0.7:
        if area_list[fa0]['area'] == tot_area:
            new_coords_list = []
            for ce in area_list[fa0]['list']:
                new_coords_list.append(coords_list[ce])
            return True, new_coords_list

    return False, None


def checkMerge(cells, label_nodes, max_label):
    cnt = 0
    major_coords = []
    for lb in range(max_label):
        lb_size = label_nodes[lb].shape[0]
        coords_list = []
        for i in range(lb_size):
            cell = cells[cnt+i]
            coords = np.array(cell.exterior.coords)
            coords_list.append(coords)

        result, new_coords_list = check_merge_coords(coords_list)
        if not result:
            return False, None
        major_coords.append(new_coords_list)
        cnt += lb_size
    return True, major_coords

def get_convexhull_from_coords(major_coords, major_points):
    max_label = len(major_coords)
    label_coords = {}
    for lb in range(max_label):
        if major_points is not None:
            if lb in major_points:
                point_array = major_points[lb]
            else:
                point_array = np.zeros((0, 2))
        else:
            point_array = np.zeros((0, 2))
            for ce in range(len(major_coords[lb])):
                coords = major_coords[lb][ce]
                point_array.append(coords)
            point_array = np.concatenate(point_array, axis=0)
        points = MultiPoint([Point(x, y) for x, y in point_array])
        label_coords[lb] = points.convex_hull

    return label_coords

def get_graph_from_coords(major_coords, major_points=None, graph_type="origin"):
    max_label = len(major_coords)

    if graph_type == "hull":
        return get_convexhull_from_coords(major_coords, major_points)

    label_coords = {}
    for lb in range(max_label):
        coords_list = major_coords[lb]
        new_coords = merge_coords(coords_list)
        label_coords.update({lb: new_coords})

    # print(label_coords)

    skeleton = get_skeleton(label_coords, hull=None)

    # print(skeleton)

    if graph_type=="origin":
        for lb in label_coords:
            label_coords[lb] = Polygon(label_coords[lb][:-1])
        return label_coords
    if graph_type=="skeleton":
        for lb in skeleton:
            if len(skeleton[lb]) <= 3:
                skeleton[lb] = Polygon(label_coords[lb][:-1])
            else:
                skeleton[lb] = Polygon(skeleton[lb][:-1])
        return skeleton
    return label_coords

def get_graph_edges(X_embedded, partitions, reduce=None, major_coords=None, use_hull=False):
    start = time.time()

    if reduce is not None:
        X_embedded = X_embedded[reduce]
        partitions = partitions[reduce]

    # print("X_embedded", X_embedded)

    max_label = partitions.max()+1
    label_datas = {}
    label_nodes = {}
    label_centers = {}
    nodes = np.zeros((0, 2))
    labels = np.zeros(0, dtype='int')
    for lb in range(max_label):
        if lb not in label_datas:
            label_datas.update({lb: X_embedded[(partitions == lb)]})
        # print(label_datas[lb])
        size = label_datas[lb].shape[0]

        cut_number = int(max(120, 800/max_label))
        if size > cut_number:
            samples = np.random.choice(size, min(2*cut_number, max(int(size*0.5), cut_number)))
            # print("random samples", size, min(2*cut_number, max(int(size*0.5), cut_number)), samples)
            label_datas[lb] = label_datas[lb][samples]

        if reduce is None:
            if size < 20:
                # inliers = label_datas[lb]
                # outliers = np.array([])
                new_nodes = label_datas[lb]
                center = new_nodes.mean(axis=0)
                new_nodes = np.concatenate((new_nodes, [center]), axis=0)
            else:
                clf = LocalOutlierFactor(contamination=0.1)
                flag = clf.fit_predict(label_datas[lb])
                inliers = label_datas[lb][(flag == 1)]
                outliers = label_datas[lb][(flag == -1)]
                # print(outliers)
                min_i = inliers.min(axis=0)
                inliers -= min_i
                # print(inliers)
                max_i = inliers.max(axis=0)
                inliers /= max_i
                # print(inliers)
                alpha = 3
                while True:
                    # print('inliers', inliers)
                    alpha_shape = alphashape.alphashape(inliers, alpha)
                    if isinstance(alpha_shape, MultiPolygon):
                        alpha *= 2/3
                        continue
                    coords = np.array(alpha_shape.exterior.coords)*max_i+min_i
                    center = np.array([alpha_shape.centroid.x, alpha_shape.centroid.y])*max_i+min_i
                    new_nodes = coords[:-1]
                    new_nodes = np.concatenate((new_nodes, [center]), axis=0)
                    break
                # print(coords)
        else:
            new_nodes = label_datas[lb]
            center = new_nodes.mean(axis=0)
            new_nodes = np.concatenate((new_nodes, [center]), axis=0)

        label_nodes.update({lb: new_nodes})
        label_centers.update({lb: center})
        nodes = np.concatenate([nodes, label_nodes[lb]], axis=0)
        labels = np.concatenate([labels, np.full(label_nodes[lb].shape[0], fill_value=lb, dtype='int')])

    if use_hull:
        datas = nodes.copy()
        size = datas.shape[0]
        datas2 = datas
        min_i = datas.min(axis=0)
        datas = datas - min_i
        # print(inliers)
        max_i = datas.max(axis=0)
        datas = datas / max_i

        alpha = 0
        while True:
            alpha_hull = alphashape.alphashape(datas, alpha)
            if isinstance(alpha_hull, MultiPolygon):
                alpha *= 2/3
                continue
            coords = np.array(alpha_hull.exterior.coords)*max_i+min_i
            alpha_hull = Polygon(coords)
            cp = np.array([alpha_hull.centroid.x, alpha_hull.centroid.y])
            coords = (coords - cp)*1.01 + cp
            alpha_hull2 = Polygon(coords)
            break

    # print("graph time", time.time()-start)

    if major_coords is None:
        it_cnt = 1
        while True:
            if it_cnt > 2:
                break
            it_cnt += 1
            start0 = time.time()
            hull = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
            if use_hull:
                hull = alpha_hull2
            # print("nodes number", nodes.shape[0])
            cells, edges = computePowerDiagramByCGAL(positions=nodes, weights=np.zeros(nodes.shape[0]), hull=hull, withEdge=True)
            # print("cell time", time.time()-start0)

            result, major_coords = checkMerge(cells, label_nodes, max_label)
            # print("check merge time", time.time()-start0)
            if result:
                break
            cnt = 0
            for lb in range(max_label):
                lb_size = label_nodes[lb].shape[0]
                for i in range(lb_size):
                    nodes[cnt + i] = (nodes[cnt + i] - label_centers[lb])*2/3 + label_centers[lb]
                cnt += lb_size

        # print(cells)
        # print("graph time", time.time()-start)

        cm = MyColorMap()

    cnt = 0
    label_coords = {}
    for lb in range(max_label):
        lb_size = label_nodes[lb].shape[0]
        if major_coords is None:
            coords_list = []
            for i in range(lb_size):
                cell = cells[cnt+i]
                coords = np.array(cell.exterior.coords)
                coords_list.append(coords)
        else:
            coords_list = major_coords[lb]
        new_coords = merge_coords(coords_list)
        label_coords.update({lb: new_coords})
        cnt += lb_size

    # print("graph time", time.time()-start)


    skeleton, edges = get_skeleton(label_coords, hull=None, withEdges=True)

    # print(edges)

    # print("graph time", time.time()-start)

    return edges


def CentersAdjustWithCons(ori_centers, cons, old_cells):

    import time
    start = time.time()

    for i in range(len(cons)):
        u, v, x, y, w = cons[i]
        dx = ori_centers[u][0] - ori_centers[v][0]
        dy = ori_centers[u][1] - ori_centers[v][1]
        dz = np.sqrt(dx*dx+dy*dy)
        dx /= dz
        dy /= dz
        if x*dy-y*dx < 0:
            x = -x
            y = -y
        # x += dy
        # y -= dx
        # z = np.sqrt(x*x+y*y)
        # x /= z
        # y /= z
        cons[i] = (u, v, x, y, w)

    cons_dict = {}
    for i in range(len(cons)):
        u, v, x, y, w = cons[i]
        if (u, v) not in cons_dict:
            cons_dict.update({(u, v): (0.0, 0.0, 0.0)})
        x0, y0, w0 = cons_dict[(u, v)]
        cons_dict[(u, v)] = (x0*w0+x*w, y0*w0+y*w, 1)

    cons = []
    for (u, v) in cons_dict:
        x, y, w = cons_dict[(u, v)]
        cons.append((u, v, x, y, w))
    # print(cons)

    N = ori_centers.shape[0]

    hull = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
    cells = old_cells.copy()

    # print("item adjust 0", time.time()-start)
    id = 0
    for cell in old_cells:
        shell = []
        for pt in cell.exterior.coords:
            npt = np.array(pt)
            npt = (npt - ori_centers[id])*0.5 + ori_centers[id]
            shell.append(npt)
        cells[id] = Polygon(shell=shell)
        id += 1
    # print("cells", cells)
    # return
    # print("item adjust 0", time.time()-start)
    # centers = torch.from_numpy(centers)
    centers = []
    for i in range(N):
        centers.append([torch.tensor(ori_centers[i][0], requires_grad=True), torch.tensor(ori_centers[i][1], requires_grad=True)])
    lamda = 0.01
    for it in range(100):
        loss = 0
        tmpx = [[0]*N for _ in range(N)]
        tmpy = [[0]*N for _ in range(N)]
        tmpz = [[0]*N for _ in range(N)]
        tmpx2 = [[0]*N for _ in range(N)]
        tmpy2 = [[0]*N for _ in range(N)]
        for u, v, x, y, w in cons:
            tmpx[u][v] = centers[u][0]-centers[v][0]
            tmpy[u][v] = centers[u][1]-centers[v][1]
            tmpz[u][v] = torch.sqrt(tmpx[u][v]*tmpx[u][v] + tmpy[u][v]*tmpy[u][v])
            tmpx2[u][v] = tmpx[u][v] / tmpz[u][v]
            tmpy2[u][v] = tmpy[u][v] / tmpz[u][v]
            loss += torch.abs(tmpx2[u][v]*x+tmpy2[u][v]*y)*w
        # print(loss)
        loss.backward()
        for i in range(N):
            with torch.no_grad():
                gradx = 0
                if centers[i][0].grad is not None:
                    gradx = centers[i][0].grad.item()
                    centers[i][0].grad.zero_()
                grady = 0
                if centers[i][1].grad is not None:
                    grady = centers[i][1].grad.item()
                    centers[i][1].grad.zero_()
                a = [(centers[i][0] - gradx*lamda).item(), (centers[i][1] - grady*lamda).item()]
                # print(a)
                if cells[i].covers(Point(a)):
                    centers[i][0] -= gradx*lamda
                    centers[i][1] -= grady*lamda

        lamda = max(0.001, lamda * 0.9)
        # print("item adjust", it, time.time()-start)

    for i in range(N):
        centers[i][0] = centers[i][0].item()
        centers[i][1] = centers[i][1].item()
    # print("new_centers", centers)
    return centers


def CentersAdjustWithConsTest(ori_centers, cons, old_cells, use_HV=False):

    import time
    start = time.time()

    for i in range(len(cons)):
        u, v, x, y, w = cons[i]
        dx = ori_centers[u][0] - ori_centers[v][0]
        dy = ori_centers[u][1] - ori_centers[v][1]
        dz = np.sqrt(dx*dx+dy*dy)
        dx /= dz
        dy /= dz
        if x*dy-y*dx < 0:
            x = -x
            y = -y
        # x += dy
        # y -= dx
        # z = np.sqrt(x*x+y*y)
        # x /= z
        # y /= z
        cons[i] = (u, v, x, y, w)

    cons_dict = {}
    for i in range(len(cons)):
        u, v, x, y, w = cons[i]
        if (u, v) not in cons_dict:
            cons_dict.update({(u, v): (0.0, 0.0, 0.0)})
        x0, y0, w0 = cons_dict[(u, v)]
        cons_dict[(u, v)] = (x0*w0+x*w, y0*w0+y*w, 1)

    cons = []
    for (u, v) in cons_dict:
        x, y, w = cons_dict[(u, v)]
        cons.append((u, v, x, y, w))
    # print(cons)

    N = ori_centers.shape[0]

    hull = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
    cells = old_cells.copy()

    # print("item adjust 0", time.time()-start)
    id = 0
    for cell in old_cells:
        shell = []
        for pt in cell.exterior.coords:
            npt = np.array(pt)
            npt = (npt - ori_centers[id])*0.9 + ori_centers[id]
            shell.append(npt)
        cells[id] = Polygon(shell=shell)
        id += 1
    # print("cells", cells)
    # return
    # print("item adjust 0", time.time()-start)
    centers = []
    for i in range(N):
        centers.append([ori_centers[i][0], ori_centers[i][1]])

    if not use_HV:
        lamda = 0.01
        for it in range(100):
            loss = 0
            tmpx = [[0]*N for _ in range(N)]
            tmpy = [[0]*N for _ in range(N)]
            tmpz = [[0]*N for _ in range(N)]
            tmpx2 = [[0]*N for _ in range(N)]
            tmpy2 = [[0]*N for _ in range(N)]
            grad_x = [0]*N
            grad_y = [0]*N
            for u, v, x, y, w in cons:
                tmpx[u][v] = centers[u][0]-centers[v][0]
                tmpy[u][v] = centers[u][1]-centers[v][1]
                tmpz[u][v] = np.sqrt(tmpx[u][v]*tmpx[u][v] + tmpy[u][v]*tmpy[u][v])
                tmpx2[u][v] = tmpx[u][v] / tmpz[u][v]
                tmpy2[u][v] = tmpy[u][v] / tmpz[u][v]
                loss += abs(tmpx2[u][v]*x+tmpy2[u][v]*y)*w
                f_flag = 1
                if tmpx2[u][v]*x+tmpy2[u][v]*y < 0:
                    f_flag = -1
                tmp_grad_x = x*w*f_flag
                tmp_grad_y = y*w*f_flag
                tmp_dx1 = tmpy[u][v]*tmpy[u][v]/np.power(tmpz[u][v],3)
                tmp_dx2 = -tmpx[u][v]*tmpy[u][v]/np.power(tmpz[u][v],3)
                tmp_dy1 = tmpx[u][v]*tmpx[u][v]/np.power(tmpz[u][v],3)
                tmp_dy2 = -tmpx[u][v]*tmpy[u][v]/np.power(tmpz[u][v],3)
                grad_x[u] += tmp_grad_x*tmp_dx1 + tmp_grad_y*tmp_dx2
                grad_x[v] -= tmp_grad_x*tmp_dx1 + tmp_grad_y*tmp_dx2
                grad_y[u] += tmp_grad_y*tmp_dy1 + tmp_grad_x*tmp_dy2
                grad_y[v] -= tmp_grad_y*tmp_dy1 + tmp_grad_x*tmp_dy2
            # print(loss)
            for i in range(N):
                gradx = grad_x[i]
                grady = grad_y[i]
                a = [(centers[i][0] - gradx*lamda), (centers[i][1] - grady*lamda)]
                # print(a)
                if cells[i].covers(Point(a)):
                    centers[i][0] -= gradx*lamda
                    centers[i][1] -= grady*lamda

            lamda = max(0.001, lamda * 0.9)
            # print("item adjust", it, time.time()-start)

    if use_HV:
        lamda = 0.01
        for it in range(50):
            loss = 0
            tmpx = [[0]*N for _ in range(N)]
            tmpy = [[0]*N for _ in range(N)]
            tmpz = [[0]*N for _ in range(N)]
            tmpx2 = [[0]*N for _ in range(N)]
            tmpy2 = [[0]*N for _ in range(N)]
            grad_x = [0]*N
            grad_y = [0]*N
            for u, v, x, y, w in cons:
                tmpx[u][v] = centers[u][0]-centers[v][0]
                tmpy[u][v] = centers[u][1]-centers[v][1]
                w = np.sqrt(x*x+y*y)*w
                if abs(tmpx[u][v])>abs(tmpy[u][v]):
                    x = 0
                    y = 1
                else:
                    y = 0
                    x = 1
                tmpz[u][v] = np.sqrt(tmpx[u][v]*tmpx[u][v] + tmpy[u][v]*tmpy[u][v])
                tmpx2[u][v] = tmpx[u][v] / tmpz[u][v]
                tmpy2[u][v] = tmpy[u][v] / tmpz[u][v]
                loss += abs(tmpx2[u][v]*x+tmpy2[u][v]*y)*w
                f_flag = 1
                if tmpx2[u][v]*x+tmpy2[u][v]*y < 0:
                    f_flag = -1
                tmp_grad_x = x*w*f_flag
                tmp_grad_y = y*w*f_flag
                # tmp_dx1 = tmpy[u][v]*tmpy[u][v]/np.power(tmpz[u][v],3)
                # tmp_dx2 = -tmpx[u][v]*tmpy[u][v]/np.power(tmpz[u][v],3)
                # tmp_dy1 = tmpx[u][v]*tmpx[u][v]/np.power(tmpz[u][v],3)
                # tmp_dy2 = -tmpx[u][v]*tmpy[u][v]/np.power(tmpz[u][v],3)
                tmp_dx1 = 1/tmpz[u][v]
                tmp_dx2 = 0
                tmp_dy1 = 1/tmpz[u][v]
                tmp_dy2 = 0
                grad_x[u] += tmp_grad_x*tmp_dx1 + tmp_grad_y*tmp_dx2
                grad_x[v] -= tmp_grad_x*tmp_dx1 + tmp_grad_y*tmp_dx2
                grad_y[u] += tmp_grad_y*tmp_dy1 + tmp_grad_x*tmp_dy2
                grad_y[v] -= tmp_grad_y*tmp_dy1 + tmp_grad_x*tmp_dy2

                # if v == 4:
                #     print("grad(4)1", u, v, x, y, w, grad_x[v], grad_y[v])
                #     print("grad(4)2", tmpx[u][v], tmpy[u][v], tmpz[u][v], tmpx2[u][v], tmpy2[u][v])
                #     print("grad(4)3", -tmp_grad_x*tmp_dx1 - tmp_grad_y*tmp_dx2, -tmp_grad_y*tmp_dy1 - tmp_grad_x*tmp_dy2)
            # print(loss)
            for i in range(N):
                gradx = grad_x[i]
                grady = grad_y[i]
                ming = 5
                for j in range(N):
                    if i==j:
                        continue
                    ming = min(ming, np.sqrt(np.power(centers[i][0]-centers[j][0], 2)+np.power(centers[i][1]-centers[j][1], 2))*10)
                gradx = min(max(gradx, -ming), ming)
                grady = min(max(grady, -ming), ming)
                a = [(centers[i][0] - gradx*lamda), (centers[i][1] - grady*lamda)]
                # print("grad", i, gradx, grady)
                # print(a)
                flag = True
                for u, v, x, y, w in cons:
                    if u==i:
                        x1 = a[0]-centers[v][0]
                        x2 = ori_centers[u][0]-ori_centers[v][0]
                        y1 = a[1]-centers[v][1]
                        y2 = ori_centers[u][1]-ori_centers[v][1]
                        if x1*x2+y1*y2 < np.sqrt(x1*x1+y1*y1)*np.sqrt(x2*x2+y2*y2)/np.sqrt(2):
                            flag = False
                            break
                    if v==i:
                        x1 = centers[u][0]-a[0]
                        x2 = ori_centers[u][0]-ori_centers[v][0]
                        y1 = centers[u][1]-a[1]
                        y2 = ori_centers[u][1]-ori_centers[v][1]
                        if x1*x2+y1*y2 < np.sqrt(x1*x1+y1*y1)*np.sqrt(x2*x2+y2*y2)/np.sqrt(2):
                            flag = False
                            break
                if flag:
                    # print("yes", i)
                    centers[i][0] -= gradx*lamda
                    centers[i][1] -= grady*lamda

            lamda = max(0.001, lamda * 0.9)
            # print("item adjust", it, time.time()-start)
            if it%10 == 0:
                N = len(centers)
                hull = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
                now_cells = computePowerDiagramByCGAL(positions=np.array(centers), weights=np.zeros(N), hull=hull)
                label_coords = {}
                for lb in range(N):
                    label_coords.update({lb: np.array(now_cells[lb].exterior.coords)})
                skeleton, edges = get_skeleton(label_coords, hull=None, withEdges=True)
                cons = edges
                # print(cons)
        
    # print("new_centers", centers)
    return centers


def CentersAdjust(X_embedded, partitions, centers, reduce=None, major_coords=None, now_hull=None):

    # print("old_centers", centers)
    N = centers.shape[0]
    hull = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
    if now_hull is not None:
        hull = now_hull
    old_cells, edges = computePowerDiagramByCGAL(positions=centers, weights=np.zeros(N), hull=hull, withEdge=True)

    if now_hull is None:
        cons = get_graph_edges(X_embedded, partitions, reduce, major_coords)
    else:
        cons = get_graph_edges(X_embedded, partitions, reduce, major_coords, use_hull=True)

    return CentersAdjustWithConsTest(centers, cons, old_cells)


def CentersAdjustZoom(zoom_partition_map, zoom_min, zoom_max, all_cells_bf, centers):

    start = time.time()

    # print("old_centers", centers)
    N = centers.shape[0]
    hull = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
    old_cells, edges = computePowerDiagramByCGAL(positions=centers, weights=np.zeros(N), hull=hull, withEdge=True)

    max_label = len(centers)
    while True:
        flag = True
        zoom_box = Polygon([[zoom_min[0], zoom_min[1]], [zoom_min[0], zoom_max[1]], [zoom_max[0], zoom_max[1]], [zoom_max[0], zoom_min[1]]])
        label_coords = {}
        for lb in range(max_label):
            cell = zoom_box.intersection(all_cells_bf[zoom_partition_map[lb]])
            if cell is None or cell.area==0:
                flag = False
                break
            new_coords = (np.array(cell.exterior.coords)-zoom_min)/(zoom_max-zoom_min)
            label_coords.update({lb: new_coords})
        if flag:
            break
        else:
            height = zoom_max[0]-zoom_min[0]
            width = zoom_max[1]-zoom_min[1]
            zoom_min[0] = max(0.0, zoom_min[0]-height/6)
            zoom_min[1] = max(0.0, zoom_min[1]-width/6)
            zoom_max[0] = min(1, zoom_max[0]+height/6)
            zoom_max[1] = min(1, zoom_max[1]+width/6)
    
    _, cons = get_skeleton(label_coords, hull=None, withEdges=True)

    polygon_list = []
    for lb in range(max_label):
        polygon_list.append(Polygon(label_coords[lb][:-1]).buffer(1e-8, quad_segs=2))

    def get_dist(x1, x2):
        return np.sqrt((x1[0]-x2[0])**2+(x1[1]-x2[1])**2)

    cons2 = []
    for lb1 in range(max_label):
        for lb2 in range(lb1+1, max_label):
            intersection = polygon_list[lb1].intersection(polygon_list[lb2])
            if intersection.is_empty:
                continue
            elif intersection.geom_type == 'LineString':
                common_edge = list(intersection.coords)
                cons2.append((lb1, lb2, common_edge[0][0]-common_edge[1][0], common_edge[0][1]-common_edge[1][1], 1))
            elif intersection.geom_type == "Polygon":
                common_edge = list(intersection.exterior.coords)
                ei = 0
                for i in range(1, len(common_edge)-1):
                    if get_dist(common_edge[i], common_edge[i+1]) > get_dist(common_edge[ei], common_edge[ei+1]):
                        ei = i
                cons2.append((lb1, lb2, common_edge[ei][0]-common_edge[ei+1][0], common_edge[ei][1]-common_edge[ei+1][1], 1))

    # print("time adjust cons", time.time()-start)

    return CentersAdjustWithConsTest(centers, cons2, old_cells)


def cutHull(hull, ratio, axis):
    bounds = hull.bounds
    bottom = bounds[0]
    top = bounds[2]
    left = bounds[1]
    right = bounds[3]
    aim_area = hull.area*ratio
    if axis=='x':
        l = bottom
        r = top
        mid = (l+r)/2
        while r-l>0.00001:
            mid = (l+r)/2
            CutP = Polygon([[bottom-0.1, left-0.1], [mid, left-0.1], [mid, right+0.1], [bottom-0.1, right+0.1]])
            new_hull = hull.intersection(CutP)
            area = 0
            if new_hull is not None:
                area = new_hull.area
            if area<aim_area:
                l = mid
            else:
                r = mid
            ans = mid
        CutP = Polygon([[bottom-0.1, left-0.1], [mid, left-0.1], [mid, right+0.1], [bottom-0.1, right+0.1]])
        CutP2 = Polygon([[top+0.1, left-0.1], [mid, left-0.1], [mid, right+0.1], [top+0.1, right+0.1]])
        return hull.intersection(CutP), hull.intersection(CutP2)
    else:
        l = left
        r = right
        mid = (l+r)/2
        while r-l>0.00001:
            mid = (l+r)/2
            CutP = Polygon([[bottom-0.1, left-0.1], [bottom-0.1, mid], [top+0.1, mid], [top+0.1, left-0.1]])
            new_hull = hull.intersection(CutP)
            area = 0
            if new_hull is not None:
                area = new_hull.area
            if area<aim_area:
                l = mid
            else:
                r = mid
        CutP = Polygon([[bottom-0.1, left-0.1], [bottom-0.1, mid], [top+0.1, mid], [top+0.1, left-0.1]])
        CutP2 = Polygon([[bottom-0.1, right+0.1], [bottom-0.1, mid], [top+0.1, mid], [top+0.1, right+0.1]])
        return hull.intersection(CutP), hull.intersection(CutP2)


def getCutWayDiagram(hull, capacity, cut_way, axis):
    cell_cap = []
    tot_cap = 0
    for cell in cut_way:
        cap = capacity[cell]
        cell_cap.append(cap)
        tot_cap += cap
    cells_dict = {}
    for i in range(len(cut_way)):
        if i < len(cut_way)-1:
            sub_hull, new_hull = cutHull(hull, cell_cap[i]/tot_cap, axis)
            hull = new_hull
        else:
            sub_hull = hull
            hull = None
        tot_cap -= cell_cap[i]
        cells_dict.update({cut_way[i]: sub_hull})
    return cells_dict


def getTreeDiagramDFS(hull, item):
    cells_dict = {}
    if item['child'] is None:
        cells_dict[item["part_id"]] = hull
        return cells_dict
    child1, child2 = item['child']
    hull1, hull2 = cutHull(hull, child1['size']/item['size'], item['axis'])
    new_dict1 = getTreeDiagramDFS(hull1, child1)
    cells_dict.update(new_dict1)
    new_dict2 = getTreeDiagramDFS(hull2, child2)
    cells_dict.update(new_dict2)
    return cells_dict


def getTreeDiagram(hull, capacity, cut_ways):
    cells_dict = getTreeDiagramDFS(hull, cut_ways['ways'][0])
    ret = []
    for i in range(len(capacity)):
        ret.append(cells_dict[i])
    return ret


def getCutDiagram(hull, capacity, cut_ways):
    if cut_ways['axis'] == 'tree':
        return getTreeDiagram(hull, capacity, cut_ways)

    way_cap = []
    tot_cap = 0
    for way in cut_ways['ways']:
        cap = 0
        for lb in way:
            cap += capacity[lb]
        way_cap.append(cap)
        tot_cap += cap

    cells_dict = {}
    for i in range(len(cut_ways['ways'])):
        axis = cut_ways['axis']
        axis2 = 'y'
        if axis == 'y':
            axis2 = 'x'
        if i < len(cut_ways['ways'])-1:
            sub_hull, new_hull = cutHull(hull, way_cap[i]/tot_cap, axis)
            hull = new_hull
        else:
            sub_hull = hull
            hull = None
        tot_cap -= way_cap[i]
        tmp_dict = getCutWayDiagram(sub_hull, capacity, cut_ways['ways'][i], axis2)
        cells_dict.update(tmp_dict)
    ret = []
    for i in range(len(capacity)):
        ret.append(cells_dict[i])
    return ret


def getPowerDiagramGrids(labels, centers, square_len, now_hull=None, now_grids=None, edge_matrix=None, compact=False):
    start = time.time()

    N = square_len * square_len
    num = labels.shape[0]
    maxLabel = centers.shape[0]
    capacity = np.zeros(maxLabel)
    for i in range(maxLabel):
        capacity[i] = (labels == i).sum() / num
    
    now_N = N
    if now_grids is not None:
        now_N = now_grids.sum()

    hull = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
    if now_hull is not None:
        hull = now_hull
    if maxLabel > 1:
        # print("get cells", cluster_centers, np.zeros(len(cluster_centers)), capacity*hull.area, hull)

        last_centers = centers.copy()
        last_cells = None
        next_centers = None
        old_edges = None
        now_centers = last_centers.copy()
        fail = 0
        if compact:
            max_it = 5
        else:
            max_it = 1

        for li in range(max_it):

            cal_cnt = 0
            w = np.zeros(len(now_centers))
            while cal_cnt < 2:
                cal_cnt += 1
                w = find_W(now_centers, w, capacity*hull.area, hull)
                cells = computePowerDiagramByCGAL(positions=now_centers, weights=w, hull=hull)

                label_coords = {}
                for i in range(maxLabel):
                    label_coords[i] = np.array(cells[i].exterior.coords)
                _, new_edges = get_skeleton(label_coords, withEdges=True)
                edges = []
                for (a, b, c, d, e) in new_edges:
                    if a > b:
                        a, b = b, a
                    if (edge_matrix is not None) and (not edge_matrix[a][b]):
                        continue
                    sheld = 2.5
                    if old_edges is not None:
                        sheld = 1.5
                    if (e > sheld/square_len) and ((a, b) not in edges):
                        edges.append((a, b))

                flag = 0
                for i in range(maxLabel):
                    if(abs(cells[i].area-capacity[i]*hull.area)*maxLabel>5/square_len) or cells[i].is_empty:
                        flag = 1
                        # print('calculate w again', i, abs(cells[i].area-capacity[i]*hull.area)*maxLabel)
                        break
                if flag == 0:
                    break

            if old_edges is None:
                old_edges = edges
                last_cells = cells
                last_centers = now_centers.copy()
                now_centers = []
                for cell in cells:
                    now_centers.append(cellCentroid(cell))
                now_centers = np.array(now_centers)
            else:
                flag = 0
                flag_back = np.zeros(maxLabel, dtype="bool")
                flag_back2 = np.zeros(maxLabel, dtype="bool")
                for (a, b) in old_edges:
                    if ((a, b) not in edges) and ((b, a) not in edges):
                        flag += 1
                        flag_back[a] = flag_back[b] = True
                        for c in range(maxLabel):
                            if (((a, c) in old_edges) or ((c, a) in old_edges)) and (((b, c) in old_edges) or ((c, b) in old_edges)):
                                flag_back2[c] = True
                if flag <= len(old_edges)*0:
                # if flag <= 3:
                    if fail <= 1:
                        fail = 0
                    last_cells = cells
                    last_centers = now_centers.copy()
                    if fail <= 1:
                        now_centers = []
                        for cell in cells:
                            now_centers.append(cellCentroid(cell))
                        now_centers = np.array(now_centers)
                    else:
                        now_centers = (last_centers + next_centers) / 2

                else:
                    fail += 1
                    next_centers = now_centers.copy()
                    if fail <= 1:
                        now_centers[flag_back] = last_centers[flag_back]
                        now_centers[flag_back2] = last_centers[flag_back2]
                        # now_centers[flag_back2] = (last_centers[flag_back2] + now_centers[flag_back2]) / 2
                    else:
                        now_centers = (last_centers + next_centers) / 2

        cells = last_cells
        centers = last_centers
    else:
        cells = [hull]
        centers = np.array([[hull.centroid.x, hull.centroid.y]])
    bounds = []
    for i in range(len(cells)):
        bounds.append(cells[i].bounds)

    # print("time", time.time() - start)

    cornel_size = 0
    while cornel_size * (cornel_size + 1) * 2 < now_N - num:
        cornel_size += 1

    elements = []
    for i in range(maxLabel):
        elements.append(np.arange(num, dtype='int')[(labels == i)])

    grids = np.dstack(np.meshgrid(np.linspace(0, 1 - 1.0 / square_len, square_len),
                                  np.linspace(0, 1 - 1.0 / square_len, square_len))) \
        .reshape(-1, 2)

    tmp = grids[:, 0].copy()
    grids[:, 0] = grids[:, 1]
    grids[:, 1] = tmp
    grids = grids + np.array([1.0, 1.0]) / 2 / square_len

    belong = np.ones(N, dtype='int')*-1

    if maxLabel>1:
        for i in range(grids.shape[0]):
            if (now_grids is not None) and (not now_grids[i]):
                continue
            list = []
            for id, cell in enumerate(cells):
                if grids[i][0]<bounds[id][0] or grids[i][0]>bounds[id][2] or grids[i][1]<bounds[id][1] or grids[i][1]>bounds[id][3]:
                    continue
                if cell.covers(Point(grids[i])):
                    list.append(id)
            if len(list)>0:
                belong[i] = random.choice(list)
            else:
                belong[i] = -1
    else:
        belong = np.zeros(N, dtype='int')

    for x in range(square_len):
        bias = x * square_len
        for y in range(square_len):
            gid = bias + y
            dx = min(x, square_len-1-x)
            dy = min(y, square_len-1-y)
            if dx+dy<cornel_size:
            # if ((x < cornel_size) or (x >= square_len - cornel_size)) and (
            #         (y < cornel_size) or (y >= square_len - cornel_size)):
                belong[gid] = -2

    centers = centers + np.random.normal(0, 1e-4, (len(centers), 2))

    now_centers = []
    for cell in cells:
        now_centers.append(cellCentroid(cell))
    now_centers = np.array(now_centers)
    full_g = np.arange(N, dtype='int')
    if now_grids is not None:
        full_g = full_g[now_grids]
    tmp_dist = np.power(cdist(now_centers, grids[full_g], "euclidean"), 2)
    opt = gridlayoutOpt.Optimizer(0)
    tmp_center = opt.solveKM(tmp_dist)
    for i in range(maxLabel):
        belong[full_g[tmp_center[i]]] = i
    # tmp_grids = np.arange(N, dtype='int')
    # consider = np.ones(N, dtype='bool')
    # if now_grids is not None:
    #     consider = now_grids
    # consider = np.logical_and(consider, (belong >= 0))
    # reduce = np.array(gridlayoutOpt.getMainConnectGrids(tmp_grids, belong, consider))
    # belong[np.logical_and(reduce==False, belong >= 0)] = -1

    unfilled = []
    row_asses = np.zeros(N, dtype='int')
    label_used = np.zeros(maxLabel, dtype='int')
    element_used = np.zeros(num, dtype='bool')
    dist_label = np.full((N, maxLabel), fill_value=1000)
    for gid in range(N):
        if belong[gid] >= 0:
            dist_label[gid][belong[gid]] = -1000

    # print("time", time.time() - start)

    neighbor = 1
    while True:
        fine = True
        # print("neighbor", neighbor)
        unfilled = []
        row_asses = np.zeros(N, dtype='int')
        label_used = np.zeros(maxLabel, dtype='int')
        element_used = np.zeros(num, dtype='bool')

        for x in range(square_len):
            bias = x * square_len
            for y in range(square_len):
                gid = bias + y
                if (now_grids is not None) and (not now_grids[gid]):
                    continue
                unfill = False
                if belong[gid]<0:
                    unfill = True
                for xx in range(2 * neighbor + 1):
                    x2 = x + xx - neighbor
                    if (x2 < 0) or (x2 >= square_len):
                        continue
                    bias2 = x2 * square_len
                    for yy in range(2 * neighbor + 1):
                        y2 = y + yy - neighbor
                        if (y2 < 0) or (y2 >= square_len):
                            continue
                        gid2 = bias2 + y2
                        if (now_grids is not None) and (not now_grids[gid2]):
                            continue
                        if belong[gid2] == -2:
                            continue
                        if belong[gid] != belong[gid2]:
                            unfill = True
                            if belong[gid] >= 0:
                                dist_label[gid][belong[gid]] = max(dist_label[gid][belong[gid]], -max(abs(xx-neighbor), abs(yy-neighbor)))
                            if belong[gid2] >= 0:
                                dist_label[gid][belong[gid2]] = min(dist_label[gid][belong[gid2]], max(abs(xx-neighbor), abs(yy-neighbor)))
                if not unfill:
                    lb = belong[gid]
                    label_used[lb] += 1
                    if label_used[lb] <= elements[lb].shape[0]:
                        row_asses[gid] = elements[lb][label_used[lb] - 1]
                        element_used[row_asses[gid]] = True
                    else:
                        fine = False
                else:
                    unfilled.append(gid)

        if not fine:
            neighbor += 1
            continue
        
        # print('try to assign')
        
        unfilled = np.array(unfilled)
        if unfilled.shape[0]>0:
            grid_positions = grids[unfilled]
            unused = np.arange(num, dtype='int')[(element_used == False)]
            # print(unfilled, unused)
            center_positions = centers[labels[unused]]
            original_cost_matrix = cdist(grid_positions, center_positions, "euclidean")
            for i in range(unused.shape[0]):
                dist = dist_label[unfilled, labels[unused[i]]]
                dist = (dist>999).astype('int')*100
                original_cost_matrix[:, i] += dist
            dummy_points = np.ones((now_N - num, 2)) * 0.5
            dummy_vertices = (1 - cdist(grid_positions, dummy_points, "euclidean")) * 100
            cost_matrix = np.concatenate((original_cost_matrix, dummy_vertices), axis=1)
            cost_matrix = np.power(cost_matrix, 2)

            # print("time", time.time() - start)

            opt = gridlayoutOpt.Optimizer(0)
            # opt = tmpKMSolver()
            # print(cost_matrix.shape)

            # row_asses2 = np.array(opt.solveKM(cost_matrix))

            original_cost_matrix2 = cdist(grid_positions, centers, "euclidean")
            for i in range(maxLabel):
                dist = dist_label[unfilled, i]
                dist = (dist>999).astype('int')*100
                original_cost_matrix2[:, i] += dist
            dummy_points2 = np.ones((1, 2)) * 0.5
            dummy_vertices2 = (1 - cdist(grid_positions, dummy_points2, "euclidean")) * 100
            cost_matrix2 = np.concatenate((original_cost_matrix2, dummy_vertices2), axis=1)
            cost_matrix2 = np.power(cost_matrix2, 2)
            labels2 = labels[unused]
            labels2 = np.concatenate((labels2, np.ones(now_N-num, dtype='int')*maxLabel), axis=0)
            row_asses2 = np.array(opt.solveKMLabel(cost_matrix2, labels2))

            # print(row_asses2)

            dummy_id = np.arange(now_N-num, dtype='int')+num
            unused = np.concatenate((unused, dummy_id), axis=0)
            
            # print("time", time.time() - start) 
            
            flag = False
            for i in range(len(row_asses2)):
                row_asses[unfilled[i]] = unused[row_asses2[i]]
                if (unused[row_asses2[i]]<num)and(dist_label[unfilled[i]][labels[unused[row_asses2[i]]]]>999):
                    # print("fail", unfilled[i])
                    flag = True
                    
            # print("time", time.time() - start) 
            
            if flag:
                # print('fail')
                neighbor += 1
                
                # if neighbor == 2:
                #     labels2 = labels.copy()
                #     if unfilled.shape[0]>0:
                #         for i in range(len(row_asses2)):
                #             if unused[row_asses2[i]]<num:
                #                 labels2[unused[row_asses2[i]]] = labels.max()+1
                #     show_grid_tmp(row_asses, labels2, square_len, "fail_powergrid0.png", False)
                #     show_grid_tmp(row_asses, labels, square_len, "fail_powergrid1.png", False)
                continue
        # print('success')
        break

    # print("time", time.time() - start)
    
    # labels2 = labels.copy()
    # if unfilled.shape[0]>0:
    #     for i in range(len(row_asses2)):
    #         if unused[row_asses2[i]]<num:
    #             labels2[unused[row_asses2[i]]] = labels.max()+1
    # show_grid_tmp(row_asses, labels2, square_len, "powergrid0.png", False)
    # show_grid_tmp(row_asses, labels, square_len, "powergrid1.png", False)

    return row_asses, cells


def getFoldlineGrids(x_bf, y_bf, grid_label, labels, centers, square_len):

    import math
    start = time.time()

    N = square_len * square_len
    num = labels.shape[0]
    maxLabel = labels.max()+1
    capacity = np.zeros(maxLabel)
    for i in range(maxLabel):
        capacity[i] = (labels == i).sum() / num

    # print("time 1", time.time() - start)

    big_x = x_bf
    big_y = y_bf
    big_grid_label = grid_label.copy()
    if ((x_bf>=15)and(y_bf>=15)):
        big_x = 10
        big_y = 10
        big_grid_label = np.full((big_x, big_y), fill_value=-1, dtype='int')

        for i in range(big_x):
            for j in range(big_y):
                min_x = i/big_x
                min_y = j/big_y
                label_count = np.zeros(maxLabel, dtype='int')
                now_x = math.ceil((min_x-1/2/x_bf)*x_bf)
                while (now_x+1/2)/x_bf <= (i+1)/big_x:
                    now_y = math.ceil((min_y-1/2/y_bf)*y_bf)
                    while (now_y+1/2)/y_bf <= (j+1)/big_y:
                        if grid_label[now_x][now_y]>=0:
                            label_count[grid_label[now_x][now_y]] += 1
                        now_y += 1
                    now_x += 1
                if label_count.max()>0:
                    big_grid_label[i][j] = np.argmax(label_count)
                else:
                    now_x = math.ceil((min_x-1/2/x_bf)*x_bf)
                    now_y = math.ceil((min_y-1/2/y_bf)*y_bf)
                    if now_x >= x_bf:
                        now_x -= 1
                    if now_y >= y_bf:
                        now_y -= 1
                    big_grid_label[i][j] = grid_label[now_x][now_y]


    tmp_len = max(big_x, big_y)
    tmp_asses = np.arange(tmp_len*tmp_len, dtype='int')
    tmp_labels = np.full(tmp_len*tmp_len, fill_value=maxLabel, dtype='int')
    tmp_partition = np.ones(tmp_len*tmp_len, dtype='int')
    for i in range(big_x):
        for j in range(big_y):
            if big_grid_label[i][j]>=0:
                tmp_labels[i*tmp_len+j] = big_grid_label[i][j]
                tmp_partition[i*tmp_len+j] = 0
    tmp_embedded = np.dstack(np.meshgrid(np.linspace(0, 1 - 1.0 / tmp_len, tmp_len),
                                  np.linspace(0, 1 - 1.0 / tmp_len, tmp_len))) \
        .reshape(-1, 2)

    tmp = tmp_embedded[:, 0].copy()
    tmp_embedded[:, 0] = tmp_embedded[:, 1]
    tmp_embedded[:, 1] = tmp

    # print("time 2", time.time() - start)

    old_asses = tmp_asses
    tmp_asses = np.array(gridlayoutOpt.grid_op_partition(1, tmp_partition,
                                                        tmp_embedded, tmp_embedded, tmp_asses, tmp_labels, True, False,
                                                        "PerimeterRatio", 0, 6, False, 2147483647, 1, []))

    for i in range(big_x):
        for j in range(big_y):
            if big_grid_label[i][j]>=0:
                big_grid_label[i][j] = tmp_labels[tmp_asses[i*tmp_len+j]]

    # print("time 3", time.time() - start)

    tmp_asses = np.array(gridlayoutOpt.grid_op_partition(1, tmp_partition,
                                                        tmp_embedded, tmp_embedded, tmp_asses, tmp_labels, False, True,
                                                        "PerimeterRatio", 0, 6, False, 2147483647, 1, []))

    tmp_change = (1-tmp_partition).astype('bool')

    from .gridOptimizer_clean import gridOptimizer
    optimizer = gridOptimizer()
    cost1 = optimizer.check_cost_type(tmp_embedded, old_asses, tmp_labels, "PerimeterRatio", tmp_change)
    cost2 = optimizer.check_cost_type(tmp_embedded, tmp_asses, tmp_labels, "PerimeterRatio", tmp_change)
    # print("cost1", cost1)
    # print("cost2", cost2)
    if cost1[2]<cost2[2]:
        tmp_asses = old_asses

    for i in range(big_x):
        for j in range(big_y):
            if big_grid_label[i][j]>=0:
                big_grid_label[i][j] = tmp_labels[tmp_asses[i*tmp_len+j]]

    # print("time 4", time.time() - start)

    tmp_labels = np.array(gridlayoutOpt.changeLabelsForConvexity(tmp_asses, tmp_labels, tmp_change,
                                                        "PerimeterRatio", tmp_change))

    for i in range(big_x):
        for j in range(big_y):
            if big_grid_label[i][j]>=0:
                big_grid_label[i][j] = tmp_labels[tmp_asses[i*tmp_len+j]]

    # print("time 5", time.time() - start)

    cornel_size = 0
    while cornel_size * (cornel_size + 1) * 2 < N - num:
        cornel_size += 1

    elements = []
    for i in range(maxLabel):
        elements.append(np.arange(num, dtype='int')[(labels == i)])

    grids = np.dstack(np.meshgrid(np.linspace(0, 1 - 1.0 / square_len, square_len),
                                  np.linspace(0, 1 - 1.0 / square_len, square_len))) \
        .reshape(-1, 2)

    tmp = grids[:, 0].copy()
    grids[:, 0] = grids[:, 1]
    grids[:, 1] = tmp
    grids = grids + np.array([1.0, 1.0]) / 2 / square_len

    belong = np.ones(N, dtype='int')*(-1)

    for i in range(big_x):
        for j in range(big_y):
            min_x = i/big_x
            min_y = j/big_y
            now_x = math.ceil((min_x-1/2/square_len)*square_len)
            while (now_x+1/2)/square_len <= (i+1)/big_x:
                now_y = math.ceil((min_y-1/2/square_len)*square_len)
                while (now_y+1/2)/square_len <= (j+1)/big_y:
                    belong[now_x*square_len+now_y] = big_grid_label[i][j]
                    now_y += 1
                now_x += 1

    # print("time 6", time.time() - start)

    again_cnt = 0
    while again_cnt < 3:
        # show_grid_tmp(np.arange(N, dtype='int'), belong, square_len, "fold-2.png")
        for x in range(square_len):
            bias = x * square_len
            for y in range(square_len):
                gid = bias + y
                dx = min(x, square_len-1-x)
                dy = min(y, square_len-1-y)
                if dx+dy<cornel_size:
                # if ((x < cornel_size) or (x >= square_len - cornel_size)) and (
                #         (y < cornel_size) or (y >= square_len - cornel_size)):
                    belong[gid] = -2

        # show_grid_tmp(np.arange(N, dtype='int'), belong, square_len, "fold-1.png")
        tmp_grids = np.arange(N, dtype='int')
        consider = np.ones(N, dtype='bool')
        consider = np.logical_and(consider, (belong >= 0))
        reduce = np.array(gridlayoutOpt.getMainConnectGrids(tmp_grids, belong, consider))
        belong[np.logical_and(reduce==False, belong >= 0)] = -1

        # show_grid_tmp(np.arange(N, dtype='int'), belong, square_len, "fold0.png")
        unfilled = []
        row_asses = np.zeros(N, dtype='int')
        label_used = np.zeros(maxLabel, dtype='int')
        element_used = np.zeros(num, dtype='bool')
        is_unfill = np.zeros(N, dtype='bool')
        dist_label = np.full((N, maxLabel), fill_value=1000)
        for gid in range(N):
            if belong[gid] >= 0:
                dist_label[gid][belong[gid]] = -1000

        # print("time", time.time() - start)

        neighbor = 1
        while True:
            # print("neighbor", neighbor)
            fine = True
            unfilled = []
            row_asses = np.zeros(N, dtype='int')
            label_used = np.zeros(maxLabel, dtype='int')
            element_used = np.zeros(num, dtype='bool')

            for x in range(square_len):
                bias = x * square_len
                for y in range(square_len):
                    gid = bias + y
                    unfill = False
                    if belong[gid]<0:
                        unfill = True
                    for xx in range(2 * neighbor + 1):
                        x2 = x + xx - neighbor
                        if (x2 < 0) or (x2 >= square_len):
                            continue
                        bias2 = x2 * square_len
                        ran = range(2 * neighbor + 1)
                        if 0<xx<2*neighbor:
                            ran = [0, 2*neighbor]
                        for yy in ran:
                            y2 = y + yy - neighbor
                            if (y2 < 0) or (y2 >= square_len):
                                continue
                            gid2 = bias2 + y2
                            if belong[gid2] == -2:
                                continue
                            if belong[gid] != belong[gid2]:
                                unfill = True
                                if belong[gid] >= 0:
                                    dist_label[gid][belong[gid]] = max(dist_label[gid][belong[gid]], -max(abs(xx-neighbor), abs(yy-neighbor)))
                                if belong[gid2] >= 0:
                                    dist_label[gid][belong[gid2]] = min(dist_label[gid][belong[gid2]], max(abs(xx-neighbor), abs(yy-neighbor)))
                    if unfill:
                        is_unfill[gid] = True

                    if not is_unfill[gid]:
                        lb = belong[gid]
                        label_used[lb] += 1
                        if label_used[lb] <= elements[lb].shape[0]:
                            row_asses[gid] = elements[lb][label_used[lb] - 1]
                            element_used[row_asses[gid]] = True
                        else:
                            fine = False
                    else:
                        unfilled.append(gid)

            if not fine:
                neighbor += 1
                continue

            # print('try to assign')

            unfilled = np.array(unfilled)
            if unfilled.shape[0]>0:
                grid_positions = grids[unfilled]
                unused = np.arange(num, dtype='int')[(element_used == False)]
                # print(unfilled, unused)
                center_positions = centers[labels[unused]]
                original_cost_matrix = cdist(grid_positions, center_positions, "euclidean")
                original_cost_matrix = np.power(original_cost_matrix, 2)*0.01
                for i in range(unused.shape[0]):
                    dist = dist_label[unfilled, labels[unused[i]]]/square_len
                    dist = (1-2*(dist<0).astype('int'))*np.power(dist, 2)
                    original_cost_matrix[:, i] += dist
                dummy_points = np.ones((N - num, 2)) * 0.5
                dummy_vertices = (1 - cdist(grid_positions, dummy_points, "euclidean")) * 100
                cost_matrix = np.concatenate((original_cost_matrix, dummy_vertices), axis=1)
                cost_matrix -= cost_matrix.min()
                opt = gridlayoutOpt.Optimizer(0)
                row_asses2 = np.array(opt.solveKM(cost_matrix))

                # print(row_asses2)

                dummy_id = np.arange(N-num, dtype='int')+num
                unused = np.concatenate((unused, dummy_id), axis=0)

                flag = False
                for i in range(len(row_asses2)):
                    if (unused[row_asses2[i]]<num)and(dist_label[unfilled[i]][labels[unused[row_asses2[i]]]]>999):
                        flag = True
                        break
                    row_asses[unfilled[i]] = unused[row_asses2[i]]
                if flag:
                    # print('fail')
                    neighbor += 1
                    continue

            # print('success')
            break

        # print("time", time.time() - start)

        # labels2 = labels.copy()*2
        # for i in range(len(row_asses2)):
        #     if unused[row_asses2[i]]<num:
        #         labels2[unused[row_asses2[i]]] += 1
        # show_grid_tmp(row_asses, labels2, square_len, "fold0.png")
        # show_grid_tmp(row_asses, labels, square_len, "fold1.png")

        tmp_partition = np.ones(N, dtype='int')
        for i in range(N):
            if(row_asses[i]<num):
                tmp_partition[i] = 0
        row_asses = np.array(gridlayoutOpt.grid_op_partition(1, tmp_partition,
                                                                    np.zeros((N,2)), np.zeros((N,2)), row_asses, labels, False, True,
                                                                    "Edges", 0, 1, False, 2147483647, 1, []))
        # show_grid_tmp(row_asses, labels, square_len, "fold2.png")

        belong = np.ones(N, dtype='int')*(-1)
        for i in range(N):
            if (row_asses[i]<num):
                belong[i] = labels[row_asses[i]]
        tmp_grids = np.arange(N, dtype='int')
        consider = np.ones(N, dtype='bool')
        consider = np.logical_and(consider, (belong >= 0))
        reduce = np.array(gridlayoutOpt.getMainConnectGrids(tmp_grids, belong, consider))
        if np.logical_and((reduce==False), consider).sum()==0:
            break
        else:
            again_cnt += 1
            # print("again !!!", again_cnt)

    major_coords = gridlayoutOpt.getConnectShape(row_asses, labels, [True]*num)
    for i in range(len(major_coords)):
        for j in range(len(major_coords[i])):
            major_coords[i][j] = np.array(major_coords[i][j])
    cells = []
    for lb in range(maxLabel):
        coords_list = major_coords[lb]
        new_coords = merge_coords(coords_list)
        cells.append(Polygon(new_coords))

    # print("time", time.time() - start)

    return row_asses, cells


def getCutGrids(labels, centers, square_len, cut_ways, now_hull=None, now_grids=None):

    start = time.time()

    N = square_len * square_len
    # print('Cut N', N)
    num = labels.shape[0]
    maxLabel = centers.shape[0]
    capacity = np.zeros(maxLabel)
    for i in range(maxLabel):
        capacity[i] = (labels == i).sum() / num
    cluster_centers = centers.copy()

    now_N = N
    if now_grids is not None:
        now_N = now_grids.sum()

    hull = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
    if now_hull is not None:
        hull = now_hull
    if maxLabel > 1:
        cells = getCutDiagram(hull, capacity, cut_ways)
    else:
        cells = [hull]

    for i in range(maxLabel):
        centers[i] = np.array([cells[i].centroid.x, cells[i].centroid.y])

    bounds = []
    for i in range(len(cells)):
        bounds.append(cells[i].bounds)

    if now_grids is None:
        text = ""
    else:
        text = str(now_grids.sum())

    # print("time", time.time() - start)

    cornel_size = 0
    while cornel_size * (cornel_size + 1) * 2 < now_N - num:
        cornel_size += 1

    elements = []
    for i in range(maxLabel):
        elements.append(np.arange(num, dtype='int')[(labels == i)])

    grids = np.dstack(np.meshgrid(np.linspace(0, 1 - 1.0 / square_len, square_len),
                                  np.linspace(0, 1 - 1.0 / square_len, square_len))) \
        .reshape(-1, 2)

    tmp = grids[:, 0].copy()
    grids[:, 0] = grids[:, 1]
    grids[:, 1] = tmp
    grids = grids + np.array([1.0, 1.0]) / 2 / square_len

    belong = np.ones(N, dtype='int')*-1

    if maxLabel>1:
        for i in range(grids.shape[0]):
            if (now_grids is not None) and (not now_grids[i]):
                continue
            list = []
            for id, cell in enumerate(cells):
                if grids[i][0]<bounds[id][0] or grids[i][0]>bounds[id][2] or grids[i][1]<bounds[id][1] or grids[i][1]>bounds[id][3]:
                    continue
                if cell.covers(Point(grids[i])):
                    list.append(id)
            if len(list)>0:
                belong[i] = random.choice(list)
            else:
                belong[i] = -1
    else:
        belong = np.zeros(N, dtype='int')

    centers = centers + np.random.normal(0, 1e-4, (len(centers), 2))

    # print("time", time.time() - start)

    again_cnt = 0
    while again_cnt < 1:
        # show_grid_tmp(np.arange(N, dtype='int'), belong, square_len, "fold-2-"+text+".png")
        for x in range(square_len):
            bias = x * square_len
            for y in range(square_len):
                gid = bias + y
                dx = min(x, square_len-1-x)
                dy = min(y, square_len-1-y)
                if dx+dy<cornel_size:
                # if ((x < cornel_size) or (x >= square_len - cornel_size)) and (
                #         (y < cornel_size) or (y >= square_len - cornel_size)):
                    belong[gid] = -2
        # show_grid_tmp(np.arange(N, dtype='int'), belong, square_len, "fold-1-"+text+".png")

        now_centers = []
        for cell in cells:
            now_centers.append(cellCentroid(cell))
        now_centers = np.array(now_centers)
        full_g = np.arange(N, dtype='int')
        if now_grids is not None:
            full_g = full_g[now_grids]
        tmp_dist = np.power(cdist(now_centers, grids[full_g], "euclidean"), 2)
        opt = gridlayoutOpt.Optimizer(0)
        tmp_center = opt.solveKM(tmp_dist)
        for i in range(maxLabel):
            belong[full_g[tmp_center[i]]] = i
        tmp_grids = np.arange(N, dtype='int')
        consider = np.ones(N, dtype='bool')
        if now_grids is not None:
            consider = now_grids
        consider = np.logical_and(consider, (belong >= 0))
        reduce = np.array(gridlayoutOpt.getMainConnectGrids(tmp_grids, belong, consider))
        belong[np.logical_and(reduce==False, belong >= 0)] = -1
        # show_grid_tmp(np.arange(N, dtype='int'), belong, square_len, "fold0-"+text+".png")
        unfilled = []
        row_asses = np.zeros(N, dtype='int')
        label_used = np.zeros(maxLabel, dtype='int')
        element_used = np.zeros(num, dtype='bool')
        is_unfill = np.zeros(N, dtype='bool')
        dist_label = np.full((N, maxLabel), fill_value=1000)
        for gid in range(N):
            if belong[gid]>=0:
                dist_label[gid][belong[gid]] = -1000

        # print("time", time.time() - start)

        neighbor = 1
        while True:
            # print("neighbor", neighbor)
            fine = True
            unfilled = []
            row_asses = np.zeros(N, dtype='int')
            label_used = np.zeros(maxLabel, dtype='int')
            element_used = np.zeros(num, dtype='bool')

            for x in range(square_len):
                bias = x * square_len
                for y in range(square_len):
                    gid = bias + y
                    if (now_grids is not None) and (not now_grids[gid]):
                        continue
                    unfill = False
                    if belong[gid]<0:
                        unfill = True
                    for xx in range(2 * neighbor + 1):
                        x2 = x + xx - neighbor
                        if (x2 < 0) or (x2 >= square_len):
                            continue
                        bias2 = x2 * square_len
                        ran = range(2 * neighbor + 1)
                        if 0<xx<2*neighbor:
                            ran = [0, 2*neighbor]
                        for yy in ran:
                            y2 = y + yy - neighbor
                            if (y2 < 0) or (y2 >= square_len):
                                continue
                            gid2 = bias2 + y2
                            if (now_grids is not None) and (not now_grids[gid2]):
                                continue
                            if belong[gid2] == -2:
                                continue
                            if belong[gid] != belong[gid2]:
                                unfill = True
                                if belong[gid]>=0:
                                    dist_label[gid][belong[gid]] = max(dist_label[gid][belong[gid]], -max(abs(xx-neighbor), abs(yy-neighbor)))
                                if belong[gid2]>=0:
                                    dist_label[gid][belong[gid2]] = min(dist_label[gid][belong[gid2]], max(abs(xx-neighbor), abs(yy-neighbor)))
                    if unfill:
                        is_unfill[gid] = True

                    if not is_unfill[gid]:
                        lb = belong[gid]
                        label_used[lb] += 1
                        if label_used[lb] <= elements[lb].shape[0]:
                            row_asses[gid] = elements[lb][label_used[lb] - 1]
                            element_used[row_asses[gid]] = True
                        else:
                            fine = False
                    else:
                        unfilled.append(gid)

            if not fine:
                neighbor += 1
                continue

            # print('try to assign')

            unfilled = np.array(unfilled)
            if unfilled.shape[0]>0:
                grid_positions = grids[unfilled]
                unused = np.arange(num, dtype='int')[(element_used == False)]
                # print(unfilled, unused)
                center_positions = centers[labels[unused]]
                original_cost_matrix = cdist(grid_positions, center_positions, "euclidean")
                original_cost_matrix = np.power(original_cost_matrix, 2)*0.01
                for i in range(unused.shape[0]):
                    original_cost_matrix[:, i] += dist_label[unfilled, labels[unused[i]]]/square_len
                dummy_points = np.ones((now_N - num, 2)) * 0.5
                dummy_vertices = (1 - cdist(grid_positions, dummy_points, "euclidean")) * 100
                cost_matrix = np.concatenate((original_cost_matrix, dummy_vertices), axis=1)
                cost_matrix -= cost_matrix.min()
                opt = gridlayoutOpt.Optimizer(0)

                # row_asses2 = np.array(opt.solveKM(cost_matrix))

                original_cost_matrix2 = cdist(grid_positions, centers, "euclidean")
                original_cost_matrix2 = np.power(original_cost_matrix2, 2)*0.01
                for i in range(maxLabel):
                    dist = dist_label[unfilled, i]
                    original_cost_matrix2[:, i] += dist/square_len
                dummy_points2 = np.ones((1, 2)) * 0.5
                dummy_vertices2 = (1 - cdist(grid_positions, dummy_points2, "euclidean")) * 100
                cost_matrix2 = np.concatenate((original_cost_matrix2, dummy_vertices2), axis=1)
                cost_matrix2 = np.power(cost_matrix2, 2)
                labels2 = labels[unused]
                labels2 = np.concatenate((labels2, np.ones(now_N - num, dtype='int') * maxLabel), axis=0)
                row_asses2 = np.array(opt.solveKMLabel(cost_matrix2, labels2))
                dummy_id = np.arange(now_N-num, dtype='int')+num
                unused = np.concatenate((unused, dummy_id), axis=0)

                flag = False
                for i in range(len(row_asses2)):
                    if (unused[row_asses2[i]]<num)and(dist_label[unfilled[i]][labels[unused[row_asses2[i]]]]>999):
                        flag = True
                        break
                    row_asses[unfilled[i]] = unused[row_asses2[i]]
                if flag:
                    # print('fail')
                    neighbor += 1
                    continue

            # print('success')
            break

        # labels2 = labels.copy()*2
        # for i in range(len(row_asses2)):
        #     if unused[row_asses2[i]]<num:
        #         labels2[unused[row_asses2[i]]] += 1
        # show_grid_tmp(row_asses, labels2, square_len, "fold0-"+str(now_grids.sum())+".png")
        # show_grid_tmp(row_asses, labels, square_len, "fold1-"+text+".png")
        tmp_partition = np.ones(N, dtype='int')
        tmp_labels = np.zeros(num+1, dtype='int')
        for i in range(num):
            tmp_labels[i] = labels[i]
        tmp_labels[num] = labels.max()+1
        for i in range(N):
            if (now_grids is not None) and (not now_grids[i]):
                row_asses[i] = num
                continue
            if(row_asses[i]<num):
                tmp_partition[i] = 0
        row_asses = np.array(gridlayoutOpt.grid_op_partition(1, tmp_partition,
                                                                    np.zeros((N,2)), np.zeros((N,2)), row_asses, labels, False, True,
                                                                    "Edges", 0, 1, False, 2147483647, 1, []))
        # show_grid_tmp(row_asses, labels, square_len, "fold2-"+text+".png")
        belong = np.ones(N, dtype='int')*(-1)
        for i in range(N):
            if (row_asses[i]<num):
                belong[i] = labels[row_asses[i]]

        # print("time", time.time() - start)

        tmp_grids = np.arange(N, dtype='int')
        consider = np.ones(N, dtype='bool')
        if now_grids is not None:
            consider = now_grids
        consider = np.logical_and(consider, (belong >= 0))
        reduce = np.array(gridlayoutOpt.getMainConnectGrids(tmp_grids, belong, consider))
        if np.logical_and((reduce==False), consider).sum()==0:
            break
        else:
            again_cnt += 1
            # print("again !!!", again_cnt)

    # show_grid_tmp(row_asses, labels, square_len, "HV.png")

    # print("time", time.time() - start)

    return row_asses, cells


def rotateEmbedded(X_embedded, labels, grid_centers):

    start = time.time()
    num = labels.shape[0]
    maxLabel = grid_centers.shape[0]
    capacity = np.zeros(maxLabel)
    for i in range(maxLabel):
        capacity[i] = (labels == i).sum() / num
    cluster_centers = grid_centers.copy()

    hull = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
    w = find_W(cluster_centers, np.zeros(len(cluster_centers)), capacity*hull.area, hull)
    cells = computePowerDiagramByCGAL(positions=cluster_centers, weights=w, hull=hull)
    label_coords = {}
    for lb in range(maxLabel):
        label_coords.update({lb: np.array(cells[lb].exterior.coords)})

    skeleton, edges = get_skeleton(label_coords, hull=None, withEdges=True)
    for i in range(len(edges)):
        u, v, x, y, w = edges[i]
        dx = cluster_centers[u][0] - cluster_centers[v][0]
        dy = cluster_centers[u][1] - cluster_centers[v][1]
        dz = np.sqrt(dx*dx+dy*dy)
        dx /= dz
        dy /= dz
        if x*dy-y*dx < 0:
            x = -x
            y = -y
        # x += dy
        # y -= dx
        # z = np.sqrt(x*x+y*y)
        # x /= z
        # y /= z
        edges[i] = (u, v, x, y, w)

    ori_centers = np.zeros((maxLabel, 2))
    for lb in range(maxLabel):
        ori_centers[lb] = X_embedded[labels==lb].mean(axis=0)
    centers = ori_centers.copy()

    # print("time rotate 1", time.time()-start)

    best_theta = 0
    best_cost = -1
    for rid in range(16):
        r = rid*2*math.pi/16
        for lb in range(maxLabel):
            tmp_x = ori_centers[lb][0]
            tmp_y = ori_centers[lb][1]
            centers[lb][0] = tmp_x*math.cos(r) - tmp_y*math.sin(r)
            centers[lb][1] = tmp_x*math.sin(r) + tmp_y*math.cos(r)
        cost = 0
        for (u, v, x, y, w) in edges:
            tmpx = centers[u][0]-centers[v][0]
            tmpy = centers[u][1]-centers[v][1]
            tmpz = np.sqrt(tmpx*tmpx + tmpy*tmpy)
            tmpx2 = tmpx / tmpz
            tmpy2 = tmpy / tmpz
            if x*tmpy2-y*tmpx2 >= 0:
                cost += np.power(abs(tmpx2*x+tmpy2*y), 2)*w
            else:
                cost += np.power(2*np.sqrt(x*x+y*y)-abs(tmpx2*x+tmpy2*y), 2)*w
        # print("theta", r, "cost", cost)
        if (best_cost==-1) or (cost<best_cost):
            best_theta = r
            best_cost = cost
    # print('rotate theta', best_theta)

    # print("time rotate 2", time.time()-start)

    new_X_embedded = X_embedded.copy()
    for i in range(X_embedded.shape[0]):
        tmp_x = X_embedded[i][0] - 0.5
        tmp_y = X_embedded[i][1] - 0.5
        new_X_embedded[i][0] = tmp_x*math.cos(best_theta) - tmp_y*math.sin(best_theta) + 0.5
        new_X_embedded[i][1] = tmp_x*math.sin(best_theta) + tmp_y*math.cos(best_theta) + 0.5
    # print("time rotate 3", time.time()-start)
    return new_X_embedded
