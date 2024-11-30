from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import alphashape
from shapely.geometry import Polygon
from shapely.geometry import Point
from scipy.spatial.distance import cdist
# import gridlayoutOpt
import grid.gridlayoutOpt as gridlayoutOpt
from grid.colors import MyColorMap
import matplotlib.pyplot as plt
import time
import math

def get_hull_area(X):
    clf = LocalOutlierFactor(contamination=0.05)
    flag = clf.fit_predict(X)
    inliers = X[(flag == 1)]


    alpha_shape = alphashape.alphashape(inliers, 0)
    area = alpha_shape.area

    # plt.scatter(X[:, 1], X[:, 0])
    # x, y = alpha_shape.exterior.xy
    # plt.plot(y, x)
    # plt.show()

    return area


def show_grid_best(row_asses, grid_labels, height, width, path='new.png', scatter=None, showNum=False, just_save=False):
    def highlight_cell(x, y, ax=None, **kwargs):
        rect = plt.Rectangle((x - .5, y - .5), 1, 1, fill=False, **kwargs)
        ax = ax or plt.gca()
        ax.add_patch(rect)
        return rect

    cm = MyColorMap()
    data = []
    for i in range(height - 1, -1, -1):
        row = []
        for j in range(width):
            row.append(cm.color(grid_labels[row_asses[width * i + j]]))
        data.append(row)
    plt.cla()
    plt.imshow(data)
    for i in range(height - 1, -1, -1):
        for j in range(width):
            highlight_cell(j, i, color="white", linewidth=1)
    if showNum:
        for i in range(height):
            for j in range(width):
                # text = plt.text(j, num - i - 1, row_asses[num * i + j], fontsize=7, ha="center", va="center",
                #                 color="w")
                text = plt.text(j, height - i - 1, grid_labels[row_asses[width * i + j]], fontsize=7, ha="center",
                                va="center",
                                color="w")
    if scatter is not None:
        for i in range(scatter.shape[0]):
            plt.scatter(scatter[i, 1] * width, height - height * scatter[i, 0] - 1, color=cm(grid_labels[i]))
    plt.axis('off')
    plt.savefig(path)
    if not just_save:
        plt.show()

def solveKM(cost_matrix):
    # print('start KM')
    # row_asses, _, _ = lapjv.lapjv(cost_matrix)
    row_asses = np.array(gridlayoutOpt.Optimizer(0).solveKM(cost_matrix))
    N = row_asses.shape[0]
    col_asses = np.zeros(shape=N, dtype='int')
    for i in range(N):
        col_asses[row_asses[i]] = i
    # print('end KM')
    return row_asses, col_asses

def getBestResult(X, square_len, label=0):
    clf = LocalOutlierFactor(contamination=0.05)
    flag = clf.fit_predict(X)
    inliers = X[(flag == 1)]
    outliers = X[(flag == -1)]

    alpha_shape = alphashape.alphashape(inliers, 0)
    area = alpha_shape.area

    # plt.scatter(X[:, 1], X[:, 0])
    # x, y = alpha_shape.exterior.xy
    # plt.plot(y, x)
    # plt.show()

    coords = np.array(alpha_shape.exterior.coords) * np.sqrt(1 / area * X.shape[0] / square_len / square_len)
    coords -= coords.min(axis=0)
    coords += coords.max(axis=0) * 0.01

    width = np.arange(coords.max(axis=0)[1] * square_len).shape[0]
    height = np.arange(coords.max(axis=0)[0] * square_len).shape[0]
    grids = np.dstack(np.meshgrid(np.arange(coords.max(axis=0)[1] * square_len) / square_len,
                                  np.arange(coords.max(axis=0)[0] * square_len) / square_len)) \
        .reshape(-1, 2)

    tmp = grids[:, 0].copy()
    grids[:, 0] = grids[:, 1]
    grids[:, 1] = tmp

    dist = cdist([coords.max(axis=0) / 2], grids, "euclidean")[0]
    order = np.argsort(dist)
    hull = Polygon(coords)
    cover = np.zeros(grids.shape[0], dtype='bool')
    for i in range(grids.shape[0]):
        if hull.covers(Point(grids[i])):
            cover[i] = True
    cnt = cover.sum()
    id = grids.shape[0] - 1
    while cnt > X.shape[0]:
        if cover[order[id]]:
            cover[order[id]] = False
            cnt -= 1
        id -= 1
    id = 0
    while cnt < X.shape[0]:
        if not cover[order[id]]:
            cover[order[id]] = True
            cnt += 1
        id += 1
    cost_matrix = cdist(grids, X*np.sqrt(1/area*X.shape[0]/square_len/square_len), "euclidean")
    cost_matrix = np.power(cost_matrix, 2)
    cost_matrix[(cover==False), :] += 10000

    dummy_vertices = np.ones((grids.shape[0], grids.shape[0]-X.shape[0])) * 10000
    cost_matrix = np.concatenate((cost_matrix, dummy_vertices), axis=1)

    row_asses, col_asses = solveKM(cost_matrix)
    # for i in range(X.shape[0]):
    #     print(X[i], grids[col_asses[i]])
    labels = np.ones(row_asses.shape[0], dtype="int")*(-1)
    for i in range(X.shape[0]):
        labels[i] = label
    # show_grid_best(row_asses, labels, height, width)

    return grids[col_asses[:X.shape[0]]]



if __name__ == '__main__':
    square_len = 20
    X = np.array([[0.01, 0.01], [0.01, 0.02], [0.02, 0.01], [0.02, 0.02], [0.03, 0.01], [0.03, 0.02]])
    getBestResult(X, square_len)
    # grids = np.dstack(np.meshgrid(np.arange(1.01*square_len)/square_len,
    #                               np.arange(2.01*square_len)/square_len)) \
    #     .reshape(-1, 2)
    #
    # tmp = grids[:, 0].copy()
    # grids[:, 0] = grids[:, 1]
    # grids[:, 1] = tmp
    #
    # print(grids)
