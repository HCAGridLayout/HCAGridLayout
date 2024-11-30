from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d, convolve1d
import time
from lsa import linear_sum_assignment as lsa
#from functools import cmp_to_key
import os

def squared_l2_distance(q, p):
    ps = np.sum(p * p, axis=-1, keepdims=True)
    qs = np.sum(q * q, axis=-1, keepdims=True)
    distance = ps - 2 * np.matmul(p, q.T) + qs.T
    return np.clip(distance, 0, np.inf)


def low_pass_filter(map_image, filter_size_x, filter_size_y, wrap=False):
    mode = "wrap" if wrap else "reflect"  # nearest

    im2 = uniform_filter1d(map_image, filter_size_y, axis=0, mode=mode)
    im2 = uniform_filter1d(im2, filter_size_x, axis=1, mode=mode)
    return im2


def get_positions_in_radius(pos, indices, r, nc, wrap):
    if wrap:
        return get_positions_in_radius_wrapped(pos, indices, r, nc)
    else:
        return get_positions_in_radius_non_wrapped(pos, indices, r, nc)


def get_positions_in_radius_non_wrapped(pos, indices, r, nc):
    H, W = indices.shape

    x = pos % W
    y = int(pos / W)

    ys = y - r
    ye = y + r + 1
    xs = x - r
    xe = x + r + 1

    # move position so the full radius is inside the images bounds
    if ys < 0:
        ys = 0
        ye = min(2 * r + 1, H)

    if ye > H:
        ye = H
        ys = max(H - 2 * r - 1, 0)

    if xs < 0:
        xs = 0
        xe = min(2 * r + 1, W)

    if xe > W:
        xe = W
        xs = max(W - 2 * r - 1, 0)

    # concatenate the chosen position to a 1D array
    positions = np.concatenate(indices[ys:ye, xs:xe])

    if nc is None:
        return positions

    chosen_positions = np.random.choice(positions, min(nc, len(positions)), replace=False)

    return chosen_positions


def get_positions_in_radius_wrapped(pos, extended_grid, r, nc):
    H, W = extended_grid.shape

    # extended grid shape is H*2, W*2
    H, W = int(H / 2), int(W / 2)
    x = pos % W
    y = int(pos / W)

    ys = (y - r + H) % H
    ye = ys + 2 * r + 1
    xs = (x - r + W) % W
    xe = xs + 2 * r + 1

    # concatenate the chosen position to a 1D array
    positions = np.concatenate(extended_grid[ys:ye, xs:xe])

    if nc is None:
        return positions

    chosen_positions = np.random.choice(positions, min(nc, len(positions)), replace=False)

    return chosen_positions


def sort_with_las(grid_asses, X, radius_factor=0.9, wrap=False):
    # for reproducible sortings
    np.random.seed(7)

    n_images_per_site = round(np.sqrt(len(grid_asses)))
    N = n_images_per_site*n_images_per_site
    num = len(X)
    X = np.concatenate((X, np.zeros((N-num, X.shape[1]))), axis=0)
    X_o = X

    X = X.reshape((n_images_per_site, n_images_per_site, -1))

    # setup of required variables
    grid_shape = X.shape[:-1]
    H, W = grid_shape
    start_time = time.time()
    # assign input vectors to random positions on the grid
    grid = np.random.permutation(X.reshape((N, -1))).reshape((X.shape)).astype(float)
    # reshape 2D grid to 1D
    flat_X = X.reshape((N, -1))
    grid_asses = np.arange(N, dtype='int')

    radius_f = max(H, W) / 2 - 1

    while True:
        print(".", end="")
        # compute filtersize that is not larger than any side of the grid
        radius = int(np.round(radius_f))

        filter_size_x = min(W - 1, int(2 * radius + 1))
        filter_size_y = min(H - 1, int(2 * radius + 1))
        # print (f"radius {radius_f:.2f} Filter size: {filter_size_x}")

        # Filter the map vectors using the actual filter radius
        grid = low_pass_filter(grid, filter_size_x, filter_size_y, wrap=wrap)

        # tot = 0
        # for i in range(H):
        #     for j in range(W):
        #         if i < H-1:
        #             tot += np.abs(grid[i][j]-grid[i+1][j]).sum()
        #         if j < W-1:
        #             tot += np.abs(grid[i][j]-grid[i][j+1]).sum()
        # print("tot1", tot)

        flat_grid = grid.reshape((N, -1))

        pixels = flat_X
        grid_vecs = flat_grid
        C = squared_l2_distance(pixels, grid_vecs)
        C = (C / C.max() * 2048).astype(int).T

        tmp, best_perm_indices = lsa(C)

        flat_X = pixels[best_perm_indices]
        grid_asses = grid_asses[best_perm_indices]

        grid = flat_X.reshape(X.shape)

        # # print(X_o[grid_asses[0]]-grid[0][0])
        # tot = 0
        # for i in range(H):
        #     for j in range(W):
        #         if i < H-1:
        #             tot += np.abs(grid[i][j]-grid[i+1][j]).sum()
        #         if j < W-1:
        #             tot += np.abs(grid[i][j]-grid[i][j+1]).sum()
        # print("tot2", tot)

        radius_f *= radius_factor
        if radius_f < 1:
            break

    print(f"\nSorted with LAS in {time.time() - start_time:.3f} seconds")

    return grid, grid_asses