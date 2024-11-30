import math
from matplotlib import pyplot as plt

line_width = 1
def show_grid(row_asses, grid_labels, square_len, color_map, path='new.png', just_save=False):
    def highlight_cell(x, y, ax=None, **kwargs):
        rect = plt.Rectangle((x - .5, y - .5), 1, 1, fill=False, **kwargs)
        ax = ax or plt.gca()
        ax.add_patch(rect)
        return rect

    data = []
    num = math.ceil(square_len)
    for i in range(num):
        row = []
        for j in range(num):
            row.append(color_map[grid_labels[row_asses[num * i + j]]])
        data.append(row)
    plt.cla()
    fig = plt.figure(figsize=(12, 12))
    plt.imshow(data)
    for i in range(num):
        for j in range(num):
            highlight_cell(i, j, color="white", linewidth=line_width)
    plt.axis('off')
    plt.savefig(path)
    if not just_save:
        plt.show()