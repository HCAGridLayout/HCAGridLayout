
class MyColorMap(object):
    def __init__(self) -> None:
        self.size = 15
        self.rgbs = [
            (171, 227, 246),
            (248, 182, 187),
            (185, 243, 190),
            (243, 228, 191),
            (216, 193, 246),
            (252, 245, 155),
            (221, 221, 221),
            (138, 170, 208),
            (191, 185, 134),
            (255, 193, 152),
            (127, 137, 253),
            (255, 136, 104),
            (175, 203, 191),
            (170, 167, 188),
            (254, 228, 179)
        ]
        self.rgbs1 = list(map(lambda x: list(map(lambda v: v / 255, x)), self.rgbs))
    
    def colorSet(self, num):
        return self.rgbs1[:num]

    def color(self, num):
        if num < 0:
            return (1, 1, 1)
        if num >= 15:
            return (0, 0, 0)
        return self.rgbs1[num]

# paint tsne scatter for test
import numpy as np
from matplotlib import pyplot as plt
class ColorScatter(object):
    def __init__(self, num = 10) -> None:
        self.num = num
        self.colorMap = MyColorMap()
        self.colors = self.colorMap.colorSet(self.num)
    
    def drawScatter(self, X, label, filename, highlight = None) -> None:
        # matplotlib draw scatter polt
        # X: [num X 2]
        # label: [num]
        plt.cla()
        ulabel = np.unique(label)
        print(ulabel)
        for i in ulabel:
            plt.scatter(X[label == i, 1], -X[label == i, 0], color=plt.cm.tab20(i))
        plt.legend(ulabel)
        plt.savefig(filename)
