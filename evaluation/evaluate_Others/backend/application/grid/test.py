import copy
import numpy as np

from shapely.geometry import Polygon, MultiPolygon, LineString, MultiPoint
from shapely import distance, Point

point_array = np.zeros((0, 2))
points = MultiPoint([Point(x, y) for x, y in point_array])
print(points.convex_hull)

polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
polygon2 = Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])


intersection = polygon.intersection(polygon2)
print(intersection)
