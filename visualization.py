import itertools
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from geometry import Point, sort_hull_points


def plot_points(points: list[Point], ax: Axes | None = None):
    x = [p.x for p in points]
    y = [p.y for p in points]
    if ax is None:
        plt.scatter(x, y)
    else:
        ax.scatter(x, y)


def plot_graph(g, upper: bool):
    plt.clf()
    adj_list = g.adjacency_list[upper]
    xs, ys = [], []
    for pt, node in adj_list.items():
        xs.append(pt.x)
        ys.append(pt.y)

        for pt2 in node.left:
            plt.plot([pt.x, pt2.x], [pt.y, pt2.y])
            plt.text((pt.x + pt2.x) / 2, (pt.y + pt2.y) / 2, s=pt2.rank)
        for pt2 in node.right:
            plt.plot([pt.x, pt2.x], [pt.y, pt2.y])
            plt.text((pt.x + pt2.x) / 2, (pt.y + pt2.y) / 2, s=pt2.rank)
    plt.scatter(xs, ys)
    plt.grid()


def plot_layers(layers, full: bool):
    sort_func = sort_hull_points if full else sorted
    clrs = ['r', 'g', 'b', 'm', 'c', 'y', 'k']
    color_cycle = itertools.cycle(clrs)

    for layer in layers:
        xs = []
        ys = []
        clr = next(color_cycle)
        layer = sort_func(layer)
        if full:
            layer.append(layer[0])
        for i, pt in enumerate(layer):
            xs.append(pt.x)
            ys.append(pt.y)
            if i + 1 < len(layer):
                plt.plot([pt.x, layer[i + 1].x], [pt.y, layer[i + 1].y], c=clr)
        plt.scatter(xs, ys, c=clr, s=2)