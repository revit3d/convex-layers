import math

from dataclasses import dataclass, field
from geometry import Point, RankedPoint, collinear, slope, convex_hull_andrew


@dataclass
class HullTreeNode:
    left: list[RankedPoint] = field(default_factory=list)
    right: list[RankedPoint] = field(default_factory=list)


class HullTree:
    def __init__(self):
        self.eps: float = 1e-12
        self.tree_height: int = 0
        self.adjacency_list: tuple[dict[Point, HullTreeNode]] | None = None

    @staticmethod
    def pad_convex(convex: list[Point], upper: bool) -> list[Point]:
        """
        Pad convex hull with vertical edges.
        This is done for uniform comparison of all points
        in convex hulls during tangent search.
        """
        sgn = -1 if upper else 1
        padded_convex = convex.copy()

        v_first = convex[0]
        padded_convex.insert(0, Point(v_first.x, v_first.y + sgn * 10))

        v_last = convex[-1]
        padded_convex.append(Point(v_last.x, v_last.y + sgn * 10))

        return padded_convex

    def tangent(
        self,
        left_convex: list[Point],
        right_convex: list[Point],
        upper: bool
    ) -> tuple[Point, Point]:
        """
        Find upper (lower) tangent of two convex hulls.
        Assuming both convex hulls are sorted by x, y
        and max x coordinate from the left convex < min x coordinate from the right convex.

        Time complexity: O(n + m), where n and m are the lengths of convex hulls.
        """
        sgn = 1 if upper else -1

        right_convex_pad = self.pad_convex(right_convex, upper=upper)
        left_convex_pad = self.pad_convex(left_convex, upper=upper)

        left_idx, right_idx = len(left_convex), 1
        update = True
        while update:
            update = False
            while (
                right_idx + 1 < len(right_convex_pad)
                and (
                    sgn * slope(left_convex_pad[left_idx], right_convex_pad[right_idx - 1])
                    > sgn * slope(left_convex_pad[left_idx], right_convex_pad[right_idx]) + self.eps
                    or sgn * slope(left_convex_pad[left_idx], right_convex_pad[right_idx]) + self.eps
                    < sgn * slope(left_convex_pad[left_idx], right_convex_pad[right_idx + 1])
                )
            ):
                right_idx += 1
                update = True

            while (
                left_idx - 1 > 0
                and (
                    sgn * slope(left_convex_pad[left_idx - 1], right_convex_pad[right_idx]) + self.eps
                    < sgn * slope(left_convex_pad[left_idx], right_convex_pad[right_idx])
                    or sgn * slope(left_convex_pad[left_idx], right_convex_pad[right_idx])
                    > sgn * slope(left_convex_pad[left_idx + 1], right_convex_pad[right_idx]) + self.eps
                )
            ):
                left_idx -= 1
                update = True
        return left_convex[left_idx - 1], right_convex[right_idx - 1]

    def add_edge(self, v1: RankedPoint, v2: RankedPoint, upper: bool, right_pos: int = -1):
        """
        Add edges (v1, v2) and (v2, v1) to hull graph.

        By default, both edges are assumed to be new 'top edges' of each vertex
        and placed on top of adjacency list. If `right_pos` parameter is passed,
        edge (v2, v1) is placed on `right_pos` position in adjacency list of v2.
        """
        swap = False
        if v2 < v1:
            v1, v2 = v2, v1
            swap = True

        v1_key = Point(v1.x, v1.y)
        v2_key = Point(v2.x, v2.y)
        assert v1_key != v2_key, f'Edge has equal vertices: {v1_key}, {v2_key}'

        v1_adj_list = self.adjacency_list[upper][v1_key].right
        v2_adj_list = self.adjacency_list[upper][v2_key].left

        if right_pos == -1:
            v1_adj_list.append(v2)
            v2_adj_list.append(v1)
        else:
            if right_pos is None:
                right_pos = -1
            if swap:
                v1_adj_list.insert(right_pos + 1, v2)
                v2_adj_list.append(v1)
            else:
                v1_adj_list.append(v2)
                v2_adj_list.insert(right_pos + 1, v1)

    def build_graph(self, points: list[Point], upper: bool, level: int = 0):
        """
        Build hull graph from a set of points recursively using divide and conquer strategy.
        Assuming points are already sorted by x and y.

        On each recursion step, we divide a current set of points on left and right half,
        apply algorithm to both halves, and merge them. Merging two subgraphs includes
        searching a tangent which connects lower (upper) subgraphs and forms convex chain
        (resp upper or lower) in resulting graph.

        Time complexity: O(n*log(n))
        """
        self.tree_height = max(self.tree_height, level)

        assert len(points) > 0
        if len(points) == 1:
            return points

        mid = len(points) // 2
        points_left, points_right = points[:mid], points[mid:]
        points_left = self.build_graph(points_left, upper=upper, level=level + 1)
        points_right = self.build_graph(points_right, upper=upper, level=level + 1)

        pt1, pt2 = self.tangent(points_left, points_right, upper=upper)
        v1 = RankedPoint(pt1.x, pt1.y, level)
        v2 = RankedPoint(pt2.x, pt2.y, level)
        self.add_edge(v1, v2, upper=upper)

        if upper:
            return convex_hull_andrew(points, only_upper=True)
        else:
            return convex_hull_andrew(points, only_lower=True)

    @staticmethod
    def intersect(p: Point, p0: Point, p1: Point) -> Point | None:
        """
        Finds intersection of line defined by points {a0, a1} and line x = p.y.
        """
        if p == p0 or p == p1:
            return p

        k = slope(p0, p1)
        if not math.isfinite(k):
            return None

        b = p0.y - k * p0.x
        if not math.isfinite(b):
            return None

        y_inter = k * p.x + b
        if not math.isfinite(y_inter):
            return None

        return Point(p.x, y_inter)

    def wrap(
        self,
        p: Point,
        a: RankedPoint | None,
        b: RankedPoint | None,
        c: RankedPoint,
        pq: bool,
        upper: bool,
        c_pos: int,
    ):
        """
        Calculate next point to wrap. Finds the highest (starred) point,
        where one of the points are being wrapped during p is pulled down,
        and updates points used during the pull operation accordingly.
        """
        sgn = 1 if upper else -1

        a_key = Point(a.x, a.y) if a else None
        b_key = Point(b.x, b.y) if b else None
        c_key = Point(c.x, c.y)

        a_hull = self.adjacency_list[upper][a_key].right if a else None
        b_hull = self.adjacency_list[upper][b_key].left if b else None
        if pq:
            c_hull = self.adjacency_list[upper][c_key].left
        else:
            c_hull = self.adjacency_list[upper][c_key].right

        min_inter = Point(p.x, sgn * float('-inf'))
        a_inter = b_inter = c_inter = ac_inter = bc_inter = min_inter
        if a is not None and len(a_hull) > 0:
            a_inter = self.intersect(p, a, a_hull[-1])  # aa'
            if a_inter is None or a_hull[-1].x > p.x:
                a_inter = min_inter
        if b is not None and len(b_hull) > 0:
            b_inter = self.intersect(p, b, b_hull[-1])  # bb'
            if b_inter is None or b_hull[-1].x < p.x:
                b_inter = min_inter
        if len(c_hull) > 0 and c_pos is not None:
            c_inter = self.intersect(p, c, c_hull[c_pos])  # cc'
            if c_inter is None and sgn * (c_hull[c_pos].y - c.y) > 0:
                c_inter = min_inter
        if a is not None:
            ac_inter = self.intersect(p, a, c)  # ac
        if b is not None:
            bc_inter = self.intersect(p, b, c)  # bc

        inter_points = [a_inter, b_inter, c_inter, ac_inter, bc_inter]
        valid_points = [pt for pt in inter_points if pt is not None]
        for inter_pt in valid_points:
            if sgn * inter_pt.y > sgn * p.y:
                inter_pt = min_inter

        max_inter = max(valid_points) if upper else min(valid_points)

        if max_inter == min_inter and len(valid_points) < len(inter_points):
            # vertical limit
            assert a is None or b is None
            # vertical edge is left by definition,
            # so we wrap it only if it is wrapped by bc, p -> inf
            if c_inter is None:
                return p, a, b, c_hull[c_pos], c_pos
            if not pq and a is not None:
                p_new = p if upper else a
                return p_new, a, b, c, c_pos
            elif pq and b is not None:
                p_new = b if upper else p
                return p_new, a, b, c, c_pos
            else:
                raise RuntimeError()

        assert math.isfinite(max_inter.y)
        p_new, a_new, b_new, c_new = p, a, b, c
        if pq:
            if c_inter is not None and c_inter == max_inter:    # c* point
                c_new = c_hull[c_pos]
                p_new = c_inter
                c_pos = -1
            elif a_inter == max_inter:      # a* point
                a_new = a_hull[-1]
                p_new = a_inter
            elif bc_inter is not None and bc_inter == max_inter:   # eps == 0
                p_new = bc_inter
            elif ac_inter is not None and ac_inter == max_inter:   # delta == 0
                p_new = ac_inter
            elif b_inter == max_inter:    # b* point
                b_new = b_hull[-1]
                p_new = b_inter
            else:
                raise RuntimeError()
        else:
            if c_inter is not None and c_inter == max_inter:    # c* point
                c_new = c_hull[c_pos]
                p_new = c_inter
                c_pos = -1
            elif b_inter == max_inter:    # b* point
                b_new = b_hull[-1]
                p_new = b_inter
            elif ac_inter is not None and ac_inter == max_inter:   # delta == 0
                p_new = ac_inter
            elif bc_inter is not None and bc_inter == max_inter:   # eps == 0
                p_new = bc_inter
            elif a_inter == max_inter:      # a* point
                a_new = a_hull[-1]
                p_new = a_inter
            else:
                raise RuntimeError()
        return p_new, a_new, b_new, c_new, c_pos

    def pull(
        self,
        p: Point,
        a: RankedPoint | None,
        b: RankedPoint | None,
        c: RankedPoint,
        pq: bool,
        upper: bool,
        c_pos: int,
    ):
        """
        Pull point p, wrapping points a', b' and c' with its three edges, pa, pb and pc,
        until any of (a, c) or (b, c) form a single edge (lie in a straight line with p).
        """
        c_rank = c.rank
        if a is None and b is None:
            return
        while not collinear(p, b, c) and not collinear(p, a, c):
            prev = (p, a, b, c)
            p, a, b, c, c_pos = self.wrap(p, a, b, c, pq=pq, upper=upper, c_pos=c_pos)
            if prev == (p, a, b, c):
                break
        while True:
            prev = (p, a, b, c)
            new_state = self.wrap(p, a, b, c, pq=pq, upper=upper, c_pos=c_pos)
            if new_state[:4] == prev or new_state[0] != p:
                break
            p, a, b, c, c_pos = new_state
        if pq:
            if collinear(p, b, c):
                new_b = RankedPoint(b.x, b.y, c_rank)
                new_c = RankedPoint(c.x, c.y, c_rank)
                self.add_edge(new_b, new_c, upper=upper, right_pos=c_pos)
            elif collinear(p, a, c):
                new_a = RankedPoint(a.x, a.y, c_rank)
                new_c = RankedPoint(c.x, c.y, c_rank)
                self.add_edge(new_a, new_c, upper=upper, right_pos=c_pos)
        else:
            if collinear(p, a, c):
                new_a = RankedPoint(a.x, a.y, c_rank)
                new_c = RankedPoint(c.x, c.y, c_rank)
                self.add_edge(new_a, new_c, upper=upper, right_pos=c_pos)
            elif collinear(p, b, c):
                new_b = RankedPoint(b.x, b.y, c_rank)
                new_c = RankedPoint(c.x, c.y, c_rank)
                self.add_edge(new_b, new_c, upper=upper, right_pos=c_pos)

    def try_remove_top(self, p: Point, c: Point, pq: bool, upper: bool):
        """
        Remove edge pc from adjacency list of c
        and return the position of 'top edge' under pc in list.
        """
        c_pos = None
        if pq:
            adj_list = self.adjacency_list[upper][c].left
        else:
            adj_list = self.adjacency_list[upper][c].right

        if adj_list[-1] == p:
            c_pos = -1
            adj_list.pop()
        else:
            c_pos = adj_list.index(p)
            if c_pos == 0:
                c_pos = None
            else:
                c_pos -= 1
            adj_list.remove(p)
        return c_pos

    def delete(self, p: Point, upper: bool, cross: bool):
        """
        Remove point p from hull graph (upper or lower), updating it accordingly.

        The main idea of the point deletion is that we have to
        'pull' the point to y = -inf for upper hull and y = inf for lower hull
        for each edge p has, it may wrap some points which already have common edge
        and eventually will connect two vertices with a new edge.

        It is important that we go from the 'lowest' to the 'highest' edge,
        as a sequence of wrappings affects the resulting edges.

        One can notice that each vertex in hull graph can have a maximum of log(n)
        left and right edges, so deletion of a vertex has O(log(n)) time complexity.
        """
        vertices_merged = self.adjacency_list[upper][p].left + self.adjacency_list[upper][p].right

        # count sort for linear complexity (log(n) items in each list)
        vertices_ranked: list[list[RankedPoint]] = [list() for _ in range(self.tree_height)]
        for v in vertices_merged:
            vertices_ranked[v.rank].append(v)
        vertices_sorted: list[RankedPoint] = [v for v_rank in vertices_ranked for v in v_rank]

        self.adjacency_list[upper].pop(p)

        a = b = None
        # deleting tangents from leaf to root
        for i, c_ranked in enumerate(reversed(vertices_sorted)):
            c = Point(c_ranked.x, c_ranked.y)
            pq = p < c
            c_pos = self.try_remove_top(p, c, pq=pq, upper=upper)
            if cross:
                err = 'p is not on top for non-top tangent in cross delete'
                assert i + 1 == len(vertices_sorted) or c_pos == -1, err
            else:
                assert c_pos == -1, 'p is not on top in direct delete'

            self.pull(p, a, b, c_ranked, pq=pq, upper=upper, c_pos=c_pos)
            if pq:
                b = c_ranked
            else:
                a = c_ranked

    def check_is_line(self, points: list[Point]) -> bool:
        """
        Checks if a set of 3 or more points form a straight line.
        """
        is_line = True
        for i in range(1, len(points) - 1):
            if not collinear(points[0], points[i], points[-1]):
                is_line = False
                break
        return len(points) > 2 and is_line

    def peel_layer(self) -> list[Point]:
        """
        Extract convex layer from the current points' state,
        updating upper and lower hull graphs accordingly.
        """
        hull_halves = [[], []]
        for i, adj_list in enumerate(self.adjacency_list):
            if len(adj_list) == 0:
                continue
            cur_pop_point_lower = next(iter(adj_list))
            while True:
                hull_halves[i].append(cur_pop_point_lower)
                if len(adj_list[cur_pop_point_lower].right) > 0:
                    next_point = adj_list[cur_pop_point_lower].right[-1]
                    cur_pop_point_lower = Point(next_point.x, next_point.y)
                else:
                    break
        lower_hull, upper_hull = hull_halves

        empty = len(lower_hull) == 0 or len(upper_hull) == 0
        if empty:
            return []

        assert lower_hull[0] == upper_hull[0] and lower_hull[-1] == upper_hull[-1]

        for p in upper_hull:
            self.delete(p, upper=True, cross=False)

        for p in lower_hull:
            self.delete(p, upper=False, cross=False)

        # check if the convex hull is a single line
        # in this case, we have already deleted all vertices
        # from both hull graphs and we have nothing to do
        upper_is_line = self.check_is_line(upper_hull)
        lower_is_line = self.check_is_line(lower_hull)
        if not upper_is_line or not lower_is_line:
            for p in upper_hull[1:-1]:
                self.delete(p, upper=False, cross=True)

            for p in lower_hull[1:-1]:
                self.delete(p, upper=True, cross=True)

        if len(lower_hull) == 1 or len(upper_hull) == 1:
            assert lower_hull == upper_hull
            convex_hull = lower_hull
        else:
            convex_hull = lower_hull[1:] + upper_hull[:-1]
        return list(set(convex_hull))

    def preprocess_points(self, points: list[Point]) -> list[Point]:
        """
        Sort unique points, scale to [-1, 1] by x and y.
        """
        points = sorted(set(points))

        xs = [pt.x for pt in points]
        ys = [pt.y for pt in points]
        self.x_min = min(xs)
        self.y_min = min(ys)
        self.x_scale = max(xs) - self.x_min
        self.y_scale = max(ys) - self.y_min

        return [
            Point(
                (pt.x - self.x_min) / self.x_scale,
                (pt.y - self.y_min) / self.y_scale,
            ) for pt in points
        ]

    def compute_layers(self, points: list[Point]) -> list[list[Point]]:
        """
        Compute all convex layers of a given multiset of points.
        Time complexity: O(n*log(n)).
        """
        if len(points) < 3:
            return [points]

        points_preprocessed_lower = self.preprocess_points(points)
        points_preprocessed_upper = self.preprocess_points(points)

        self.adjacency_list = (
            {p: HullTreeNode() for p in points_preprocessed_lower},
            {p: HullTreeNode() for p in points_preprocessed_upper},
        )
        self.build_graph(points_preprocessed_lower, upper=False)
        self.build_graph(points_preprocessed_upper, upper=True)

        layers = []
        while True:
            convex_hull = self.peel_layer()
            if len(convex_hull) == 0:
                break
            layers.append(convex_hull)
        return [
            [
                Point(
                    (pt.x * self.x_scale) + self.x_min,
                    (pt.y * self.y_scale) + self.y_min,
                ) for pt in layer
            ] for layer in layers
        ]
