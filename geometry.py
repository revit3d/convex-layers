import math
import numpy as np

from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    eps = 1e-12

    def __eq__(self, other):
        # return self.x == other.x and self.y == other.y
        x_eq = y_eq = False
        if math.isfinite(self.x) and math.isfinite(other.x):
            x_eq = abs(self.x - other.x) < self.eps
        else:
            x_eq = self.x == other.x
        if math.isfinite(self.y) and math.isfinite(other.y):
            y_eq = abs(self.y - other.y) < self.eps
        else:
            y_eq = self.y == other.y
        return x_eq and y_eq

    def __lt__(self, other):
        return (
            self != other
            and (
                self.x < other.x
                or abs(self.x - other.x) < self.eps and self.y < other.y
            )
        )


@dataclass(frozen=True)
class RankedPoint(Point):
    rank: int


def cross(o: Point, a: Point, b: Point) -> float:
    """
    Cross product of segments oa and ob.
    """
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)


def slope(a: Point, b: Point, tol: float = 1e-12) -> float:
    """
    Calculates slope of a line defined by points a and b.
    Note that in this implementation, slope can be equal to -inf
    in case a.x == b.x and a.y > b.y.
    """
    if abs(a.x - b.x) < tol:
        return float('inf') * (b.y - a.y)
    return (b.y - a.y) / (b.x - a.x)


def collinear(p: Point, p0: Point | None, p1: Point | None, tol: float = 1e-10) -> bool:
    """
    Collinearity check for segments [p, p0] and [p, p1].
    """
    if p0 is None or p1 is None:
        return False
    return abs(cross(p, p0, p1)) < tol


def convex_hull_andrew(points: list[Point], only_upper=False, only_lower=False) -> list[Point]:
        """
        Andrew's monotone chain algorithm for convex hull.
        Assumes input is sorted by (x, y). Time complexity: O(n).
        """
        if len(points) <= 2:
            return points

        if not only_upper:
            lower = []  # lower hull
            for p in points:
                while len(lower) >= 2 and cross(lower[-2], lower[-1], p) < 0:
                    lower.pop()
                lower.append(p)

        if not only_lower:
            upper = []  # upper hull
            for p in reversed(points):
                while len(upper) >= 2 and cross(upper[-2], upper[-1], p) < 0:
                    upper.pop()
                upper.append(p)

        if only_upper:
            return upper[::-1]
        elif only_lower:
            return lower

        # remove duplicate points
        return list(set(lower[:-1] + upper[:-1]))


def sort_hull_points(points):
    """
    Sort hull points by polar angle.
    """
    if len(points) <= 2:
        return points

    cx = sum(p.x for p in points) / len(points)
    cy = sum(p.y for p in points) / len(points)

    def polar_angle(p: Point):
        return np.arctan2(p.y - cy, p.x - cx)

    return sorted(points, key=polar_angle)
