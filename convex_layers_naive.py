from geometry import Point, convex_hull_andrew


class NaiveHullBuilder:
    def compute_layers(self, points: list[Point]) -> list[list[Point]]:
        preprocessed_points = set(points)

        layers = []
        while preprocessed_points:
            hull = convex_hull_andrew(sorted(preprocessed_points))
            layers.append(hull)

            for point in hull:
                preprocessed_points.discard(point)

        return layers
