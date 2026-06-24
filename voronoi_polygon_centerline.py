# SPDX-License-Identifier: MIT
#
# Voronoi-based polygon centerline from fitodic/centerline (same algorithm as the PyPI
# ``centerline`` package's geometry module). Reference:
# https://github.com/fitodic/centerline/blob/master/src/centerline/geometry.py
#
# Vendored to avoid requiring Fiona/GDAL (listed as dependencies of the PyPI package).

from __future__ import annotations

from numpy import array
from scipy.spatial import Voronoi
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree


class TooFewRidgesError(RuntimeError):
    """Too few Voronoi ridges lie strictly inside the polygon; adjust interpolation_distance."""

    pass


class InvalidInputTypeError(RuntimeError):
    pass


class FitodicVoronoiCenterline:
    """Voronoi ridges of a densified polygon boundary, clipped to the polygon interior."""

    def __init__(
        self,
        input_geometry: Polygon | MultiPolygon,
        interpolation_distance: float = 0.5,
        **attributes,
    ):
        self._input_geometry = input_geometry
        self._interpolation_distance = abs(interpolation_distance)

        if not isinstance(self._input_geometry, (Polygon, MultiPolygon)):
            raise InvalidInputTypeError(
                "Input geometry must be Polygon or MultiPolygon."
            )

        self._min_x, self._min_y = self._get_reduced_coordinates()
        for key, value in attributes.items():
            setattr(self, key, value)

        self.geometry = self._construct_centerline_union()

    def _get_reduced_coordinates(self):
        min_x = int(min(self._input_geometry.envelope.exterior.xy[0]))
        min_y = int(min(self._input_geometry.envelope.exterior.xy[1]))
        return min_x, min_y

    def _construct_centerline_union(self):
        vertices, ridges = self._get_voronoi_vertices_and_ridges()
        linestrings = []
        for ridge in ridges:
            if self._ridge_is_finite(ridge):
                starting_point = self._create_point_with_restored_coordinates(
                    x=vertices[ridge[0]][0], y=vertices[ridge[0]][1]
                )
                ending_point = self._create_point_with_restored_coordinates(
                    x=vertices[ridge[1]][0], y=vertices[ridge[1]][1]
                )
                linestrings.append(LineString((starting_point, ending_point)))

        str_tree = STRtree(linestrings)
        linestrings_indexes = str_tree.query(
            self._input_geometry, predicate="contains"
        )
        contained_linestrings = [linestrings[i] for i in linestrings_indexes]
        if len(contained_linestrings) < 2:
            raise TooFewRidgesError(
                "Too few ridges inside polygon; adjust interpolation_distance."
            )

        return unary_union(contained_linestrings)

    def _get_voronoi_vertices_and_ridges(self):
        borders = self._get_densified_borders()
        voronoi_diagram = Voronoi(borders)
        vertices = voronoi_diagram.vertices
        ridges = voronoi_diagram.ridge_vertices
        return vertices, ridges

    @staticmethod
    def _ridge_is_finite(ridge):
        return -1 not in ridge

    def _create_point_with_restored_coordinates(self, x, y):
        return (x + self._min_x, y + self._min_y)

    def _get_densified_borders(self):
        polygons = self._extract_polygons_from_input_geometry()
        points = []
        for polygon in polygons:
            points += self._get_interpolated_boundary(polygon.exterior)
            if self._polygon_has_interior_rings(polygon):
                for interior in polygon.interiors:
                    points += self._get_interpolated_boundary(interior)

        return array(points)

    def _extract_polygons_from_input_geometry(self):
        if isinstance(self._input_geometry, MultiPolygon):
            return (polygon for polygon in self._input_geometry.geoms)
        return (self._input_geometry,)

    @staticmethod
    def _polygon_has_interior_rings(polygon):
        return len(polygon.interiors) > 0

    def _get_interpolated_boundary(self, boundary):
        line = LineString(boundary)
        first_point = self._get_coordinates_of_first_point(line)
        last_point = self._get_coordinates_of_last_point(line)
        intermediate_points = self._get_coordinates_of_interpolated_points(line)
        return [first_point] + intermediate_points + [last_point]

    def _get_coordinates_of_first_point(self, linestring):
        return self._create_point_with_reduced_coordinates(
            x=linestring.xy[0][0], y=linestring.xy[1][0]
        )

    def _get_coordinates_of_last_point(self, linestring):
        return self._create_point_with_reduced_coordinates(
            x=linestring.xy[0][-1], y=linestring.xy[1][-1]
        )

    def _get_coordinates_of_interpolated_points(self, linestring):
        intermediate_points = []
        interpolation_distance = self._interpolation_distance
        line_length = linestring.length
        step = interpolation_distance
        while step < line_length:
            point = linestring.interpolate(step)
            intermediate_points.append(
                self._create_point_with_reduced_coordinates(x=point.x, y=point.y)
            )
            step += interpolation_distance

        return intermediate_points

    def _create_point_with_reduced_coordinates(self, x, y):
        return (x - self._min_x, y - self._min_y)
