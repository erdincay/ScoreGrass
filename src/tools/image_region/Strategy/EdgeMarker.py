from src.tools.image_region.Model.Line import Line
from src.tools.image_region.Strategy.RegionInitializer import generate_square_linked_coordinates

__author__ = 'Kern'


def generate_edge_coordinates(coord_map):
    coords_set = set()
    for coord in coord_map:
        for linked_coord in generate_square_linked_coordinates(coord):
            if linked_coord not in coord_map:
                coords_set.add(coord)
                break
    return coords_set


def make_region_edge(region):
    coords_set = generate_edge_coordinates(region.coord_map)
    return Line(coords_set)
