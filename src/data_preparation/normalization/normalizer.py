import numpy as np
import trimesh
from src.data_preparation.enums.normalization_strategy import NormalizationStrategy


class Normalizer:

    @staticmethod
    def normalize(mesh:trimesh.Trimesh, strategy:NormalizationStrategy=NormalizationStrategy.NO_NORMALIZATION):
        if strategy == NormalizationStrategy.ZERO_TO_ONE:
            new_vertices = Normalizer.normalize_zero_to_one(mesh)
            mesh.vertices = new_vertices
        elif strategy == NormalizationStrategy.MINUS_ONE_TO_ONE:
            new_vertices =  Normalizer.normalize_minus_one_to_one(mesh)
            mesh.vertices = new_vertices
        return mesh

    @staticmethod
    def normalize_zero_to_one(mesh:trimesh.Trimesh):
        vertices_coordinates = mesh.vertices
        min_vals = vertices_coordinates.min(axis=0)
        max_vals = vertices_coordinates.max(axis=0)
        ranges = max_vals - min_vals
        normalization_dimension = ranges.max()
        center_shift = (1 - ranges / normalization_dimension) / 2
        normalized = (vertices_coordinates - min_vals) / normalization_dimension
        return normalized + center_shift

    @staticmethod
    def normalize_minus_one_to_one(mesh:trimesh.Trimesh):
        normalized = Normalizer.normalize_zero_to_one(mesh) * 2
        return normalized - 1
