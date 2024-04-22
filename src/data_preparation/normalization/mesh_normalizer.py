import trimesh

from src.data_preparation.normalization.models.ifc_entity_normalization_result_model import IfcEntityNormalizationResultModel
from src.data_preparation.normalization.enums.normalization_strategy import NormalizationStrategy


class MeshNormalizer:

    @staticmethod
    def normalize(mesh:trimesh.Trimesh,
                  strategy:NormalizationStrategy=NormalizationStrategy.NO_NORMALIZATION
                  ) -> IfcEntityNormalizationResultModel:
        if strategy == NormalizationStrategy.ZERO_TO_ONE:
            new_vertices, normalization_factor = MeshNormalizer.normalize_zero_to_one(mesh)
            mesh.vertices = new_vertices
        elif strategy == NormalizationStrategy.MINUS_ONE_TO_ONE:
            new_vertices, normalization_factor =  MeshNormalizer.normalize_minus_one_to_one(mesh)
            mesh.vertices = new_vertices
        else:
            normalization_factor = 1
        return IfcEntityNormalizationResultModel(mesh=mesh,
                                                 normalization_factor=normalization_factor,
                                                 normalization_strategy=strategy)

    @staticmethod
    def normalize_zero_to_one(mesh:trimesh.Trimesh) -> tuple:
        vertices_coordinates = mesh.vertices
        min_vals = vertices_coordinates.min(axis=0)
        max_vals = vertices_coordinates.max(axis=0)
        ranges = max_vals - min_vals
        normalization_factor = ranges.max()
        center_shift = (1 - ranges / normalization_factor) / 2
        normalized_vertices = (vertices_coordinates - min_vals) / normalization_factor
        return normalized_vertices + center_shift, normalization_factor

    @staticmethod
    def normalize_minus_one_to_one(mesh:trimesh.Trimesh) -> tuple:
        normalized_vertices, normalization_factor = MeshNormalizer.normalize_zero_to_one(mesh)
        return 2 * normalized_vertices - 1, normalization_factor / 2
