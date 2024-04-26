import trimesh
from src.dataset.normalization.models.ifc_entity_normalization_result_model import IfcEntityNormalizationResultModel
from src.dataset.normalization.enums.normalization_strategy import NormalizationStrategy


class MeshNormalizer:
	"""
    A class for normalizing 3D mesh objects according to specified normalization strategies.
    """
	
	@staticmethod
	def normalize(mesh: trimesh.Trimesh,
				  strategy: NormalizationStrategy = NormalizationStrategy.NO_NORMALIZATION
				  ) -> IfcEntityNormalizationResultModel:
		"""
        Normalize a mesh object using a specified normalization strategy.

        Args:
            mesh (trimesh.Trimesh): The mesh object to normalize.
            strategy (NormalizationStrategy): The normalization strategy to apply. Defaults to NO_NORMALIZATION.

        Returns:
            IfcEntityNormalizationResultModel: A model containing the normalized mesh, the normalization factor,
                                               and the applied normalization strategy.
        """
		if strategy == NormalizationStrategy.ZERO_TO_ONE:
			new_vertices, normalization_factor = MeshNormalizer.normalize_zero_to_one(mesh)
			mesh.vertices = new_vertices
		elif strategy == NormalizationStrategy.MINUS_ONE_TO_ONE:
			new_vertices, normalization_factor = MeshNormalizer.normalize_minus_one_to_one(mesh)
			mesh.vertices = new_vertices
		else:
			normalization_factor = 1
		
		return IfcEntityNormalizationResultModel(mesh=mesh,
												 normalization_factor=normalization_factor,
												 normalization_strategy=strategy)
	
	@staticmethod
	def normalize_zero_to_one(mesh: trimesh.Trimesh) -> tuple:
		"""
        Normalize the vertex coordinates of a mesh to range between 0 and 1.
        Args:
            mesh (trimesh.Trimesh): The mesh object whose vertices are to be normalized.
        Returns:
            tuple: A tuple containing the normalized vertices and the maximum range used for normalization.
        """
		vertices_coordinates = mesh.vertices
		min_vals = vertices_coordinates.min(axis=0)
		max_vals = vertices_coordinates.max(axis=0)
		ranges = max_vals - min_vals
		normalization_factor = ranges.max()
		center_shift = (1 - ranges / normalization_factor) / 2
		normalized_vertices = (vertices_coordinates - min_vals) / normalization_factor
		return normalized_vertices + center_shift, normalization_factor
	
	@staticmethod
	def normalize_minus_one_to_one(mesh: trimesh.Trimesh) -> tuple:
		"""
        Normalize the vertex coordinates of a mesh to range between -1 and 1.
        Args:
            mesh (trimesh.Trimesh): The mesh object whose vertices are to be normalized.
        Returns:
            tuple: A tuple containing the normalized vertices adjusted to range from -1 to 1,
                   and half the maximum range used in the normalization.
        """
		normalized_vertices, normalization_factor = MeshNormalizer.normalize_zero_to_one(mesh)
		return 2 * normalized_vertices - 1, normalization_factor / 2
