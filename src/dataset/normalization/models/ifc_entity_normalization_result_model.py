import trimesh
from src.dataset.normalization.enums.normalization_strategy import NormalizationStrategy


class IfcEntityNormalizationResultModel:
	"""
	A model to store the results of the normalization process of an IFC entity's mesh.
	Attributes:
		mesh (trimesh.Trimesh): The normalized mesh object.
		normalization_factor (float): The factor used to scale the mesh during normalization.
		normalization_strategy (NormalizationStrategy): The strategy applied to normalize the mesh.
	"""
	
	def __init__(self, mesh: trimesh.Trimesh, normalization_factor: float,
				 normalization_strategy: NormalizationStrategy):
		"""
		Initializes the IfcEntityNormalizationResultModel with the normalized mesh and related data.
		Args:
			mesh (trimesh.Trimesh): The normalized mesh.
			normalization_factor (float): The scale factor used during the normalization process.
			normalization_strategy (NormalizationStrategy): The normalization strategy applied.
		"""
		self._mesh = mesh
		self._normalization_factor = normalization_factor
		self._normalization_strategy = normalization_strategy
	
	@property
	def mesh(self):
		"""
		Gets the normalized mesh object.

		Returns:
			trimesh.Trimesh: The normalized mesh.
		"""
		return self._mesh
	
	@property
	def normalization_factor(self):
		"""
		Gets the factor used to normalize the mesh.

		Returns:
			float: The normalization factor.
		"""
		return self._normalization_factor
