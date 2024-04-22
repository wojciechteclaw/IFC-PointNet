import trimesh

from src.dataset.normalization.enums.normalization_strategy import NormalizationStrategy


class IfcEntityNormalizationResultModel:
	
	def __init__(self,
				 mesh:trimesh.Trimesh,
				 normalization_factor:float,
				 normalization_strategy:NormalizationStrategy):
		self._mesh = mesh
		self._normalization_factor = normalization_factor
		self._normalization_strategy = normalization_strategy
		
	@property
	def mesh(self):
		return self._mesh
	
	@property
	def normalization_factor(self):
		return self._normalization_factor
