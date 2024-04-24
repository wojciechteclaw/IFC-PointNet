import trimesh

from src.dataset.extractors.extractor import Extractor
from src.dataset.normalization.enums.normalization_strategy import NormalizationStrategy
from src.dataset.normalization.mesh_normalizer import MeshNormalizer


class PlyEntityExtractor(Extractor):
	
	def __init__(self,
				 file_path: str,
				 normalization_strategy=NormalizationStrategy.NO_NORMALIZATION):
		super().__init__(file_path, normalization_strategy)

	def extract(self, number_of_points_per_mesh_entity: int):
		mesh = trimesh.load(self.file_path)
		normalization_result = MeshNormalizer.normalize(mesh, self.normalization_strategy)
		# here we can extract the normalization factor
		point_cloud = normalization_result.mesh.sample(number_of_points_per_mesh_entity)
		return point_cloud, self.ifc_class
