import numpy as np
import trimesh

from src.dataset.augmentation.mesh_rotation_augmentation import MeshRotationAugmentation
from src.dataset.extractors.extractor import Extractor
from src.dataset.normalization.enums.normalization_strategy import NormalizationStrategy
from src.dataset.normalization.mesh_normalizer import MeshNormalizer


class ObjEntityExtractor(Extractor):
	"""
	Extracts data from OBJ files and normalizes them using a specified normalization strategy.
	"""
	
	def __init__(self, file_path: str, normalization_strategy=NormalizationStrategy.NO_NORMALIZATION):
		"""
		Initializes the ObjEntityExtractor with a file path and a normalization strategy.
		Args:
			file_path (str): Path to the OBJ file for data extraction.
			normalization_strategy (NormalizationStrategy): Normalization strategy to be used for data extraction.
		"""
		super().__init__(file_path, normalization_strategy)

	def extract(self, number_of_points_per_mesh_entity: int, augment: bool = False, add_noise: bool = False):
		"""
		Extracts a point cloud from an OBJ file, normalized according to the specified strategy.
		Args:
			number_of_points_per_mesh_entity (int): The number of points per mesh entity to extract.
		Returns:
			tuple: A tuple containing the point cloud data and the IFC class of the mesh.
		"""
		mesh = trimesh.load(self.file_path)
		if augment:
			mesh = MeshRotationAugmentation.augment_single_element(mesh)
		normalization_result = MeshNormalizer.normalize(mesh, self.normalization_strategy)
		point_cloud = normalization_result.mesh.sample(number_of_points_per_mesh_entity)
		if add_noise:
			noise = np.random.normal(0, 0.010, (point_cloud.shape))
			point_cloud += noise
		return point_cloud, self.ifc_class
