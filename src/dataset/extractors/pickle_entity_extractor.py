from src.data_preparation.entities.ifc_entity import IfcEntity
from src.dataset.extractors.extractor import Extractor


class PickleEntityExtractor(Extractor):
	
	def __init__(self, file_path: str):
		super().__init__(file_path)
	
	def extract(self, number_of_points_per_mesh_entity: int):
		entity = IfcEntity.load(self.file_path)
		trimesh = entity.get_trimesh()
		point_cloud = trimesh.sample(number_of_points_per_mesh_entity)
		return point_cloud, self.ifc_class
