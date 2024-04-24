import trimesh

from src.dataset.extractors.extractor import Extractor


class ObjEntityExtractor(Extractor):
	
	def __init__(self, file_path: str):
		super().__init__(file_path)
		
	def extract(self, number_of_points_per_mesh_entity: int):
		mesh = trimesh.load(self.file_path)
		point_cloud = mesh.sample(number_of_points_per_mesh_entity)
		return point_cloud, self.ifc_class
