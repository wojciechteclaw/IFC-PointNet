import logging
import os
import uuid
import pickle
import trimesh
import os.path as osp

from src.data_preparation.entities.ifc_geometry_entity import IfcGeometryEntity


class IfcEntity:
	
	def __init__(self,
				 ifc_geometry_entity: IfcGeometryEntity,
				 object_class:str,
				 object_name:str,
				 object_type:str,
				 source_file_name:str):
		
		self._faces = ifc_geometry_entity.faces
		self._global_id = ifc_geometry_entity.guid
		self._location = ifc_geometry_entity.location
		self._object_class = object_class
		self._object_name = object_name
		self._object_type = object_type
		self._rotation = ifc_geometry_entity.rotation
		self._source_file_name = source_file_name
		self._vertices = ifc_geometry_entity.vertices
		
		self.trimesh_instance = None

	def dump(self, output_directory:str):
		try:
			file_name = self.get_file_name()
			file_path = osp.join(output_directory, self._object_class, file_name)
			if not osp.exists(osp.join(output_directory, self._object_class)):
				os.makedirs(osp.join(output_directory, self._object_class))
			with open(file_path, 'wb') as file:
				pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
		except:
			logging.error(f"Error while dumping the file from {self.source_file_name} with guid: {self.global_id}")
	
	def get_file_name(self):
		return f"{self._object_class}_{uuid.uuid4()}.pkl".lower()
	
	def get_trimesh(self):
		return trimesh.Trimesh(vertices=self.vertices, faces=self.faces)
	
	@staticmethod
	def load(file_path:str):
		try:
			with open(file_path, 'rb') as file:
				return pickle.load(file)
		except FileNotFoundError:
			logging.error(f"File not found: {file_path}")
		finally:
			file.close()
		return None
	
	@property
	def faces(self):
		return self._faces
	
	@property
	def global_id(self):
		return self._global_id
	
	@property
	def location(self):
		return self._location
	
	@property
	def object_class(self):
		return self._object_class
	
	@property
	def object_name(self):
		return self._object_name
	
	@property
	def object_type(self):
		return self._object_type
	
	@property
	def rotation(self):
		return self._rotation
	
	@property
	def source_file_name(self):
		return self._source_file_name
	
	@property
	def vertices(self):
		return self._vertices
