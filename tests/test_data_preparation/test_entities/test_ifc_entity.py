import os
import os.path as osp
import unittest
from unittest.mock import patch

import numpy as np
import trimesh

from src.data_preparation.entities.ifc_entity import IfcEntity
from src.data_preparation.entities.ifc_geometry_entity import IfcGeometryEntity

current_dir = osp.dirname(osp.realpath(__file__))
sample_element_name = "sample_entity.pkl"
sample_obj_file_name = "norm_test_duct.obj"
assets_path = osp.join(current_dir, "../../assets")
pkl_object_path = osp.join(assets_path, sample_element_name)
sample_obj_file_path = osp.join(assets_path, sample_obj_file_name)

class IfcEntityBase:
	
	def setUpEntity(self):
		self.entity = IfcEntity.load(pkl_object_path)
	
	def test_faces(self):
		assert self.entity.faces is not None
		assert isinstance(self.entity.faces, np.ndarray)
	
	def test_faces_shape(self):
		shape = self.entity.faces.shape
		assert shape is not None
		assert isinstance(shape, tuple)
		assert len(shape) == 2
		assert shape[1] == 3
		
	def test_global_id(self):
		assert self.entity.global_id is not None
		assert isinstance(self.entity.global_id, str)
		
	def test_location(self):
		assert self.entity.location is not None
		assert isinstance(self.entity.location, np.ndarray)
	
	def test_location_shape(self):
		shape = self.entity.location.shape
		assert shape is not None
		assert isinstance(shape, tuple)
		assert len(shape) == 1
		assert shape[0] == 3
		
	def test_object_class(self):
		assert self.entity.object_class is not None
		assert isinstance(self.entity.object_class, str)
		
	def test_object_name(self):
		assert self.entity.object_name is not None
		assert isinstance(self.entity.object_name, str)
		
	def test_object_type(self):
		assert self.entity.object_type is not None
		assert isinstance(self.entity.object_type, str)
		
	def test_rotation(self):
		assert self.entity.rotation is not None
		assert isinstance(self.entity.rotation, np.ndarray)
		
	def test_rotation_shape(self):
		shape = self.entity.rotation.shape
		assert shape is not None
		assert isinstance(shape, tuple)
		assert len(shape) == 2
		assert shape[0] == 3
		assert shape[1] == 3
		
	def test_source_file_name(self):
		assert self.entity.source_file_name is not None
		assert isinstance(self.entity.source_file_name, str)
		
	def test_vertices(self):
		assert self.entity.vertices is not None
		assert isinstance(self.entity.vertices, np.ndarray)
		
	def test_vertices_shape(self):
		shape = self.entity.vertices.shape
		assert shape is not None
		assert isinstance(shape, tuple)
		assert len(shape) == 2
		assert shape[1] == 3
		
	def test_trimesh_instance(self):
		mesh = self.entity.get_trimesh()
		assert mesh is not None
		assert isinstance(mesh, trimesh.Trimesh)
		
	def tearDown(self):
		resulting_dir = os.path.join(assets_path, "results")
		if not os.path.exists(resulting_dir):
			os.makedirs(resulting_dir)
class TestIfcEntity1(unittest.TestCase, IfcEntityBase):
	
	def setUp(self):
		self.setUpEntity()


class TestIfcEntity2(unittest.TestCase, IfcEntityBase):
	
	def setUp(self):
		mesh = trimesh.load(sample_obj_file_path)
		self.mesh = mesh
		self.rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
		self.location = np.array([0, 0, 0])
		self.guid = "guid"
		self.ifc_class = "ifc_test_class"
		self.ifc_type = "super fancy type name"
		self.ifc_name = "super fancy name"
		self.source_file_name = "awesome model.ifc"
		self.dump_file_path = osp.join(assets_path, "results")
		
		self.number_of_faces = len(mesh.faces)
		self.number_of_vertices = len(mesh.vertices)
		
		
		ifc_geometry_entity = IfcGeometryEntity(
			faces=mesh.faces,
			guid=self.guid,
			location=self.location,
			rotation=self.rotation,
			valid=True,
			vertices=mesh.vertices
		)
		self.entity = IfcEntity(
			ifc_geometry_entity=ifc_geometry_entity,
			object_class=self.ifc_class,
			object_name=self.ifc_name,
			object_type=self.ifc_type,
			source_file_name=self.source_file_name
		)
	
	@patch('src.data_preparation.entities.ifc_entity.IfcEntity.get_file_name', return_value="ifc_entity_test_file.pkl")
	def test_dump(self, mock_get_file_name):
		self.entity.dump(self.dump_file_path)
		assert osp.exists(self.dump_file_path)
		mock_get_file_name.assert_called_once()
		
	def test_load(self):
		file_path = osp.join(assets_path, "results", self.ifc_class, "ifc_entity_test_file.pkl")
		self.entity = IfcEntity.load(file_path)
		assert self.entity is not None
		assert isinstance(self.entity, IfcEntity)
		
	def test_loaded_faces(self):
		assert self.entity.faces is not None
		assert isinstance(self.entity.faces, np.ndarray)
		assert len(self.entity.faces) == self.number_of_faces
	
	def test_loaded_vertices(self):
		assert self.entity.vertices is not None
		assert isinstance(self.entity.vertices, np.ndarray)
		assert len(self.entity.vertices) == self.number_of_vertices
		
	def test_loaded_location(self):
		assert self.entity.location is not None
		assert isinstance(self.entity.location, np.ndarray)
		assert (self.entity.location == self.location).all()
		
	def test_loaded_rotation(self):
		assert self.entity.rotation is not None
		assert isinstance(self.entity.rotation, np.ndarray)
		assert (self.entity.rotation == self.rotation).all()
		
	def test_loaded_global_id(self):
		assert self.entity.global_id is not None
		assert isinstance(self.entity.global_id, str)
		assert self.entity.global_id == self.guid
		
	def test_loaded_object_class(self):
		assert self.entity.object_class is not None
		assert isinstance(self.entity.object_class, str)
		assert self.entity.object_class == self.ifc_class
		
	def test_loaded_object_name(self):
		assert self.entity.object_name is not None
		assert isinstance(self.entity.object_name, str)
		assert self.entity.object_name == self.ifc_name
		
	def test_loaded_object_type(self):
		assert self.entity.object_type is not None
		assert isinstance(self.entity.object_type, str)
		assert self.entity.object_type == self.ifc_type
		
	
	
		