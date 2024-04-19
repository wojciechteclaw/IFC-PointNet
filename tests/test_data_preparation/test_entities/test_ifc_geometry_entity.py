import unittest

import numpy as np
import pytest
from src.data_preparation.entities.ifc_geometry_entity import IfcGeometryEntity


class IfcGeometryEntityBase:
	
	def setUpEntity(self,
					expected_faces:np.array,
					expected_guid:str,
					expected_location:np.array,
					expected_rotation:np.array,
					expected_valid:bool,
					expected_vertices:np.array):
		self.expected_faces = expected_faces
		self.expected_guid = expected_guid
		self.expected_location = expected_location
		self.expected_rotation = expected_rotation
		self.expected_valid = expected_valid
		self.expected_vertices = expected_vertices
		self.ifc_geometry_entity = IfcGeometryEntity(faces=self.expected_faces,
													 guid=self.expected_guid,
													 location=self.expected_location,
													 rotation=self.expected_rotation,
													 valid=self.expected_valid,
													 vertices=self.expected_vertices)
		
	def setUpDefaultEntity(self):
		self.expected_faces = np.array([0])
		self.expected_guid = ""
		self.expected_location = np.array([0])
		self.expected_rotation = np.array([0])
		self.expected_valid = False
		self.expected_vertices = np.array([0])
		self.ifc_geometry_entity = IfcGeometryEntity()
		
	def test_faces(self):
		assert isinstance(self.ifc_geometry_entity.faces, np.ndarray)
		assert (self.ifc_geometry_entity.faces == self.expected_faces).all()

	def test_guid(self):
		assert isinstance(self.ifc_geometry_entity.guid, str)
		assert self.ifc_geometry_entity.guid == self.expected_guid
		
	def test_location(self):
		assert isinstance(self.ifc_geometry_entity.location, np.ndarray)
		assert (self.ifc_geometry_entity.location == self.expected_location).all()
		
	def test_rotation(self):
		assert isinstance(self.ifc_geometry_entity.rotation, np.ndarray)
		assert (self.ifc_geometry_entity.rotation == self.expected_rotation).all()
	
	def test_vertices(self):
		assert isinstance(self.ifc_geometry_entity.vertices, np.ndarray)
		assert (self.ifc_geometry_entity.vertices == self.expected_vertices).all()
		
	def test_valid(self):
		assert isinstance(self.ifc_geometry_entity.valid, bool)
		assert self.ifc_geometry_entity.valid == self.expected_valid
		
class TestIfcGeometryEntity1(unittest.TestCase, IfcGeometryEntityBase):
	
	def setUp(self):
		expected_faces = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
		expected_guid = "guid"
		expected_location = np.array([1, 2, 3])
		expected_rotation = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
		expected_valid = True
		expected_vertices = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
		self.setUpEntity(expected_faces=expected_faces,
						 expected_guid=expected_guid,
						 expected_location=expected_location,
						 expected_rotation=expected_rotation,
						 expected_valid=expected_valid,
						 expected_vertices=expected_vertices)
		
class TestIfcGeometryEntity2(unittest.TestCase, IfcGeometryEntityBase):
	
	def setUp(self):
		self.setUpDefaultEntity()
