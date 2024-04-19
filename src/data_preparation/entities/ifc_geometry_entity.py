import numpy as np


class IfcGeometryEntity:
	
	def __init__(self,
				 faces:np.array = np.array([0]),
				 guid:str="",
				 location:np.array = np.array([0]),
				 rotation:np.array = np.array([0]),
				 valid:bool=False,
				 vertices:np.array = np.array([0])):
		self._valid = valid
		self._faces = faces
		self._guid = guid
		self._location = location
		self._rotation = rotation
		self._vertices = vertices
		
	@property
	def faces(self):
		return self._faces
	
	@property
	def guid(self):
		return self._guid
	
	@property
	def location(self):
		return self._location
	
	@property
	def rotation(self):
		return self._rotation
	
	@property
	def vertices(self):
		return self._vertices
	
	@property
	def valid(self):
		return self._valid
