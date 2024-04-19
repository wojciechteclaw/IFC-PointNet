import ifcopenshell as ifcopenshell
import numpy as np
from src.data_preparation.entities.ifc_geometry_entity import IfcGeometryEntity
from src.data_preparation.entities.ifc_entity import IfcEntity


class IfcElementGeometryExtractor:
	
	def __init__(self,
				 ifc_element: ifcopenshell.entity_instance,
				 output_directory: str,
				 source_model_name: str,
				 settings:ifcopenshell.geom.settings):
		self._ifc_element = ifc_element
		self._output_directory = output_directory
		self._source_model_name = source_model_name
		self._settings = settings
		
	def get_geometry(self):
		try:
			shape = ifcopenshell.geom.create_shape(self._settings, self._ifc_element)
			if shape:
				vertices = np.array(shape.geometry.verts).reshape(-1, 3)
				faces = np.array(shape.geometry.faces).reshape(-1, 3)
				placement = ifcopenshell.util.shape.get_shape_matrix(shape)
				location = placement[:,3][0:3]
				rotation = placement[:3, :3]
				guid = shape.guid
				
				return IfcGeometryEntity(faces=faces,
										 guid=guid,
										 location=location,
										 rotation=rotation,
										 valid=True,
										 vertices=vertices)
		except:
			pass
		return IfcGeometryEntity()
		
			
		
	def extract_geometry(self):
		geometry_entity = self.get_geometry()
		if geometry_entity.valid:
			
			object_class = self._ifc_element.is_a()
			object_type = self._ifc_element.ObjectType
			object_name = self._ifc_element.Name
			
			
			ifc_entity = IfcEntity(ifc_geometry_entity=geometry_entity,
								   object_class=object_class,
								   object_name=object_name,
								   object_type=object_type,
								   source_file_name=self._source_model_name)
			ifc_entity.dump(output_directory=self._output_directory)
			
			