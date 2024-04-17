import os.path

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.shape

from src.data_preparation.ifc_extractor.ifc_element_geometry_extractor import IfcElementGeometryExtractor


class SingleIfcModelProcessor:
	
	def __init__(self, ifc_model_path: str, extracted_elements_dir_path: str):
		self._ifc_model_path = ifc_model_path
		self._extracted_elements_dir_path = extracted_elements_dir_path
		self._ifc_model = None
		self._settings = None
		self._source_model_name = None
		
	def setup_ifc_file(self):
		self._ifc_model = ifcopenshell.open(self._ifc_model_path)
		self._settings = ifcopenshell.geom.settings()
		self._source_model_name = self._get_source_model_name()
		
	def _get_source_model_name(self):
		return os.path.basename(self._ifc_model_path)
	
	def process_single_ifc_model(self):
		for ifc_element in self._ifc_model:
			global_id = SingleIfcModelProcessor._get_element_global_id(ifc_element)
			if global_id:
				ifc_element_geometry_extractor = IfcElementGeometryExtractor(ifc_element=ifc_element,
																			 output_directory=self._extracted_elements_dir_path,
																			 source_model_name=self._source_model_name,
																			 settings=self._settings)
				ifc_element_geometry_extractor.extract_geometry()
		return True
	
	@staticmethod
	def _get_element_global_id(ifc_element):
		try:
			return ifc_element.GlobalId
		except AttributeError:
			return None
