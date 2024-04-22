from typing import List


class DatasetGenerator:
	def __init__(self,
				 data_dir_path: str,
				 output_path: str = None,
				 dataset_name: str = "",
				 number_of_point_per_mesh: int = 2048,
				 ifc_classes:List[str] = []):
		self._data_dir_path = data_dir_path
		self._number_of_point_per_mesh = number_of_point_per_mesh
		self._ifc_classes = ifc_classes
		self._output_path = output_path
		self._dataset_name = self.__get_dataset_name(dataset_name)
		
	def __get_dataset_name(self, initial_name):
		if initial_name:
			return initial_name
		else:
			if len(self.ifc_classes) > 0:
				classes = "_".join(self.ifc_classes)
			else:
				classes = "all"
			return f"{classes}_{self.number_of_point_per_mesh}"

	def process_ifc_entities(self):
		pass
	
	def create_datasets(self):
		pass
	
	
	
	
	@property
	def data_dir_path(self):
		return self._data_dir_path
	
	@property
	def output_path(self):
		return self._output_path
	
	@property
	def ifc_classes(self):
		return self._ifc_classes
		