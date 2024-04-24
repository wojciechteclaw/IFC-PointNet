import os
import os.path as osp
from typing import List

from src.dataset.settings.data_split import DataSplit
from src.dataset.normalization.enums.normalization_strategy import NormalizationStrategy


class DatasetSettings:
	
	def __init__(self, **kwargs):
		
		self._datasplit = DataSplit(**kwargs)
		self._raw_data_path = kwargs.get('raw_data_path', "../data/raw/")
		self._minimum_number_of_items_for_class = kwargs.get('minimum_number_of_items_for_class', 0)
		self._number_of_point_per_mesh = kwargs.get('number_of_point_per_mesh', 2048)
		self._output_path = kwargs.get('output_path', "../data/")
		self._normalization:NormalizationStrategy = kwargs.get('normalization', NormalizationStrategy.NO_NORMALIZATION)
		
		ifc_classes = kwargs.get('ifc_classes', [])
		dataset_name = kwargs.get('dataset_name', "")

		self._ifc_classes = self.__get_valid_ifc_categories(ifc_classes)
		self._dataset_name = self.__get_dataset_name(dataset_name)
		self._dataset_path = osp.join(self.output_path, self.dataset_name)
	
	def __get_valid_ifc_categories(self, ifc_categories: List[str]=[]):
		valid_ifc_categories = []
		if len(ifc_categories) == 0:
			for ifc_class_directory in os.listdir(self.raw_data_path):
				directory_path = osp.join(self.raw_data_path, ifc_class_directory)
				if os.path.isdir(directory_path) \
				   and len(os.listdir(directory_path)) > self.minimum_number_of_items_for_class:
					valid_ifc_categories.append(ifc_class_directory.lower())
		else:
			for ifc_class in ifc_categories:
				directory_path = osp.join(self.raw_data_path, ifc_class)
				if os.path.isdir(osp.join(self.raw_data_path, ifc_class)) \
				   and len(os.listdir(directory_path)) > self.minimum_number_of_items_for_class:
					valid_ifc_categories.append(ifc_class.lower())
		return valid_ifc_categories

	def __get_dataset_name(self, initial_name):
		if initial_name:
			return initial_name
		else:
			if not len(self.ifc_classes) > 0:
				raise ValueError("No valid IFC classes found")
			classes = len(self.ifc_classes)
			return f"{self.normalization.value}_{classes}_classes_with_{self.number_of_point_per_mesh}_points_per_mesh-min_{self.minimum_number_of_items_for_class}_items_per_class"
	
	@property
	def dataset_name(self):
		return self._dataset_name
	
	@property
	def dataset_path(self):
		return self._dataset_path
	
	@property
	def datasplit(self):
		return self._datasplit
	
	@property
	def ifc_classes(self):
		return self._ifc_classes

	@property
	def minimum_number_of_items_for_class(self):
		return self._minimum_number_of_items_for_class
	
	@property
	def number_of_point_per_mesh(self):
		return self._number_of_point_per_mesh
	
	@property
	def output_path(self):
		return self._output_path
	
	@property
	def raw_data_path(self):
		return self._raw_data_path
	
	@property
	def testing_data_ratio(self):
		return self._datasplit.testing_ratio
	
	@property
	def validating_data_ratio(self):
		return self._datasplit.validating_ratio
	
	@property
	def training_data_ratio(self):
		return self._datasplit.training_ratio