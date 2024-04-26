import os
import os.path as osp
from typing import List

from src.dataset.settings.data_split import DataSplit
from src.dataset.normalization.enums.normalization_strategy import NormalizationStrategy


class DatasetSettings:
	"""
	Class to manage settings for a dataset configuration, including data paths, data splitting, and normalization.

	Attributes are configured through keyword arguments passed during initialization, with defaults provided for each.
	"""
	
	def __init__(self, **kwargs):
		"""
		Initializes the settings for dataset configuration.

		Args:
			**kwargs: Various keyword arguments for setting up the dataset, such as:
				'raw_data_path' (str): Path to the raw data directory (default: "../data/raw/").
				'minimum_number_of_items_for_class' (int): Minimum number of items per class for inclusion (default: 0).
				'number_of_point_per_mesh' (int): Number of points per mesh (default: 2048).
				'output_path' (str): Base output path for processed datasets (default: "../data/").
				'normalization' (NormalizationStrategy): Normalization strategy to apply (default: NormalizationStrategy.NO_NORMALIZATION).
				'ifc_classes' (List[str]): Specific IFC classes to include, empty list uses all available classes.
				'dataset_name' (str): Optional specific name for the dataset.
		"""
		self._datasplit = DataSplit(**kwargs)
		self._raw_data_path = kwargs.get('raw_data_path', "../data/raw/")
		self._minimum_number_of_items_for_class = kwargs.get('minimum_number_of_items_for_class', 0)
		self._number_of_point_per_mesh = kwargs.get('number_of_point_per_mesh', 2048)
		self._output_path = kwargs.get('output_path', "../data/")
		self._normalization = kwargs.get('normalization', NormalizationStrategy.NO_NORMALIZATION)
		
		ifc_classes = kwargs.get('ifc_classes', [])
		dataset_name = kwargs.get('dataset_name', "")
		
		self._ifc_classes = self.__get_valid_ifc_categories(ifc_classes)
		self._dataset_name = self.__get_dataset_name(dataset_name)
		self._dataset_path = osp.join(self.output_path, self.dataset_name)
	
	def __get_valid_ifc_categories(self, ifc_categories: List[str] = []):
		"""
		Retrieves and validates IFC classes from the specified categories or the directory structure.
		Args:
			ifc_categories (List[str]): Specific IFC classes to validate.
		Returns:
			List[str]: A list of valid IFC categories based on directory presence and item count criteria.
		"""
		valid_ifc_categories = []
		if len(ifc_categories) == 0:
			for ifc_class_directory in os.listdir(self.raw_data_path):
				directory_path = osp.join(self.raw_data_path, ifc_class_directory)
				if os.path.isdir(directory_path) and len(
						os.listdir(directory_path)) > self.minimum_number_of_items_for_class:
					valid_ifc_categories.append(ifc_class_directory.lower())
		else:
			for ifc_class in ifc_categories:
				directory_path = osp.join(self.raw_data_path, ifc_class)
				if os.path.isdir(directory_path) and len(
						os.listdir(directory_path)) > self.minimum_number_of_items_for_class:
					valid_ifc_categories.append(ifc_class.lower())
		return valid_ifc_categories
	
	def __get_dataset_name(self, initial_name):
		"""
		Generates or confirms a dataset name based on input and configuration.
		Args:
			initial_name (str): Initial name suggestion for the dataset.
		Returns:
			str: The confirmed or generated dataset name.
		"""
		if initial_name:
			return initial_name
		else:
			if not len(self.ifc_classes) > 0:
				raise ValueError("No valid IFC classes found")
			classes = len(self.ifc_classes)
			return f"N-{self._normalization.value}_{classes}_classes_with_{self.number_of_point_per_mesh}_points_per_mesh-min_{self.minimum_number_of_items_for_class}_items_per_class"
	
	@property
	def dataset_name(self):
		"""str: Returns the name of the dataset."""
		return self._dataset_name
	
	@property
	def dataset_path(self):
		"""str: Returns the computed path for the dataset based on the output path and dataset name."""
		return self._dataset_path
	
	@property
	def datasplit(self):
		"""DataSplit: Returns the DataSplit object used for this dataset."""
		return self._datasplit
	
	@property
	def ifc_classes(self):
		"""List[str]: Returns the list of valid IFC classes for the dataset."""
		return self._ifc_classes
	
	@property
	def minimum_number_of_items_for_class(self):
		"""int: Returns the minimum number of items per class required for inclusion in the dataset."""
		return self._minimum_number_of_items_for_class
	
	@property
	def number_of_point_per_mesh(self):
		"""int: Returns the number of points each mesh should have in the dataset."""
		return self._number_of_point_per_mesh
	
	@property
	def output_path(self):
		"""str: Returns the base path where the dataset will be output."""
		return self._output_path
	
	@property
	def raw_data_path(self):
		"""str: Returns the path to the directory containing raw data."""
		return self._raw_data_path
	
	@property
	def testing_data_ratio(self):
		"""float: Returns the testing data split ratio from the DataSplit object."""
		return self._datasplit.testing_ratio
	
	@property
	def validating_data_ratio(self):
		"""float: Returns the validating data split ratio from the DataSplit object."""
		return self._datasplit.validation_ratio
	
	@property
	def training_data_ratio(self):
		"""float: Returns the training data split ratio from the DataSplit object."""
		return self._datasplit.training_ratio
