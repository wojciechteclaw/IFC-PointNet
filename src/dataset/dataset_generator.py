import json
import os
import os.path as osp
import shutil
from typing import List

import numpy as np

from src.dataset.ifc_pointnet_dataset import IfcPointNetDataset
from src.dataset.settings.dataset_settings import DatasetSettings


class DatasetGenerator:

	training_dataset_name = "training"
	testing_dataset_name = "testing"
	validating_dataset_name = "validating"
	ifc_classes_map_file_name = "ifc_classes_map.json"
	dataset_extension = ".pt"
	
	def __init__(self, dataset_settings: DatasetSettings):
		self._dataset_settings = dataset_settings
		self._training_dataset = None
		self._testing_dataset = None
		self._validating_dataset = None
		self._ifc_classes_map = {}
	
	def process_data(self):
		if not osp.exists(self.dataset_settings.dataset_path):
			os.makedirs(self.dataset_settings.dataset_path)
		if self.datasets_exist():
			self.load_datasets()
		else:
			if not osp.exists(self._dataset_settings.dataset_path):
				os.makedirs(self._dataset_settings.dataset_path)
			training_data_paths, testing_data_paths, validating_data_paths = self.get_dataset_raw_data_paths()
			training_data_path = osp.join(self.dataset_settings.dataset_path, self.training_dataset_name)
			testing_data_path = osp.join(self.dataset_settings.dataset_path, self.testing_dataset_name)
			validating_data_path = osp.join(self.dataset_settings.dataset_path, self.validating_dataset_name)
			
			# Assume that ifc entities are splited into training, testing and validating data dirs
			self.get_ifc_classes_map(validating_data_paths)
			
			training_dataset = self.create_dataset(training_data_paths, training_data_path)
			testing_dataset = self.create_dataset(testing_data_paths, testing_data_path)
			validating_dataset = self.create_dataset(validating_data_paths, validating_data_path)
			
			self._training_dataset = training_dataset
			self._testing_dataset = testing_dataset
			self._validating_dataset = validating_dataset
		return self._training_dataset, self._testing_dataset, self._validating_dataset
	
	def get_ifc_classes_map(self, data_paths:List[str]):
		ifc_classes = [os.path.basename(path).split("_")[0] for path in data_paths]
		self._ifc_classes_map = {ifc_class: i for i, ifc_class in enumerate(set(ifc_classes))}
		with open(osp.join(self.dataset_settings.dataset_path, self.ifc_classes_map_file_name), "w") as f:
			json.dump(self._ifc_classes_map, f)
	
	def load_ifc_classes_map(self):
		with open(osp.join(self.dataset_settings.dataset_path, self.ifc_classes_map_file_name), "r") as f:
			self._ifc_classes_map = json.load(f)
	
	def create_dataset(self, raw_data_paths: List[str], dataset_dir_path:str):
		self.copy_files_to_directory(raw_data_paths, dataset_dir_path)
		dataset = IfcPointNetDataset(dataset_dir_path, self._dataset_settings, self._ifc_classes_map)
		dataset.process()
		return dataset

	def copy_files_to_directory(self, paths:List[str], dir_path:str):
		if not osp.exists(dir_path):
			os.makedirs(dir_path)
		for path in paths:
			shutil.copy(path, dir_path)

	def get_dataset_raw_data_paths(self):
		training_data_raw_paths = []
		testing_data_raw_paths = []
		validating_data_raw_paths = []

		training_ratio = self.dataset_settings.training_data_ratio
		testing_ratio = self.dataset_settings.testing_data_ratio

		for ifc_class in self.dataset_settings.ifc_classes:
			class_dir_path = osp.join(self.dataset_settings.raw_data_path, ifc_class)
			all_files = os.listdir(class_dir_path)
			indices = np.random.permutation(len(all_files))
			selected_indices = indices[:self.dataset_settings.minimum_number_of_items_for_class]

			end_train = int(len(selected_indices) * training_ratio)
			end_test = end_train + int(len(selected_indices) * testing_ratio)
			
			training_files = [osp.join(class_dir_path, all_files[i]) for i in selected_indices[:end_train]]
			testing_files = [osp.join(class_dir_path, all_files[i]) for i in selected_indices[end_train:end_test]]
			validating_files = [osp.join(class_dir_path, all_files[i]) for i in selected_indices[end_test:]]
			
			training_data_raw_paths.extend(training_files)
			testing_data_raw_paths.extend(testing_files)
			validating_data_raw_paths.extend(validating_files)
			
		return training_data_raw_paths, testing_data_raw_paths, validating_data_raw_paths

	def datasets_exist(self):
		training_dataset_path = osp.join(self.dataset_settings.dataset_path, self.training_dataset_name + self.dataset_extension)
		testing_dataset_path = osp.join(self.dataset_settings.dataset_path, self.testing_dataset_name  + self.dataset_extension)
		validating_dataset_path = osp.join(self.dataset_settings.dataset_path, self.validating_dataset_name  + self.dataset_extension)
		return osp.exists(training_dataset_path) and osp.exists(testing_dataset_path) and osp.exists(validating_dataset_path)
		
	def load_datasets(self):
		self.load_ifc_classes_map()
		training_dataset_path = osp.join(self.dataset_settings.dataset_path, self.training_dataset_name)
		testing_dataset_path = osp.join(self.dataset_settings.dataset_path, self.testing_dataset_name)
		validating_dataset_path = osp.join(self.dataset_settings.dataset_path, self.validating_dataset_name)
		training_dataset = IfcPointNetDataset(training_dataset_path, self._dataset_settings)
		testing_dataset = IfcPointNetDataset(testing_dataset_path, self._dataset_settings)
		validating_dataset = IfcPointNetDataset(validating_dataset_path, self._dataset_settings)
		training_dataset.load()
		testing_dataset.load()
		validating_dataset.load()
		self._training_dataset = training_dataset
		self._testing_dataset = testing_dataset
		self._validating_dataset = validating_dataset

	@property
	def dataset_settings(self):
		return self._dataset_settings
	
	@property
	def ifc_classes_map(self):
		return self._ifc_classes_map
