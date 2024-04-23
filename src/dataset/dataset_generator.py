import os
import os.path as osp
import shutil
from typing import List

import numpy as np
import torch

from src.dataset.settings.dataset_settings import DatasetSettings


class DatasetGenerator:
	
	training_dataset_name = "training"
	testing_dataset_name = "testing"
	validating_dataset_name = "validating"
	
	
	def __init__(self, dataset_settings: DatasetSettings):
		self._dataset_settings = dataset_settings
		self._training_dataset = None
		self._testing_dataset = None
		self._validating_dataset = None
	
	def process_data(self):
		if not osp.exists(self.dataset_settings.dataset_path):
			os.makedirs(self.dataset_settings.dataset_path)
		if self.datasets_exist():
			self.load_datasets()
		else:
			if not osp.exists(self._dataset_settings.dataset_path):
				os.makedirs(self._dataset_settings.dataset_path)
			training_data_paths, testing_data_paths, validating_data_paths = self.get_dataset_raw_data_paths()
			trainig_data_path = osp.join(self.dataset_settings.dataset_path, self.training_dataset_name)
			testing_data_path = osp.join(self.dataset_settings.dataset_path, self.testing_dataset_name)
			validating_data_path = osp.join(self.dataset_settings.dataset_path, self.validating_dataset_name)
			self.copy_files_to_directory(training_data_paths, trainig_data_path)
			self.copy_files_to_directory(testing_data_paths, testing_data_path)
			self.copy_files_to_directory(validating_data_paths, validating_data_path)
		return self.get_datasets()
	
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
		trainig_dataset_path = osp.join(self.dataset_settings.dataset_path, self.training_dataset_name + ".pt")
		testing_dataset_path = osp.join(self.dataset_settings.dataset_path, self.testing_dataset_name  + ".pt")
		validating_dataset_path = osp.join(self.dataset_settings.dataset_path, self.validating_dataset_name  + ".pt")
		return osp.exists(trainig_dataset_path) and osp.exists(testing_dataset_path) and osp.exists(validating_dataset_path)
		
	def load_datasets(self):
		training_dataset = torch.load(osp.join(self.dataset_settings.dataset_path, self.training_dataset_name + ".pt"))
		testing_dataset = torch.load(osp.join(self.dataset_settings.dataset_path, self.testing_dataset_name + ".pt"))
		validating_dataset = torch.load(osp.join(self.dataset_settings.dataset_path, self.validating_dataset_name + ".pt"))
		self._training_dataset = training_dataset
		self._testing_dataset = testing_dataset
		self._validating_dataset = validating_dataset

	@property
	def dataset_settings(self):
		return self._dataset_settings
