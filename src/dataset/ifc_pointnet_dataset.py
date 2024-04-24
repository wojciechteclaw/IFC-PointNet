import os
import os.path as osp

import torch

from src.dataset.ifc_entity_processor_factory import IfcEntityProcessorFactory
from src.dataset.settings.dataset_settings import DatasetSettings


class IfcPointNetDataset:
	
	dataset_extension = ".pt"
	
	def __init__(self,
				 dataset_path: str,
				 dataset_settings: DatasetSettings,
				 ifc_classes_map: dict = {}):
		self.dataset_path = dataset_path
		self.dataset_settings = dataset_settings
		self._data = []
		self._ifc_classes_map = ifc_classes_map
		
	def __len__(self):
		return len(self._data)
	
	def __getitem__(self, idx):
		return self._data[idx]

	def load(self):
		path = self.get_path()
		self._data = torch.load(path)
	
	def process(self):
		for file in os.listdir(self.dataset_path):
			file_path = osp.join(self.dataset_path, file)
			extractor = IfcEntityProcessorFactory.get_extractor(file_path)
			points, ifc_class = extractor.extract(self.dataset_settings.number_of_point_per_mesh)
			tensor = torch.from_numpy(points).float().T
			ifc_class = self._ifc_classes_map.get(ifc_class, -1)
			self._data.append((tensor, ifc_class))
		self.save()

	def save(self):
		path = self.get_path()
		torch.save(self._data, path)

	def get_path(self):
		parent_dir = osp.dirname(self.dataset_path)
		basename = osp.basename(self.dataset_path)
		return osp.join(parent_dir, basename + self.dataset_extension)
