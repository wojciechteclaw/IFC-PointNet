import os.path as osp
from abc import abstractmethod

from src.dataset.normalization.enums.normalization_strategy import NormalizationStrategy


class Extractor:
	
	def __init__(self, file_path: str, normalization_strategy=NormalizationStrategy.NO_NORMALIZATION):
		self._file_path:str = file_path
		self._normalization_strategy = normalization_strategy

	@abstractmethod
	def extract(self, number_of_points_per_mesh_entity: int):
		raise NotImplementedError
	
	@property
	def file_path(self):
		return self._file_path

	@property
	def ifc_class(self):
		# Assume that class name is the first part of the file name and is separated by '_'
		return osp.basename(self._file_path).split('_')[0].lower()

	@property
	def normalization_strategy(self):
		return self._normalization_strategy