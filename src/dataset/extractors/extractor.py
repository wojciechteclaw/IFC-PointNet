from abc import abstractmethod


class Extractor:
	
	def __init__(self, file_path: str):
		self._file_path = file_path

	@abstractmethod
	def extract(self):
		raise NotImplementedError
	
	@property
	def file_path(self):
		return self._file_path
