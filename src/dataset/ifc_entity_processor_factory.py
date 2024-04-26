import os.path as osp

from src.dataset.extractors.extractor import Extractor
from src.dataset.extractors.obj_entity_extractor import ObjEntityExtractor
from src.dataset.extractors.pickle_entity_extractor import PickleEntityExtractor


class IfcEntityProcessorFactory:
	"""
	A factory class to create instances of different types of data extractors based on the file extension.
	This class provides a static method to get the appropriate extractor for handling specific file types.
	"""
	
	@staticmethod
	def get_extractor(file_path: str) -> Extractor:
		"""
		Returns an extractor instance appropriate for the given file path based on its extension.
		Args:
			file_path (str): The path of the file for which an extractor is needed.
		Returns:
			Extractor: An instance of an extractor suitable for the file type.
		Raises:
			ValueError: If the file extension is not supported, indicating the file cannot be processed.
		Examples:
			>>> factory = IfcEntityProcessorFactory.get_extractor("example.obj")
			>>> print(type(factory))
			<class 'src.dataset.extractors.obj_entity_extractor.ObjEntityExtractor'>

			>>> factory = IfcEntityProcessorFactory.get_extractor("example.pkl")
			>>> print(type(factory))
			<class 'src.dataset.extractors.pickle_entity_extractor.PickleEntityExtractor'>
		"""
		file_extension = osp.basename(file_path).split('.')[-1]
		match file_extension:
			case 'obj':
				return ObjEntityExtractor(file_path)
			case 'pkl':
				return PickleEntityExtractor(file_path)
			case _:
				raise ValueError(f'Unsupported file extension: {file_extension}')
