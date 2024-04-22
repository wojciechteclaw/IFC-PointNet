import os.path as osp

from src.dataset.extractors.obj_entity_extractor import ObjEntityExtractor
from src.dataset.extractors.pickle_entity_extractor import PickleEntityExtractor


class IfcEntityProcessorFactory:
	@staticmethod
	def get_extractor(file_path:str):
		file_extension = osp.basename(file_path).split('.')[-1]
		match file_extension:
			case 'obj':
				return ObjEntityExtractor(file_path)
			case 'ifc':
				return PickleEntityExtractor(file_path)
			case _:
				raise ValueError(f'Unsupported file extension: {file_extension}')
