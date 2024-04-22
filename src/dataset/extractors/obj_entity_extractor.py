from src.dataset.extractors.extractor import Extractor


class ObjEntityExtractor(Extractor):
	
	def __init__(self, file_path: str):
		super().__init__(file_path)
		
	def extract(self):
		print('hello from ObjExtractor')
