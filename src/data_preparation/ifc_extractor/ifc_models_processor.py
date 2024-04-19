import glob
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from tqdm import tqdm

from src.data_preparation.ifc_extractor.single_ifc_model_processor import SingleIfcModelProcessor


class IfcModelsExtractor:
	
	def __init__(self,
				ifc_models_dir_path: str,
				extracted_elements_dir_path: str):
		self._extracted_elements_dir_path = extracted_elements_dir_path
		self._ifc_models_dir_path = ifc_models_dir_path
		self._ifc_models_paths: List[str] = []
		
	def extract_ifc_path_from_all_dirs(self):
		ifc_model_paths = glob.glob(self._ifc_models_dir_path + "/**/*.ifc", recursive=True)
		return ifc_model_paths
	
	def process_model(self, ifc_model_path):
		ifc_model_processor = SingleIfcModelProcessor(ifc_model_path=ifc_model_path,
													  extracted_elements_dir_path=self._extracted_elements_dir_path)
		ifc_model_processor.setup_ifc_file()
		success = ifc_model_processor.process_single_ifc_model()
		if not success:
			logging.error(f"Error while processing {ifc_model_path}")
	
	def extract_elements(self, use_multiprocessing: bool = False, num_workers: int = 10):
		ifc_model_paths = self.extract_ifc_path_from_all_dirs()
		if use_multiprocessing:
			assert num_workers > 0
			with ThreadPoolExecutor(max_workers=num_workers) as executor:
				futures = [executor.submit(self.process_model, path) for path in tqdm(ifc_model_paths)]
				for future in as_completed(futures):
					future.result()
		else:
			for path in tqdm(ifc_model_paths):
				self.process_model(path)