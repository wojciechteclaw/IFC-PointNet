import os
from src.data_preparation.ifc_extractor.ifc_models_processor import IfcModelsExtractor


current_dir = os.path.dirname(os.path.realpath(__file__))
sample_element_name = "norm_test_duct.obj"
ifc_models_path = os.path.join(current_dir, "../../assets/test_ifc_models")

def test_extract_ifc_path_from_all_dirs():
	ifc_models_extractor = IfcModelsExtractor(ifc_models_path, current_dir)
	ifc_model_paths = ifc_models_extractor.extract_ifc_path_from_all_dirs()
	assert len(ifc_model_paths) == 2
	
def test_extract_elements():
	export_result = os.path.join(os.path.dirname(ifc_models_path), "extracted_elements")
	
	ifc_models_extractor = IfcModelsExtractor(ifc_models_path, export_result)
	ifc_models_extractor.extract_elements()