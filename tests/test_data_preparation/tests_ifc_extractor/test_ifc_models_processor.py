import os
import shutil

from src.data_preparation.ifc_extractor.ifc_models_processor import IfcModelsExtractor


current_dir = os.path.dirname(os.path.realpath(__file__))
sample_element_name = "norm_test_duct.obj"
assets_path = os.path.join(current_dir, "../../assets")
ifc_models_path = os.path.join(assets_path, "test_ifc_models")

def test_extract_ifc_path_from_all_dirs():
	ifc_models_extractor = IfcModelsExtractor(ifc_models_path, current_dir)
	ifc_model_paths = ifc_models_extractor.extract_ifc_path_from_all_dirs()
	assert len(ifc_model_paths) == 2
	
def test_extract_elements():
	resulting_dir = os.path.join(assets_path, "results")
	if not os.path.exists(resulting_dir):
		os.makedirs(resulting_dir)
	ifc_models_extractor = IfcModelsExtractor(ifc_models_path, resulting_dir)
	ifc_models_extractor.extract_elements()
	
def test_cleanup():
    resulting_dir = os.path.join(assets_path, "results")
    shutil.rmtree(resulting_dir)