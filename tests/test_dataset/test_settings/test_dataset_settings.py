import os
import os.path as osp
import shutil

import pytest

from src.data_preparation.ifc_extractor.ifc_models_processor import IfcModelsExtractor
from src.dataset.settings.dataset_settings import DatasetSettings

current_dir = os.path.dirname(os.path.realpath(__file__))
assets_path = os.path.join(current_dir, "../../assets")
ifc_models_path = os.path.join(assets_path, "test_ifc_models")

def test_aaa_prepare_ifc_models():
	resulting_dir = os.path.join(assets_path, "results")
	if not os.path.exists(resulting_dir):
		os.makedirs(resulting_dir)
	ifc_models_extractor = IfcModelsExtractor(ifc_models_path, resulting_dir)
	ifc_models_extractor.extract_elements()

@pytest.mark.parametrize("ifc_classes, expected_list_of_valid_categories, minimum_number_of_items_for_class", [
	(['ifcbeam', 'ifcdoor', 'ifcflowfitting','ifcflowsegment','ifcfurnishingelement','ifcopeningelement', 'ifcwallstandardcase'],
     ['ifcflowfitting','ifcflowsegment','ifcfurnishingelement','ifcopeningelement', 'ifcwallstandardcase'],
	 50),
	
	([],
	 ['ifcflowfitting','ifcflowsegment','ifcfurnishingelement','ifcopeningelement', 'ifcwallstandardcase'],
	 50),
	([],
	 ['ifcbeam', 'ifcbuildingelementproxy', 'ifccovering', 'ifcdoor', 'ifcflowfitting', 'ifcflowsegment',
	  'ifcflowterminal', 'ifcfooting', 'ifcfurnishingelement', 'ifcmember', 'ifcopeningelement', 'ifcrailing',
	  'ifcslab', 'ifcspace', 'ifcstairflight', 'ifcwall', 'ifcwallstandardcase', 'ifcwindow'],
	 0),
	(['ifcbeam', 'ifcdoor', 'ifcflowfitting', 'ifcflowsegment', 'ifcfurnishingelement', 'ifcopeningelement', 'ifcwallstandardcase'],
	 ['ifcflowfitting', 'ifcflowsegment'],
	 100),
	([],
	 ['ifcflowfitting', 'ifcflowsegment'],
	 100),
])
def test_dataset_settings__get_valid_ifc_classes(ifc_classes, expected_list_of_valid_categories, minimum_number_of_items_for_class):
	settings = {
		"ifc_classes": ifc_classes,
		"minimum_number_of_items_for_class": minimum_number_of_items_for_class,
		"raw_data_path": osp.join(assets_path, "results"),
		"output_path": osp.join(assets_path, "sample_output")
	}
	
	dataset_settings = DatasetSettings(**settings)
	valid_categories = dataset_settings.ifc_classes
	assert valid_categories == expected_list_of_valid_categories


@pytest.mark.parametrize("ifc_classes, number_of_points_per_mesh, minimum_number_of_items_for_class, expected_dataset_name", [
	(['ifcbeam', 'ifcdoor', 'ifcflowfitting', 'ifcflowsegment', 'ifcfurnishingelement', 'ifcopeningelement',
	  'ifcwallstandardcase'],
	 2048,
	 50,
	"5_classes_with_2048_points_per_mesh-min_50_items_per_class"),
	
	([],
	 1024,
	 50,
	 "5_classes_with_1024_points_per_mesh-min_50_items_per_class"),
	
	([],
	 2137,
	 0,
	 "18_classes_with_2137_points_per_mesh-min_0_items_per_class"),
	
	(['ifcbeam', 'ifcdoor', 'ifcflowfitting', 'ifcflowsegment', 'ifcfurnishingelement', 'ifcopeningelement',
	  'ifcwallstandardcase'],
	 841,
	 100,
	 "2_classes_with_841_points_per_mesh-min_100_items_per_class"),
	([],
	 700,
	 100,
	 "2_classes_with_700_points_per_mesh-min_100_items_per_class"),
])
def test_dataset_settings__get_dataset_name(ifc_classes,
											number_of_points_per_mesh,
											minimum_number_of_items_for_class,
											expected_dataset_name):
	settings = {
		"ifc_classes": ifc_classes,
		"minimum_number_of_items_for_class": minimum_number_of_items_for_class,
		"raw_data_path": osp.join(assets_path, "results"),
		"output_path": osp.join(assets_path, "sample_output"),
		"number_of_point_per_mesh": number_of_points_per_mesh
	}
	
	dataset_settings = DatasetSettings(**settings)
	assert dataset_settings.dataset_name == expected_dataset_name


def test_zzz_cleanup():
    resulting_dir = os.path.join(assets_path, "results")
    shutil.rmtree(resulting_dir)