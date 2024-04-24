import os
import os.path as osp
import shutil

import pytest

from src.data_preparation.ifc_extractor.ifc_models_processor import IfcModelsExtractor
from src.dataset.dataset_generator import DatasetGenerator
from src.dataset.ifc_pointnet_dataset import IfcPointNetDataset
from src.dataset.settings.dataset_settings import DatasetSettings

current_dir = os.path.dirname(os.path.realpath(__file__))
assets_path = os.path.join(current_dir, "../assets")
ifc_models_path = os.path.join(assets_path, "test_ifc_models")

def test_aaa_prepare_ifc_models():
	resulting_dir = os.path.join(assets_path, "results")
	if not os.path.exists(resulting_dir):
		os.makedirs(resulting_dir)
	ifc_models_extractor = IfcModelsExtractor(ifc_models_path, resulting_dir)
	ifc_models_extractor.extract_elements()

@pytest.mark.parametrize("ifc_classes, minimum_number_of_items_for_class, train_ratio, test_ratio, validation_ratio", [
	(['ifcflowfitting', 'ifcflowsegment'], 20, 0.75, 0.15, 0.1),
	(['ifcflowfitting', 'ifcflowsegment'], 40, 0.75, 0.15, 0.1),
	(['ifcflowfitting', 'ifcflowsegment'], 80, 0.55, 0.25, 0.2),
	(['ifcspace'], 20, 0.8, 0.1, 0.1),
])
def test_get_dataset_raw_data_paths(ifc_classes, minimum_number_of_items_for_class, train_ratio, test_ratio, validation_ratio):
	settings = {
		"ifc_classes": ifc_classes,
		"minimum_number_of_items_for_class": minimum_number_of_items_for_class,
		"raw_data_path": osp.join(assets_path, "results"),
		"output_path": osp.join(assets_path, "sample_dataset_results"),
		"dataset_name": "test_dataset",
		"training_split_ratio": train_ratio,
		"testing_split_ratio": test_ratio,
		"validation_split_ratio": validation_ratio
	}
	dataset_settings = DatasetSettings(**settings)
	dataset_generator = DatasetGenerator(dataset_settings)
	training_paths, testing_paths, validation_paths = dataset_generator.get_dataset_raw_data_paths()
	
	number_of_classes = len(ifc_classes)
	assert len(training_paths) == number_of_classes * train_ratio * minimum_number_of_items_for_class
	assert len(testing_paths) == number_of_classes * test_ratio * minimum_number_of_items_for_class
	assert len(validation_paths) == number_of_classes * validation_ratio * minimum_number_of_items_for_class


@pytest.mark.parametrize("ifc_classes, minimum_number_of_items_for_class, train_ratio, test_ratio, validation_ratio, expected_train, expected_test, expected_validation", [
	(['ifcflowfitting', 'ifcflowsegment'], 60, 0.75, 0.15, 0.1, 90, 18, 12),
	(['ifcflowfitting', 'ifcflowsegment'], 40, 0.75, 0.15, 0.1, 60, 12, 8),
	(['ifcflowfitting', 'ifcflowsegment'], 80, 0.55, 0.25, 0.2, 88, 40, 32),
	(['ifcspace'], 20, 0.8, 0.1, 0.1, 16, 2, 2),
])
def test_copy_files_to_directory(ifc_classes,
								  minimum_number_of_items_for_class,
								  train_ratio,
								  test_ratio,
								  validation_ratio,
								  expected_train,
								  expected_test,
								  expected_validation):
	settings = {
		"ifc_classes": ifc_classes,
		"minimum_number_of_items_for_class": minimum_number_of_items_for_class,
		"raw_data_path": osp.join(assets_path, "results"),
		"output_path": osp.join(assets_path, "sample_dataset_results"),
		"dataset_name": "test_dataset",
		"training_split_ratio": train_ratio,
		"testing_split_ratio": test_ratio,
		"validation_split_ratio": validation_ratio
	}
	dataset_settings = DatasetSettings(**settings)
	dataset_generator = DatasetGenerator(dataset_settings)
	training_paths, testing_paths, validation_paths = dataset_generator.get_dataset_raw_data_paths()
	# train path
	train_path = osp.join(dataset_generator.dataset_settings.dataset_path, dataset_generator.training_dataset_name)
	dataset_generator.copy_files_to_directory(training_paths, train_path)
	assert len(os.listdir(train_path)) == expected_train
	shutil.rmtree(train_path)
	
	# test path
	test_path = osp.join(dataset_generator.dataset_settings.dataset_path, dataset_generator.testing_dataset_name)
	dataset_generator.copy_files_to_directory(testing_paths, test_path)
	assert len(os.listdir(test_path)) == expected_test
	shutil.rmtree(test_path)
	
	# validation path
	validation_path = osp.join(dataset_generator.dataset_settings.dataset_path, dataset_generator.validating_dataset_name)
	dataset_generator.copy_files_to_directory(validation_paths, validation_path)
	assert len(os.listdir(validation_path)) == expected_validation
	shutil.rmtree(validation_path)


@pytest.mark.parametrize(
	"ifc_classes, minimum_number_of_items_for_class, train_ratio, test_ratio, validation_ratio",
	[
		(['ifcflowfitting', 'ifcflowsegment'], 60, 0.75, 0.15, 0.1),
		(['ifcflowfitting', 'ifcflowsegment'], 40, 0.75, 0.15, 0.1),
		(['ifcflowfitting', 'ifcflowsegment'], 80, 0.55, 0.25, 0.2),
		(['ifcspace'], 20, 0.8, 0.1, 0.1),
	])
def test_create_dataset(ifc_classes,
								 minimum_number_of_items_for_class,
								 train_ratio,
								 test_ratio,
								 validation_ratio):
	settings = {
		"ifc_classes": ifc_classes,
		"minimum_number_of_items_for_class": minimum_number_of_items_for_class,
		"raw_data_path": osp.join(assets_path, "results"),
		"output_path": osp.join(assets_path, "sample_dataset_results"),
		"dataset_name": "test_dataset",
		"training_split_ratio": train_ratio,
		"testing_split_ratio": test_ratio,
		"validation_split_ratio": validation_ratio
	}
	dataset_settings = DatasetSettings(**settings)
	dataset_generator = DatasetGenerator(dataset_settings)
	training_paths, testing_paths, validation_paths = dataset_generator.get_dataset_raw_data_paths()
	# train path
	train_path = osp.join(dataset_generator.dataset_settings.dataset_path, dataset_generator.training_dataset_name)
	
	# test path
	test_path = osp.join(dataset_generator.dataset_settings.dataset_path, dataset_generator.testing_dataset_name)
	
	# validation path
	validation_path = osp.join(dataset_generator.dataset_settings.dataset_path, dataset_generator.validating_dataset_name)
	
	dataset_generator.get_ifc_classes_map(validation_paths)
	
	def test_datasets_loading():
		dataset_generator_jp2 = DatasetGenerator(dataset_settings)
		dataset_generator_jp2.load_datasets()
		assert len(dataset_generator_jp2._training_dataset) == len(training_paths)
		assert len(dataset_generator_jp2._testing_dataset) == len(testing_paths)
		assert len(dataset_generator_jp2._validating_dataset) == len(validation_paths)
		assert dataset_generator_jp2.ifc_classes_map == dataset_generator.ifc_classes_map
		
		assert isinstance(dataset_generator_jp2._training_dataset, IfcPointNetDataset)
		assert isinstance(dataset_generator_jp2._testing_dataset, IfcPointNetDataset)
		assert isinstance(dataset_generator_jp2._validating_dataset, IfcPointNetDataset)
		
	
	dataset_train = dataset_generator.create_dataset(training_paths, train_path)
	dataset_test = dataset_generator.create_dataset(testing_paths, test_path)
	dataset_valid = dataset_generator.create_dataset(validation_paths, validation_path)

	assert len(dataset_train) == len(training_paths)
	assert len(dataset_test) == len(testing_paths)
	assert len(dataset_valid) == len(validation_paths)
	
	assert dataset_train.dataset_extension == ".pt"
	assert dataset_test.dataset_extension == ".pt"
	assert dataset_valid.dataset_extension == ".pt"
	
	assert osp.exists(train_path + dataset_train.dataset_extension)
	assert osp.exists(test_path + dataset_test.dataset_extension)
	assert osp.exists(validation_path + dataset_valid.dataset_extension)
	
	test_datasets_loading()

	# clear
	shutil.rmtree(train_path)
	shutil.rmtree(test_path)
	shutil.rmtree(validation_path)

def test_zzz_cleanup():
    resulting_dir = os.path.join(assets_path, "results")
    sample_dataset_results_dir = os.path.join(assets_path, "sample_dataset_results")
    shutil.rmtree(resulting_dir)
    shutil.rmtree(sample_dataset_results_dir)