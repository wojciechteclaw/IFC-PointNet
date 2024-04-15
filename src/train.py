import os
import os.path as osp

from src.utils.data_preparation.create_loaders import create_loaders
from src.model.pointnet_classifier import PointNetClassifier


def train():
	pass


if __name__ == "__main__":
	pass
	DATASET_NAME = "sample_1000_XYZ"
	
	DATASET_PATH = osp.join(osp.dirname(osp.realpath(__file__)), "../data", DATASET_NAME)

	train_loader, valid_loader, test_loader = create_loaders(DATASET_PATH, test_ratio=0.20, validation_ratio=0.10)
	#
	# model = PointNetClassifier()
	# model.train(train_loader, valid_loader, epochs=10)
	# model.test(test_loader)
