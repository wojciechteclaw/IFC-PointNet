import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# import sys
# sys.path.append(r"src\utils\create_dataset.py")
# from create_dataset import create_loaders
# sys.path.append(r"src\model\pointnet_classifier.py")
# from pointnet_classifier import PointNetClassifier
# from src.model.pointnet_classifier import PointNetClassifier
from .model.pointnet_classifier import PointNetClassifier


def train():
    pass

if __name__ == "__main__":

    ROOT = r"data\sample_1000_XYZ"

    train_dataloader, valid_dataloader, test_dataloader = create_loaders(ROOT, test_ratio=0.20, validation_ratio=0.10)
    model = PointNetClassifier()