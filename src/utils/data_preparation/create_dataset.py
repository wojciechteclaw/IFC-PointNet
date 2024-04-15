import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.model.conv_1d_bn_block import Conv1dBN
from src.model.dense_linear_bn_layer import DenseLinearBN

from .data_preparation.PointCloudDataset import PointCloudDataset

def create_loaders(dir_path, test_ratio=0.20, validation_ratio=0.10):
    """To simplify data preparation, this method returns the three loaders: train, validation, test"""

    object_paths = []
    for subdir, dirs, files in os.walk(dir_path):
        for file in tqdm(files, desc=f"Processing {subdir}"):
            object_paths.append(os.path.join(subdir, file))

    # TODO normalize?
    # TODO embbed XYZ creation here?

    train_valid_paths, test_paths = train_test_split(
        object_paths, test_size=test_ratio, random_state=42
    )
    # now split into train and validation (test here means validation)
    train_paths, valid_paths = train_test_split(
        train_valid_paths,
        test_size=validation_ratio * (1 - test_ratio),
        random_state=42,
    )

    # Create DataSets
    train_dataset = PointCloudDataset(file_paths=train_paths)
    valid_dataset = PointCloudDataset(file_paths=valid_paths)
    test_dataset = PointCloudDataset(file_paths=test_paths)
    # Create a DataLoader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)
    return train_loader, valid_loader, test_loader


# Example usage
if __name__ == "__main__":

    ROOT = r"data\sample_1000_XYZ"

    train, valid, test = create_loaders(ROOT, test_ratio=0.20, validation_ratio=0.10)

    # Iterate through the dataset
    for batch in train:
        point_clouds, labels = batch
