import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


class PointCloudDataset(Dataset):
    """Stores the samples and their corresponding labels"""

    def __init__(self, file_paths):
        """loading the list of file paths and extracting information"""
        self.object_paths, self.labels, self.uids = [], [], []
        for file in tqdm(file_paths, desc=f"Processing"):
            self.uids.append(file[0:6])
            self.object_paths.append(file)
            self.labels.append(Path(file).stem)
        # change string labels to integers ('IfcWall'-->12)
        self.label_mapping = {
            label: idx for idx, label in enumerate(sorted(set(self.labels)))
        }
        self.labels = [self.label_mapping[label] for label in self.labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, pos):
        """fetching an individual item from the dataset. Load the label and the point cloud from an XYZ file, apply transformations."""
        point_cloud = np.loadtxt(self.object_paths[pos], dtype=np.float32)
        label = self.labels[pos]
        # TODO rotate?
        # turn to tensors:
        points_tensor = torch.from_numpy(point_cloud)
        label_tensor = torch.tensor(label, dtype=torch.int)
        return points_tensor, label_tensor


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
