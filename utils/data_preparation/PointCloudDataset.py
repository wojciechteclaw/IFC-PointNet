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
        # TODO add scale factor
        # turn to tensors:
        points_tensor = torch.from_numpy(point_cloud)
        label_tensor = torch.tensor(label, dtype=torch.int)
        return points_tensor, label_tensor

