import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from pathlib import Path



class PointCloudDataset(Dataset):
    """Stores the samples and their corresponding labels"""

    # TODO: I would suggest to to split the __init__ method into two methods: __init__ and process
    # TODO When we create a dataset all the data should be stored in the memory. __getitem__ should only return them,
    # not open them and process them. It can lead to low performance.
    
    # I would suggest to create a separate method like: process_paths where we open every single file, extract the data and store it in the memory.
    # After the iteration process we dump the whole file:
    # torch.save(<MY_DATA>, osp.join(<SOME PATH>, "data.pt"))
    
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
        # turn to tensors
        # TODO: convert to a tensor before, because this method will be called extremely often. In my opition this method should be only:
        # return self.points_tensor[pos], self.label_tensor[pos]
        points_tensor = torch.from_numpy(point_cloud)
        label_tensor = torch.tensor(label, dtype=torch.int)
        return points_tensor, label_tensor


# data: List[Tuple]
# Each tuple should contains points_tensor, label_tensor
