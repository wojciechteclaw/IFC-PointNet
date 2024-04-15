import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from src.utils.data_preparation.point_cloud_dataset import PointCloudDataset

# OPTIONAL I would suggest to move this method to some class like LoadersGenerator
# and then add a static method to create_loaders
def create_loaders(dir_path, test_ratio=0.20, validation_ratio=0.10):
    """To simplify data preparation, this method returns the three loaders: train, validation, test"""

    object_paths = []
    for subdir, dirs, files in os.walk(dir_path):
        for file in tqdm(files, desc=f"Processing {subdir}"):
            object_paths.append(os.path.join(subdir, file))

    # TODO normalize?
    # TODO embbed XYZ creation here?

    # OPTIONAL TODO: is it necessary to install the package to use a single method ? It's quite simple to implement
    # Example:
    # 		file_names = np.array([f.split(".")[0] for f in os.listdir(self._raw_data_dir_meshes)])
    # 		count_files = len(file_names)
    # 		indices = np.arange(count_files)
    # 		np.random.shuffle(indices)
    #
    # 		# Calculate split sizes
    # 		test_size = int(self._data_split.testing_ratio * count_files)
    # 		validation_size = int(self._data_split.validation_ratio * count_files)
    # 		train_size = count_files - test_size - validation_size
    #
    # 		assert train_size + test_size + validation_size == count_files
    #
    # 		train_indices = indices[:train_size]
    # 		test_indices = indices[train_size:train_size + test_size]
    # 		validation_indices = indices[train_size + test_size:]
    #
    # 		return (file_names[train_indices].tolist(),
    # 				file_names[test_indices].tolist(),
    # 				file_names[validation_indices].tolist())
    train_valid_paths, test_paths = train_test_split(
        object_paths, test_size=test_ratio, random_state=42
    )
    # now split into train and validation (test here means validation)
    # TODO splitting should be done separatelly for each class to avoid class imbalance in datasets
    
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
    # TODO move to training loop. Return datasets only.
    # It is much easier to adjust the number of workers and batch size depending upon the hardware limitations
    # and the size of the dataset.
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)
    return train_loader, valid_loader, test_loader


# TODO Remove - write tests if necessary to check the correctness of the solution
# Example usage
if __name__ == "__main__":

    ROOT = r"data\sample_1000_XYZ"

    train, valid, test = create_loaders(ROOT, test_ratio=0.20, validation_ratio=0.10)

    # Iterate through the dataset
    for batch in train:
        point_clouds, labels = batch
