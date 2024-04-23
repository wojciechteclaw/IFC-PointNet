

class IfcPointNetDataset:
	
		def __init__(self, dataset_path: str):
			self.dataset_path = dataset_path
			self._data = []
			
		def __len__(self):
			return len(self._data)
		
		def __getitem__(self, idx):
			return self._data[idx]