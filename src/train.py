import os.path as osp
import torch

from src.utils.data_preparation.create_dataset import create_dataset
from src.model.pointnet_classifier import PointNetClassifier

    # TODO looks like we already have dataloaders below, but can't see where batch size and worksers are configured 
	# Create a DataLoader
    # train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    # valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=True, num_workers=2)
    # test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)


def train(model: torch.nn.Module,
		  train_loader: torch.utils.data.DataLoader,
		  validation_loader: torch.utils.data.DataLoader,
		  test_loader: torch.utils.data.DataLoader,
		  epochs: int,
		  loss_fn: torch.nn.Module,
		  device: torch.device,
		  optimizer: torch.optim.Optimizer):
	
	model.to(device)
	for epoch in range(epochs):
		model.train()
		for batch in train_loader:
			points, labels = batch
			points = points.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()
			pred = model(points)
			loss = loss_fn(pred, labels)
			loss.backward()
			optimizer.step()
		print(f"Epoch {epoch} training loss: {loss.item()}")
		model.eval()
		with torch.no_grad():
			for batch in validation_loader:
				points, labels = batch
				points = points.to(device)
				labels = labels.to(device)
				pred = model(points)
				loss = loss_fn(pred, labels)
			print(f"Epoch {epoch} validation loss: {loss.item()}")
			
	# Test the model
	model.eval()
	with torch.no_grad():
		for batch in test_loader:
			points, labels = batch
			points = points.to(device)
			labels = labels.to(device)
			pred = model(points)
			loss = loss_fn(pred, labels)
		print(f"Test loss: {loss.item()}")


if __name__ == "__main__":
	
	DATASET_NAME = "sample_1000_XYZ"
	DATASET_PATH = osp.join(osp.dirname(osp.abspath(__file__)), "../data", DATASET_NAME)
	
	epochs = 1024
	
	train_loader, validation_loader, test_loader = create_dataset(DATASET_PATH, test_ratio=0.20, validation_ratio=0.10)
	
	model = PointNetClassifier()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	loss_fn = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	
	train(model, train_loader, validation_loader, test_loader, epochs, loss_fn, device, optimizer)
