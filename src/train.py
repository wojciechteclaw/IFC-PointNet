import torch

from src.dataset.dataset_generator import DatasetGenerator
from src.dataset.normalization.enums.normalization_strategy import NormalizationStrategy
from src.dataset.settings.dataset_settings import DatasetSettings
from src.model.pointnet_classifier import PointNetClassifier

def calculate_accuracy(y_pred, y_true):
    predictions = y_pred.argmax(1)
    correct = (predictions == y_true).sum().float()
    accuracy = correct / y_true.shape[0]
    return accuracy


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
			# We use only logits
			logits = pred[0]
			loss = loss_fn(logits, labels)
			acc = calculate_accuracy(logits, labels)
			loss.backward()
			optimizer.step()
			
			torch.save(model.state_dict(), "pointnet_model.pt")
			exit()
		print(f"Epoch {epoch} training loss: {loss.item()} acc: {acc.item()}")
		if epoch % 8 == 0 and epoch != 0:
			model.eval()
			with torch.no_grad():
				for batch in validation_loader:
					points, labels = batch
					points = points.to(device)
					labels = labels.to(device)
					pred = model(points)
					logits = pred[0]
					acc = calculate_accuracy(logits, labels)
					loss = loss_fn(logits, labels)
				print(f"Epoch {epoch} validation loss: {loss.item()} acc: {acc.item()}")
	# Test the model
	model.eval()
	with torch.no_grad():
		for batch in test_loader:
			points, labels = batch
			points = points.to(device)
			labels = labels.to(device)
			pred = model(points)
			logits = pred[0]
			acc = calculate_accuracy(logits, labels)
			loss = loss_fn(logits, labels)
		print(f"Test loss: {loss.item()} acc: {acc.item()}")


if __name__ == "__main__":
	
	settings = {
		"raw_data_path": "../ready",
		"output_path": "../pointnet_results",
		"minimum_number_of_items_for_class": 100,
		"number_of_point_per_mesh": 2048,
		"normalization": NormalizationStrategy.ZERO_TO_ONE,
		"ifc_classes": ["IfcWall", "IfcSlab", "IfcDoor", "IfcWindow", "IfcBeam", "IfcColumn", "IfcStair"],
		"training_split_ratio": 0.70,
		"testing_split_ratio": 0.15,
		"validation_split_ratio": 0.15
	}
	
	dataset_settings = DatasetSettings(**settings)
	dataset_generator = DatasetGenerator(dataset_settings=dataset_settings)
	train_dataset, validation_dataset, test_dataset = dataset_generator.process_data()
	
	NUMBER_OF_CLASSES = len(dataset_generator.ifc_classes_map.keys())
	epochs = 1024

	# Create a DataLoader
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)
	validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=32, shuffle=True, num_workers=4)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
	#
	model = PointNetClassifier(num_classes=NUMBER_OF_CLASSES)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	loss_fn = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	#
	train(model, train_loader, validation_loader, test_loader, epochs, loss_fn, device, optimizer)
