import torch
import os.path as osp

from src.dataset.dataset_generator import DatasetGenerator
from src.dataset.normalization.enums.normalization_strategy import NormalizationStrategy
from src.dataset.settings.dataset_settings import DatasetSettings
from src.model.loss_function import LossFunction
from src.model.pointnet_classifier import PointNetClassifier

def calculate_accuracy(y_pred, y_true):
    predictions = y_pred.argmax(1)
    correct = (predictions == y_true).sum().float()
    accuracy = correct / y_true.shape[0]
    return accuracy

def train_epoch(model: torch.nn.Module,
				loader: torch.utils.data.DataLoader,
				device:torch.device,
				loss_fn: torch.nn.Module,
				optimizer: torch.optim.Optimizer):
    model.train()
    total_loss, total_acc = 0, 0
    for batch in loader:
        points, labels = batch
        points = points.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # based on https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet_utils.py#L88
        logits, _, trans_feat = model(points)
        loss = loss_fn(logits, labels, trans_feat)
        #
        acc = calculate_accuracy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += acc.item()
    return total_loss / len(loader), total_acc / len(loader)

def validate_model(model: torch.nn.Module,
				   loader: torch.utils.data.DataLoader,
				   device:torch.device,
				   loss_fn: torch.nn.Module):
    model.eval()
    total_loss, total_acc = 0, 0
    with torch.no_grad():
        for batch in loader:
            points, labels = batch
            points = points.to(device)
            labels = labels.to(device)
            # based on https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet_utils.py#L88
            logits, _, trans_feat = model(points)
            loss = loss_fn(logits, labels, trans_feat)
            #
            acc = calculate_accuracy(logits, labels)
            total_loss += loss.item()
            total_acc += acc.item()
    return total_loss / len(loader), total_acc / len(loader)

def train(model: torch.nn.Module,
          train_loader: torch.utils.data.DataLoader,
          validation_loader: torch.utils.data.DataLoader,
          test_loader: torch.utils.data.DataLoader,
          epochs:int,
          loss_fn: torch.nn.Module,
          device: torch.device,
          optimizer: torch.optim.Optimizer,
          model_save_path):
    
    best_acc = 0
    model.to(device)
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, device, loss_fn, optimizer)
        if epoch % 8 == 0 and epoch != 0:
            val_loss, val_acc = validate_model(model, validation_loader, device, loss_fn)
            print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), model_save_path)
        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    test_loss, test_acc = validate_model(model, test_loader, device, loss_fn)
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')

if __name__ == "__main__":
    
    settings = {
        "raw_data_path": "../ready",
        "output_path": "../pointnet_results",
        "minimum_number_of_items_for_class": 256,
        "number_of_point_per_mesh": 2048,
        "normalization": NormalizationStrategy.ZERO_TO_ONE,
        "ifc_classes": ["IfcWall", "IfcSlab", "IfcDoor", "IfcWindow", "IfcBeam", "IfcColumn", "IfcStair", "IfcFlowFitting"],
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
    
    model_path = osp.join(dataset_settings.output_path, "pointnet_model.pth")
    
    model = PointNetClassifier(num_classes=NUMBER_OF_CLASSES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = LossFunction()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train(model, train_loader, validation_loader, test_loader, epochs, loss_fn, device, optimizer, model_path)
