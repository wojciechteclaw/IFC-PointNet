import torch
from torch import nn
import torch.nn.functional as F

from src.model.dense_linear_bn_layer import DenseLinearBN
from src.model.pointnet_features_transformer import PointNetFeaturesTransformer


class PointNetClassifier(nn.Module):
	def __init__(self, num_classes:int = 10):
		super().__init__()
		self.transform = PointNetFeaturesTransformer()
		self.fc1 = DenseLinearBN(1024, 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, num_classes)
		
		self.bn2 = nn.BatchNorm1d(256)
		self.dropout = nn.Dropout(p=0.5)
		self.logsoftmax = nn.LogSoftmax(dim=1)
	
	def forward(self, x: torch.Tensor) -> tuple:
		xb, matrix3x3, matrix64x64 = self.transform(x)
		xb = self.fc1(xb)
		xb = F.relu(self.bn2(self.dropout(self.fc2(xb))))
		output = self.fc3(xb)
		return self.logsoftmax(output), matrix3x3, matrix64x64
