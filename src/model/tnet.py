import torch
from torch import nn
from src.model.conv_1d_bn_block import Conv1dBN
from src.model.dense_linear_bn_layer import DenseLinearBN


class Tnet(nn.Module):
	
	def __init__(self, num_features=3):
		super().__init__()
		self.num_features = num_features
		
		self.convolution1 = Conv1dBN(num_features, 64)
		self.convolution2 = Conv1dBN(64, 128)
		self.convolution3 = Conv1dBN(128, 1024)
		
		self.dense1 = DenseLinearBN(1024, 512)
		self.dense2 = DenseLinearBN(512, 256)
		self.fc3 = nn.Linear(256, num_features * num_features)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x.shape == (batch_size, number_of_point, 3)
		batch_size = x.size(0)
		xb = self.convolution1(x)
		xb = self.convolution2(xb)
		xb = self.convolution3(xb)
		
		pool = nn.MaxPool1d(xb.size(-1))(xb)
		flat = nn.Flatten(1)(pool)
		
		xb = self.dense1(flat)
		xb = self.dense2(xb)

		init = torch.eye(self.num_features, requires_grad=True).repeat(batch_size, 1, 1)
		if xb.is_cuda:
			init = init.cuda()
		matrix = self.fc3(xb).view(-1, self.num_features, self.num_features) + init
		return matrix
		