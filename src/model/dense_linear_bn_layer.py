from torch import nn
import torch.nn.functional as F
import torch


class DenseLinearBN(nn.Module):
	def __init__(self, in_channels: int, out_channels: int, use_activation: bool = True):
		"""
		Initialize the DenseLinearBN module with a fully connected layer and a batch normalization layer.

		Args:
			in_channels (int): Number of features in the input.
			out_channels (int): Number of features in the output.

		Attributes:
			dense (nn.Linear): Fully connected (dense) layer.
			bn (nn.BatchNorm1d): Batch normalization layer.
		"""
		super(DenseLinearBN, self).__init__()
		self._use_activation = use_activation
		
		self._in_channels = in_channels
		self._out_channels = out_channels
		
		self.dense = nn.Linear(in_channels, out_channels)
		self.bn = nn.BatchNorm1d(out_channels)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass of the DenseLinearBN module. Applies a linear transformation,
		optionally followed by batch normalization (if batch size is more than one),
		and a ReLU activation.

		Args:
			x (torch.Tensor): Input tensor of shape (batch_size, in_features)

		Returns:
			torch.Tensor: Output tensor after applying the dense layer, batch normalization (conditional), and ReLU activation.
		"""
		x = self.dense(x)
		if x.shape[0] > 1:
			x = self.bn(x)
		if self._use_activation:
			x = F.relu(x)
		return x
	
	@property
	def in_channels(self) -> int:
		"""
		Number of features in the input.

		Returns:
			int: The number of input features.
		"""
		return self._in_channels
	
	@property
	def out_channels(self) -> int:
		"""
		Number of features in the output.

		Returns:
			int: The number of output features.
		"""
		return self._out_channels
