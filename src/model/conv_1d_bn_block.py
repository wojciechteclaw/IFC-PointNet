from torch import nn
import torch.nn.functional as F
import torch


class Conv1dBN(nn.Module):
	def __init__(self, in_channels: int, out_channels: int):
		"""
		Initialize the Conv1dBN module with convolutional and batch normalization layers.

		Args:
			in_channels (int): Number of channels in the input.
			out_channels (int): Number of channels produced by the convolution.

		Attributes:
			conv (nn.Conv1d): 1D convolutional layer.
			bn (nn.BatchNorm1d): Batch normalization layer.
		"""
		super(Conv1dBN, self).__init__()
		
		self._in_channels = in_channels
		self._out_channels = out_channels
		
		self.conv = nn.Conv1d(in_channels=in_channels,
							  out_channels=out_channels,
							  kernel_size=1,
							  padding="valid")
		self.bn = nn.BatchNorm1d(out_channels, momentum=0.0)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass of the Conv1dBN module.

		Args:
			x (torch.Tensor): Input tensor of shape (batch, channels, length)

		Returns:
			torch.Tensor: Output tensor after applying convolution, batch normalization, and ReLU activation.
		"""
		x = self.conv(x)
		x = self.bn(x)
		x = F.relu(x)
		return x
	
	@property
	def in_channels(self) -> int:
		"""
		Number of channels in the input.

		Returns:
			int: The number of input channels.
		"""
		return self._in_channels
	
	@property
	def out_channels(self) -> int:
		"""
		Number of channels produced by the convolution.

		Returns:
			int: The number of output channels.
		"""
		return self._out_channels
