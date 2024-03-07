from torch import nn
import torch.nn.functional as F

class Conv1dBN(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(Conv1dBN, self).__init__()
		
		self._in_channels = in_channels
		self._out_channels = out_channels
		
		self.conv = nn.Conv1d(in_channels = in_channels,
							  out_channels = out_channels,
							  kernel_size=1,
							  padding="valid")
		self.bn = nn.BatchNorm1d(out_channels, momentum=0.0)
	
	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		x = F.relu(x)
		return x
	
	@property
	def in_channels(self):
		return self._in_channels
	
	@property
	def out_channels(self):
		return self._out_channels
