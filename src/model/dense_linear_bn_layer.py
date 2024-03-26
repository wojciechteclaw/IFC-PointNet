from torch import nn
import torch.nn.functional as F


class DenseLinearBN(nn.Module):
		def __init__(self, in_channels, out_channels):
			
			self._in_channels = in_channels
			self._out_channels = out_channels
			
			super(DenseLinearBN, self).__init__()
			self.dense = nn.Linear(in_channels, out_channels)
			self.bn = nn.BatchNorm1d(out_channels, momentum=0.0)
		
		def forward(self, x):
			x = self.dense(x)
			x = self.bn(x)
			return F.relu(x)

		@property
		def in_channels(self):
			return self._in_channels
		
		@property
		def out_channels(self):
			return self._out_channels
