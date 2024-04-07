import torch
from torch import nn


class OrthogonalRegularizer(nn.Module):
	"""
	Orthogonal regularizer for PyTorch models.
	:param
	num_features: int Number of features in the tensor.
	l2reg: float L2 regularization factor.
	"""
	
	def __init__(self, num_features:int, l2reg:float=0.001):
		super(OrthogonalRegularizer, self).__init__()
		self._num_features = num_features
		self._l2reg = l2reg
		self._eye = torch.eye(num_features)
	
	def forward(self, x):
		"""
		Forward function.
		:param x:
		:return:
		"""
		x = x.view(-1, self._num_features, self._num_features)
		xxt = torch.bmm(x, x.transpose(1, 2))
		return torch.sum(self._l2reg * (xxt - self._eye).pow(2))

	@property
	def num_features(self):
		"""
		Number of features in the tensor.
		"""
		return self._num_features
	
	@property
	def l2reg(self):
		"""
		Return L2 regularization factor.
		"""
		return self._l2reg
