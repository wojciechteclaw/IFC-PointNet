from src.model.conv_1d_bn_block import Conv1dBN
import torch
import unittest

class Conv1dBnTestBase():
	
	def setUpTestEntity(self,
						in_channels,
						out_channels,
						number_of_points):
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.conv1d_bn = Conv1dBN(in_channels=self.in_channels, out_channels=self.out_channels)
		self.number_of_points = number_of_points
		
		self.x = torch.randn((1, self.in_channels, self.number_of_points), dtype=torch.float32)
		
	def test_in_channels_should_return_int_as_set(self):
		assert self.conv1d_bn.in_channels == self.in_channels
		
	def test_out_channels_should_return_int_as_set(self):
		assert self.conv1d_bn.out_channels == self.out_channels
		
	def test_conv_layer_should_return_torch_conv1d(self):
		assert isinstance(self.conv1d_bn.conv, torch.nn.Conv1d)
		
	def test_bn_layer_should_return_torch_batchnorm1d(self):
		assert isinstance(self.conv1d_bn.bn, torch.nn.BatchNorm1d)
		
	def test_forward_should_return_expected_output_shape(self):
		result = self.conv1d_bn(self.x)
		# simulate a single mesh
		assert result.shape == (1, self.out_channels, self.number_of_points)
		
	def test_forward_shouldnot_return_nan(self):
		result = self.conv1d_bn(self.x)
		assert not torch.isnan(result).any()
		
class TestConv1dBnBlockConfig1(unittest.TestCase, Conv1dBnTestBase):
	
	def setUp(self):
		self.setUpTestEntity(in_channels=3,
							 out_channels=5,
							 number_of_points=2137)

class TestConv1dBnBlockConfig2(unittest.TestCase, Conv1dBnTestBase):
	
	def setUp(self):
		self.setUpTestEntity(in_channels=3,
							 out_channels=5,
							 number_of_points=841)