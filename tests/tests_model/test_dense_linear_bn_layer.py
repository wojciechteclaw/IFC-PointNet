import random
import torch
import unittest
from src.model.dense_linear_bn_layer import DenseLinearBN

class DenseLinearBnTestBase():

    def setUpTestEntity(self, in_channels, out_channels, number_of_features):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dense_linear_bn = DenseLinearBN(in_channels=self.in_channels, out_channels=self.out_channels)
        self.number_of_features = number_of_features
        self.random_number_of_samples = random.randint(1, 10)

        self.x = torch.randn((self.random_number_of_samples, self.in_channels), dtype=torch.float32)

    def test_in_channels_should_return_int_as_set(self):
        assert self.dense_linear_bn.in_channels == self.in_channels

    def test_out_channels_should_return_int_as_set(self):
        assert self.dense_linear_bn.out_channels == self.out_channels

    def test_dense_layer_should_return_torch_linear(self):
        assert isinstance(self.dense_linear_bn.dense, torch.nn.Linear)

    def test_bn_layer_should_return_torch_batchnorm1d(self):
        assert isinstance(self.dense_linear_bn.bn, torch.nn.BatchNorm1d)

    def test_forward_should_return_expected_output_shape(self):
        result = self.dense_linear_bn(self.x)
        assert result.shape == (self.random_number_of_samples, self.out_channels)

    def test_forward_shouldnot_return_nan(self):
        result = self.dense_linear_bn(self.x)
        assert not torch.isnan(result).any()

class TestDenseLinearBNConfig1(unittest.TestCase, DenseLinearBnTestBase):

    def setUp(self):
        self.setUpTestEntity(in_channels=128, out_channels=64, number_of_features=1000)

class TestDenseLinearBNConfig2(unittest.TestCase, DenseLinearBnTestBase):

    def setUp(self):
        self.setUpTestEntity(in_channels=64, out_channels=32, number_of_features=500)
