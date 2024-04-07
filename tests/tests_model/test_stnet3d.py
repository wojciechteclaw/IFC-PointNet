import unittest
import torch
from src.model.conv_1d_bn_block import Conv1dBN
from src.model.dense_linear_bn_layer import DenseLinearBN
from src.model.stnet3d import STNet3d

class STNet3dTestBase():
    def setUpTestEntity(self, number_of_points):
        self.stnet3d = STNet3d()
        self.number_of_points = number_of_points
        self.random_number_of_meshes = torch.randint(1, 10, (1,)).item()
        self.x = torch.randn((self.random_number_of_meshes, 3, self.number_of_points), dtype=torch.float32)

    def test_layers_initialization(self):
        self.assertIsInstance(self.stnet3d.conv1, Conv1dBN)
        self.assertIsInstance(self.stnet3d.conv2, Conv1dBN)
        self.assertIsInstance(self.stnet3d.conv3, Conv1dBN)
        self.assertIsInstance(self.stnet3d.fc1, DenseLinearBN)
        self.assertIsInstance(self.stnet3d.fc2, DenseLinearBN)
        self.assertIsInstance(self.stnet3d.fc3, torch.nn.Linear)

    def test_forward_shape(self):
        result = self.stnet3d(self.x)
        expected_shape = (self.random_number_of_meshes, 3, 3)
        self.assertEqual(result.shape, expected_shape)

    def test_forward_no_nan(self):
        result = self.stnet3d(self.x)
        self.assertFalse(torch.isnan(result).any())

class TestSTNet3dConfig1(unittest.TestCase, STNet3dTestBase):
    def setUp(self):
        self.setUpTestEntity(number_of_points=1024)

class TestSTNet3dConfig2(unittest.TestCase, STNet3dTestBase):
    def setUp(self):
        self.setUpTestEntity(number_of_points=512)
