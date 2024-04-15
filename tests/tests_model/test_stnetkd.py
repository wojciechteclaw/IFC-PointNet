import unittest
import torch
from src.model.conv_1d_bn_block import Conv1dBN
from src.model.dense_linear_bn_layer import DenseLinearBN
from src.model.stnetkd import STNetkd

class STNetkdTestBase():
    def setUpTestEntity(self, num_classes, number_of_points):
        self.num_classes = num_classes
        self.stnetkd = STNetkd(num_features=self.num_classes)
        self.number_of_points = number_of_points
        self.random_number_of_meshes = torch.randint(1, 10, (1,)).item()
        self.x = torch.randn((self.random_number_of_meshes, self.num_classes, self.number_of_points), dtype=torch.float32)

    def test_layers_initialization(self):
        self.assertIsInstance(self.stnetkd.conv1, Conv1dBN)
        self.assertIsInstance(self.stnetkd.conv2, Conv1dBN)
        self.assertIsInstance(self.stnetkd.conv3, Conv1dBN)
        self.assertIsInstance(self.stnetkd.fc1, DenseLinearBN)
        self.assertIsInstance(self.stnetkd.fc2, DenseLinearBN)
        self.assertIsInstance(self.stnetkd.fc3, torch.nn.Linear)

    def test_forward_shape(self):
        result = self.stnetkd(self.x)
        expected_shape = (self.random_number_of_meshes, self.num_classes, self.num_classes)
        self.assertEqual(result.shape, expected_shape)

    def test_forward_no_nan(self):
        result = self.stnetkd(self.x)
        self.assertFalse(torch.isnan(result).any())

class TestSTNetkdConfig1(unittest.TestCase, STNetkdTestBase):
    def setUp(self):
        self.setUpTestEntity(num_classes=64, number_of_points=1024)

class TestSTNetkdConfig2(unittest.TestCase, STNetkdTestBase):
    def setUp(self):
        self.setUpTestEntity(num_classes=128, number_of_points=512)
