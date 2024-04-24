import unittest
import torch
from src.model.pointnet_classifier import PointNetClassifier

class PointNetClsTestBase():
    def setUpTestEntity(self, num_classes, feature_transform, points_per_mesh):
        self.pointnetcls = PointNetClassifier(num_classes=num_classes, feature_transform=feature_transform)
        self.k = num_classes
        self.feature_transform = feature_transform
        self.random_number_of_meshes = torch.randint(1, 10, (1,)).item()
        # Assuming the number of points in each point cloud is 2048
        self.x = torch.randn((self.random_number_of_meshes, 3, points_per_mesh), dtype=torch.float32)

    def test_classification_output_shape(self):
        log_softmax_output, _, _ = self.pointnetcls(self.x)
        expected_shape = (self.random_number_of_meshes, self.k)
        self.assertEqual(log_softmax_output.shape, expected_shape)

    def test_feature_transform_layers(self):
        if self.feature_transform:
            self.assertIsNotNone(self.pointnetcls.feat.fstn)
        else:
            self.assertIsNone(self.pointnetcls.feat.fstn)

    def test_no_nan_in_output(self):
        log_softmax_output, trans, trans_feat = self.pointnetcls(self.x)
        self.assertFalse(torch.isnan(log_softmax_output).any())
        self.assertFalse(torch.isnan(trans).any())
        if self.feature_transform:
            self.assertFalse(torch.isnan(trans_feat).any())

class TestPointNetClsWithFeatureTransform(unittest.TestCase, PointNetClsTestBase):
    def setUp(self):
        self.setUpTestEntity(num_classes=21, feature_transform=False, number_of_sample_points=841)

class TestPointNetClsWithFeatureTransformAndAdditionalPoints(unittest.TestCase, PointNetClsTestBase):
    def setUp(self):
        self.setUpTestEntity(num_classes=37, feature_transform=True, number_of_sample_points=4096)

class TestPointNetClsWithFeatureTransform(unittest.TestCase, PointNetClsTestBase):
    def setUp(self):
        self.setUpTestEntity(num_classes=37, feature_transform=True, number_of_sample_points=2137)