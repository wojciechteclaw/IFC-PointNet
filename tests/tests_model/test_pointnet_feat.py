import unittest
import torch
from src.model.pointnet_feat import PointNetfeat

class PointNetfeatTestBase():
    def setUpTestEntity(self, number_of_points, feature_transform):
        self.pointnetfeat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.number_of_points = number_of_points
        self.random_number_of_meshes = torch.randint(64, 128, (1,)).item()
        self.x = torch.randn((self.random_number_of_meshes, 3, self.number_of_points), dtype=torch.float32)
        self.feature_transform = feature_transform

    def test_stn_initialization(self):
        self.assertIsNotNone(self.pointnetfeat.stn)

    def test_fstn_initialization(self):
        if self.feature_transform:
            self.assertIsNotNone(self.pointnetfeat.fstn)
        else:
            self.assertIsNone(self.pointnetfeat.fstn)

    def test_forward_global_feature_shape(self):
        global_feat, trans, trans_feat = self.pointnetfeat(self.x)
        expected_shape = (self.random_number_of_meshes, 1024)
        self.assertEqual(global_feat.shape, expected_shape)

    def test_forward_transformation_matrix_shape(self):
        _, trans, trans_feat = self.pointnetfeat(self.x)
        self.assertEqual(trans.shape, (self.random_number_of_meshes, 3, 3))
        if self.feature_transform:
            self.assertEqual(trans_feat.shape, (self.random_number_of_meshes, 64, 64))
        else:
            assert True

    def test_forward_no_nan(self):
        global_feat, trans, trans_feat = self.pointnetfeat(self.x)
        self.assertFalse(torch.isnan(global_feat).any())
        self.assertFalse(torch.isnan(trans).any())
        if self.feature_transform:
            self.assertFalse(torch.isnan(trans_feat).any())

class TestPointNetfeatGlobalFeatureTrue(unittest.TestCase, PointNetfeatTestBase):
    def setUp(self):
        self.setUpTestEntity(number_of_points=1024, feature_transform=False)

class TestPointNetfeatGlobalFeatureTrueWithFeatureTransform(unittest.TestCase, PointNetfeatTestBase):
    def setUp(self):
        self.setUpTestEntity(number_of_points=2044, feature_transform=True)
		
class TestPointNetfeatGlobalFeatureFalseWithFeatureTransform(unittest.TestCase, PointNetfeatTestBase):
    def setUp(self):
        self.setUpTestEntity(number_of_points=2137, feature_transform=False)

