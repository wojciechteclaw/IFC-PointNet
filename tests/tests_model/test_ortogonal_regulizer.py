import torch
import unittest
from src.model.ortogonal_regulizer import OrthogonalRegularizer

class OrthogonalRegularizerTestBase():

    def setUpTestEntity(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.regularizer = OrthogonalRegularizer(num_features=self.num_features, l2reg=self.l2reg)

    def test_l2reg_should_return_float_as_set(self):
        assert self.regularizer.l2reg == self.l2reg

    def test_regularization_loss_for_orthogonal_matrix(self):
        orthogonal_matrix = torch.eye(self.num_features).repeat(2, 1, 1)  # Using 2 as a batch size for example
        loss = self.regularizer(orthogonal_matrix)
        expected_loss = torch.tensor(0.0)
        self.assertTrue(torch.isclose(loss, expected_loss, atol=1e-6))

    def test_regularization_loss_for_non_orthogonal_matrix(self):
        non_orthogonal_matrix = torch.rand(2, self.num_features, self.num_features)  # Non-orthogonal random matrices
        loss = self.regularizer(non_orthogonal_matrix)
        self.assertTrue(loss > 0)

class TestOrthogonalRegularizerConfig1(unittest.TestCase, OrthogonalRegularizerTestBase):

    def setUp(self):
        self.setUpTestEntity(num_features=10, l2reg=0.001)

class TestOrthogonalRegularizerConfig2(unittest.TestCase, OrthogonalRegularizerTestBase):

    def setUp(self):
        self.setUpTestEntity(num_features=5, l2reg=0.002)

class TestOrthogonalRegularizerConfig3(unittest.TestCase, OrthogonalRegularizerTestBase):

    def setUp(self):
        self.setUpTestEntity(num_features=128, l2reg=0.002)
        
class TestOrthogonalRegularizerConfig4(unittest.TestCase, OrthogonalRegularizerTestBase):

    def setUp(self):
        self.setUpTestEntity(num_features=1024, l2reg=0.0025)