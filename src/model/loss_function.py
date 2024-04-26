import torch
import torch.nn.functional as F

from src.model.feature_transform_regulizer import feature_transform_reguliarzer


class LossFunction(torch.nn.Module):
	"""
    Custom loss function that combines negative log-likelihood loss with a matrix
    difference loss to regularize the feature transformations in a neural network model.

    This loss function is particularly useful for models dealing with structured data where
    preserving the geometric properties through transformations is beneficial, e.g., in
    point cloud processing.

    Attributes:
        mat_diff_loss_scale (float): A scaling factor for the matrix difference loss to
                                     control its impact relative to the main loss component.
    """
	
	def __init__(self, mat_diff_loss_scale=0.001):
		"""
        Initializes the LossFunction class with the specified scale for the matrix difference loss.

        Args:
            mat_diff_loss_scale (float): The scaling factor for the matrix difference loss component.
                                         Default value is 0.001.
        """
		super(LossFunction, self).__init__()
		self.mat_diff_loss_scale = mat_diff_loss_scale
	
	def forward(self, pred, target, trans_feat, weight=None):
		"""
        Forward pass for calculating the loss.

        Args:
            pred (torch.Tensor): The predictions from the model (logits).
            target (torch.Tensor): The true labels.
            trans_feat (torch.Tensor): The transformation features from the model.
            weight (torch.Tensor): Class weights for handling imbalanced data.

        Returns:
            torch.Tensor: The computed total loss combining nll_loss and the scaled matrix difference loss.
        """
		loss = F.nll_loss(pred, target, weight=weight)
		mat_diff_loss = feature_transform_reguliarzer(trans_feat)
		total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
		return total_loss
