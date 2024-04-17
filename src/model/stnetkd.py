import torch
import torch.nn as nn
import numpy as np
from src.model.conv_1d_bn_block import Conv1dBN
from src.model.dense_linear_bn_layer import DenseLinearBN

class STNetkd(nn.Module):
    """
    Spatial Transformer Network for feature space (k-dimensional). This module predicts a transformation matrix
    for the k-dimensional features of the input data, enhancing model invariance to geometric perturbations.

    Attributes:
        conv1, conv2, conv3 (Conv1dBN): Convolutional layers with batch normalization.
        fc1, fc2 (DenseLinearBN): Fully connected layers with batch normalization.
        fc3 (nn.Linear): Fully connected layer that outputs the parameters of the kxk feature transformation matrix.
        num_features (int): The number of features (k) for the transformation matrix.
    """

    def __init__(self, num_features: int = 64):
        """
        Initializes the Spatial Transformer Network for k-dimensional features.

        Args:
            num_features (int): Number of features in the feature space transformation matrix.
        """
        super(STNetkd, self).__init__()
        self.conv1 = Conv1dBN(num_features, 64)
        self.conv2 = Conv1dBN(64, 128)
        self.conv3 = Conv1dBN(128, 1024)
        self.fc1 = DenseLinearBN(1024, 512)
        self.fc2 = DenseLinearBN(512, 256)
        self.fc3 = nn.Linear(256, num_features * num_features)
        self.num_features = num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the STNetkd. Processes the input through convolutions and dense layers to predict a
        transformation matrix for the feature space of the input data.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, num_points, num_features).

        Returns:
            torch.Tensor: A batch of transformation matrices, each of shape (num_features, num_features),
                          for transforming the feature space of the input data.
        """
        batchsize = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        # Initialize with the identity matrix to encourage the network to learn only the necessary transformations
        iden = torch.from_numpy(
            np.eye(self.num_features)
            .flatten()
            .astype(np.float32))\
            .view(1, self.num_features * self.num_features)\
            .repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.num_features, self.num_features)
        return x
