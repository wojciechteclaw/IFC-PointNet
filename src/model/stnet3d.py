import torch
import torch.nn as nn
import numpy as np
from src.model.conv_1d_bn_block import Conv1dBN
from src.model.dense_linear_bn_layer import DenseLinearBN

class STNet3d(nn.Module):
    """
    Spatial Transformer Network for 3D data. This module applies a series of convolutions
    and fully connected layers to predict a transformation matrix for 3D point clouds.

    Attributes:
        conv1, conv2, conv3 (Conv1dBN): Convolutional layers with batch normalization.
        fc1, fc2 (DenseLinearBN): Fully connected layers with batch normalization.
        fc3 (nn.Linear): Final fully connected layer that outputs the transformation parameters.
        relu (nn.ReLU): ReLU activation function used after each layer except the last linear layer.
    """

    def __init__(self):
        """
        Initializes the Spatial Transformer Network with predefined layer sizes and configurations.
        """
        super(STNet3d, self).__init__()
        self.conv1 = Conv1dBN(3, 64)
        self.conv2 = Conv1dBN(64, 128)
        self.conv3 = Conv1dBN(128, 1024)
        self.fc1 = DenseLinearBN(1024, 512)
        self.fc2 = DenseLinearBN(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the STNet3d. Takes an input batch of point clouds, processes it through
        convolutional and dense layers, and outputs a batch of 3x3 transformation matrices.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_points, 3).

        Returns:
            torch.Tensor: Output tensor of 3x3 transformation matrices for each example in the batch,
                          shaped (batch_size, 3, 3).
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

        # Add identity matrix to make the output close to an affine transformation
        iden = torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32))\
                    .view(1, 9)\
                    .repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x
