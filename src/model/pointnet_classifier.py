import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.pointnet_feat import PointNetfeat
from src.model.dense_linear_bn_layer import DenseLinearBN

class PointNetClassifier(nn.Module):
    def __init__(self, num_classes: int = 2, feature_transform: bool = False):
        """
        Initializes the PointNetClassifier model for point cloud classification.
        This model architecture includes feature extraction layers, fully connected layers
        with batch normalization, and dropout for regularization.

        Args:
            num_classes (int): Number of classes for classification.
            feature_transform (bool): Whether to apply feature transformation.

        Attributes:
            feature_transform (bool): Tracks if feature transformation is applied.
            feat (PointNetfeat): Feature extraction module.
            fc1 (DenseLinearBN): First fully connected layer with batch normalization.
            fc2 (DenseLinearBN): Second fully connected layer with batch normalization.
            fc3 (nn.Linear): Final fully connected layer outputting class scores.
            dropout (nn.Dropout): Dropout layer for reducing overfitting.
        """
        super(PointNetClassifier, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = DenseLinearBN(1024, 512)
        self.fc2 = DenseLinearBN(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass of the model which computes the logits for each class, along with
        transformation matrices if feature_transform is enabled.

        Args:
            x (torch.Tensor): Input tensor representing point cloud data.

        Returns:
            tuple: Contains logits after log softmax, feature transformation matrix,
                   and global feature transformation matrix (if feature_transform is True).
        """
        x, trans, trans_feat = self.feat(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x, trans_feat
