import torch
import torch.nn as nn
from src.model.conv_1d_bn_block import Conv1dBN
from src.model.stnet3d import STNet3d
from src.model.stnetkd import STNetkd

class PointNetfeat(nn.Module):
    """
    PointNet feature network module that transforms raw point cloud data into a global feature vector.
    It includes spatial transformer networks (STN) and 1D convolutional blocks with batch normalization.

    Attributes:
        stn (STNet3d): Spatial transformer network for input points.
        conv1, conv2, conv3 (Conv1dBN): 1D convolutional layers with batch normalization.
        global_feat (bool): If True, uses global feature vector for classification.
        feature_transform (bool): If True, applies feature transformation to enhance model robustness.
        fstn (STNetkd or None): Feature space transformer network, activated if feature_transform is True.
    """

    def __init__(self, global_feat: bool = True, feature_transform: bool = False):
        """
        Initializes the PointNet feature module.

        Args:
            global_feat (bool): If True, the network outputs a global feature vector.
            feature_transform (bool): If True, a feature transformer network is included to apply learned transformations to the features.
        """
        super(PointNetfeat, self).__init__()
        self.stn = STNet3d()
        self.conv1 = Conv1dBN(3, 64)
        self.conv2 = Conv1dBN(64, 128)
        self.conv3 = Conv1dBN(128, 1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        self.fstn = STNetkd(num_features=64) if feature_transform else None

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass of the PointNet feature extractor.

        Args:
            x (torch.Tensor): The input tensor of shape (batch size, number of points, 3).

        Returns:
            tuple: Depending on the setting of global_feat, returns either:
                   - A global feature vector, transformation matrix, and an optional feature transformation matrix.
                   - Concatenated global and point features, transformation matrix, and an optional feature transformation matrix.
        """
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = self.conv1(x)

        trans_feat = None
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)

        pointfeat = x
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat
