import torch
import torch.nn as nn
from src.model.conv_1d_bn_block import Conv1dBN
from src.model.stnet3d import STNet3d
from src.model.stnetkd import STNetkd

class PointNetfeat(nn.Module):
    """Feature transformer - converts XYZ into features vector (Nx1024)"""
    def __init__(self, global_feat=True, feature_transform=False):
        super(PointNetfeat, self).__init__()
        self.stn = STNet3d()
        self.conv1 = Conv1dBN(3, 64)
        self.conv2 = Conv1dBN(64, 128)
        self.conv3 = Conv1dBN(128, 1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        self.fstn = None
        if self.feature_transform:
            self.fstn = STNetkd(num_features=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = self.conv1(x)

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

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
