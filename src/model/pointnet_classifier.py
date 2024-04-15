import torch.nn as nn
import torch.nn.functional as F
from src.model.pointnet_feat import PointNetfeat
from src.model.dense_linear_bn_layer import DenseLinearBN

class PointNetClassifier(nn.Module):
    def __init__(self, num_classes=2, feature_transform=False):
        super(PointNetClassifier, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = DenseLinearBN(1024, 512)
        self.fc2 = DenseLinearBN(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.3)    # eliminates 30% random features

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat
