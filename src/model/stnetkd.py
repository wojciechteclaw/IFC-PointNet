import torch
import torch.nn as nn
import numpy as np
from src.model.conv_1d_bn_block import Conv1dBN
from src.model.dense_linear_bn_layer import DenseLinearBN

class STNetkd(nn.Module):
    def __init__(self, num_features=64):
        super(STNetkd, self).__init__()
        self.conv1 = Conv1dBN(num_features, 64)
        self.conv2 = Conv1dBN(64, 128)
        self.conv3 = Conv1dBN(128, 1024)
        self.fc1 = DenseLinearBN(1024, 512)
        self.fc2 = DenseLinearBN(512, 256)
        self.fc3 = nn.Linear(256, num_features * num_features)
        self.num_features = num_features

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

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
