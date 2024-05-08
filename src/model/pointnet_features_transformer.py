import torch
import torch.nn as nn
from src.model.tnet import Tnet
from src.model.conv_1d_bn_block import Conv1dBN

class PointNetFeaturesTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transform = Tnet(num_features=3)
        self.feature_transform = Tnet(num_features=64)
        
        self.convolution1 = Conv1dBN(3, 64)
        self.convolution2 = Conv1dBN(64, 128)
        self.convolution3 = Conv1dBN(128, 1024, use_activation=False)
        
    def forward(self, x: torch.Tensor) -> tuple:
        matrix3x3 = self.input_transform(x)
        xb = torch.bmm(torch.transpose(x, 1, 2), matrix3x3).transpose(1, 2)
        xb = self.convolution1(xb)
        matrix64x64 = self.feature_transform(xb)
        xb = torch.bmm(torch.transpose(xb, 1, 2), matrix64x64).transpose(1, 2)
        xb = self.convolution2(xb)
        xb = self.convolution3(xb)
        xb = nn.MaxPool1d(xb.size(-1))(xb)
        output = nn.Flatten(1)(xb)
        return output, matrix3x3, matrix64x64
  