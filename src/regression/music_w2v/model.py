import torch
from torch import nn
from modules import Conv_2d, Conv_emb, Res_2d_mp, Res_2d


class MyModel(nn.Module):
    def __init__(self, ndim=64):
        super(MyModel, self).__init__()

        # Short-chunk CNN
        self.spec_bn = nn.BatchNorm2d(1)
        self.layer1 = Conv_2d(1, ndim, pooling=2)
        self.layer2 = Res_2d_mp(ndim, ndim, pooling=2)
        self.layer3 = Conv_2d(ndim, ndim*2, pooling=2)
        self.layer4 = Res_2d_mp(ndim*2, ndim*2, pooling=2)
        self.layer5 = Res_2d_mp(ndim*2, ndim*2, pooling=2)
        self.layer6 = Res_2d_mp(ndim*2, ndim*2, pooling=(2,3))
        self.layer7 = Conv_2d(ndim*2, ndim*4, pooling=(2,3))
        self.layer8 = Conv_emb(ndim*4, ndim*4)

        # projection
        self.fc1 = nn.Linear(ndim*4, ndim)
        self.bn = nn.BatchNorm1d(ndim)
        self.fc2 = nn.Linear(ndim, 300)

        # others
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)

    def spec_to_embedding(self, spec):
        # input normalization
        out = spec.unsqueeze(1)
        out = self.spec_bn(out)

        # CNN
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = out.squeeze(2)
        out = nn.MaxPool1d(out.size(-1))(out)
        out = out.view(out.size(0), -1)

        # projection
        out = self.fc1(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

    def forward(self, spec):
        logits = self.spec_to_embedding(spec)
        return logits
