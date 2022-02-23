import torch
from torch import nn
from src.metric_learning.modules import Conv_2d, Conv_emb, Res_2d_mp
from transformers import DistilBertModel


class MyModel(nn.Module):
    def __init__(self, ndim=64, edim=64, cdim=1):
        super(MyModel, self).__init__()

        # song embedding
        self.spec_bn = nn.BatchNorm2d(1)
        self.layer1 = Conv_2d(1, ndim, pooling=2)
        self.layer2 = Res_2d_mp(ndim, ndim, pooling=2)
        self.layer3 = Conv_2d(ndim, ndim*2, pooling=2)
        self.layer4 = Res_2d_mp(ndim*2, ndim*2, pooling=2)
        self.layer5 = Res_2d_mp(ndim*2, ndim*2, pooling=2)
        self.layer6 = Res_2d_mp(ndim*2, ndim*2, pooling=(2,3))
        self.layer7 = Conv_2d(ndim*2, ndim*4, pooling=(2,3))
        self.layer8 = Conv_emb(ndim*4, ndim*4)
        self.song_fc1 = nn.Linear(ndim*4, ndim*2)
        self.song_bn = nn.BatchNorm1d(ndim*2)
        self.song_fc2 = nn.Linear(ndim*2, edim)

        # text embedding
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased', return_dict=True)
        self.bert.train()
        self.text_fc1 = nn.Linear(768, ndim*2)
        self.text_bn = nn.BatchNorm1d(ndim*2)
        self.text_fc2 = nn.Linear(ndim*2, edim)

        # tag embedding
        self.tag_fc1 = nn.Linear(300, ndim*2)
        self.tag_bn = nn.BatchNorm1d(ndim*2)
        self.tag_fc2 = nn.Linear(ndim*2, edim)

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
        out = self.song_fc1(out)
        out = self.song_bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.song_fc2(out)
        return out

    def text_to_embedding(self, token, mask):
        out = self.bert(token, attention_mask=mask)['last_hidden_state'][:, 0]
        out = self.text_fc1(out)
        out = self.text_bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.text_fc2(out)
        return out

    def tag_to_embedding(self, tag):
        out = self.tag_fc1(tag)
        out = self.tag_bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.tag_fc2(out)
        return out

    def forward(self, tag, spec, token, mask):
        tag_emb = self.tag_to_embedding(tag)
        song_emb = self.spec_to_embedding(spec)
        text_emb = self.text_to_embedding(token, mask)
        return tag_emb, song_emb, text_emb
