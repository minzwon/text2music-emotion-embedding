import torch
from torch import nn
from transformers import DistilBertModel


class MyModel(nn.Module):
    def __init__(self, ndim=64, edim=64):
        super(MyModel, self).__init__()

        # RoBERTa module for text embedding
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased', return_dict=True)
        self.bert.train()
        self.fc1 = nn.Linear(768, edim)
        self.bn = nn.BatchNorm1d(edim)
        self.fc2 = nn.Linear(edim, 300)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def text_to_embedding(self, token, mask):
        out = self.bert(token, attention_mask=mask)['last_hidden_state'][:, 0]
        out = self.fc1(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

    def forward(self, token, mask):
        logits = self.text_to_embedding(token, mask)
        return logits
