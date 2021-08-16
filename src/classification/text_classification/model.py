import torch
from torch import nn
from transformers import DistilBertModel


class MyModel(nn.Module):
    def __init__(self, ndim=64, edim=64, cdim=1):
        super(MyModel, self).__init__()

        # DistilBERT module for text embedding
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased', return_dict=True)
        self.bert.train()
        self.fc1 = nn.Linear(768, edim)
        self.bn = nn.BatchNorm1d(edim)
        self.fc2 = nn.Linear(edim, cdim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def text_to_embedding(self, token, mask):
        out = self.bert(token, attention_mask=mask)['last_hidden_state'][:, 0]
        out = self.fc1(out)
        out = self.bn(out)
        emb = self.relu(out)
        out = self.fc2(emb)
        return emb, out

    def forward(self, token, mask):
        emb, logits = self.text_to_embedding(token, mask)
        return emb, logits
