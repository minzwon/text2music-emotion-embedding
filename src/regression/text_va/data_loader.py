import os
import sys
import pickle
import tqdm
import numpy as np
import pandas as pd
import random
from torch.utils import data
from transformers import DistilBertTokenizer


class MyDataset(data.Dataset):
    def __init__(self, split='TRAIN', dataset='isear'):
        self.split = split
        self.dataset = dataset
        self.text_tags = np.load('./data/text_%s_moods.npy' % dataset)

        if split == 'TRAIN':
            # text
            self.text_ix = np.load('./data/text_%s_train_ix.npy' % dataset)
        elif split == 'VALID':
            # text
            self.text_ix = np.load('./data/text_%s_valid_ix.npy' % dataset)
        elif split == 'TEST':
            # text
            self.text_ix = np.load('./data/text_%s_test_ix.npy' % dataset)

        # text
        corpus = np.load('./data/text_%s_corpus.npy' % dataset, allow_pickle=True)
        self.text_vads = np.load('./data/text_%s_vads.npy' % dataset)[:, :2]
        self.text_binaries = np.load('./data/text_%s_binaries.npy' % dataset)
        
        # tokenize
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        encoding = tokenizer(list(corpus), return_tensors='pt', padding=True, truncation=True)
        self.text_tokens = np.concatenate([encoding['input_ids'].unsqueeze(0), encoding['attention_mask'].unsqueeze(0)])

    def __getitem__(self, index):
        ix = self.text_ix[index]
        token = self.text_tokens[0][ix]
        mask = self.text_tokens[1][ix]
        vad = self.text_vads[ix]
        binary = self.text_binaries[ix]

        return token, mask, vad.astype('float32'), binary.astype('float32')

    def __len__(self):
        return len(self.text_ix)


