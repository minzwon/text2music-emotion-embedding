import os
import random
import torch
import time
import pickle
import tqdm
import numpy as np
import pandas as pd
from sklearn import metrics
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from pytorch_lightning.core.lightning import LightningModule

from data_loader import MyDataset
from model import MyModel


class Solver(LightningModule):
    def __init__(self, config):
        super().__init__()
        # configuration
        self.lr = config.lr
        self.data_path = config.data_path
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.mode = config.mode
        self.dataset = config.dataset
        self.moods = np.load('./data/text_%s_moods.npy'%self.dataset)

        # model
        self.model = MyModel(ndim=config.ndim, edim=config.edim, cdim=len(self.moods))
        self.loss_function = nn.CrossEntropyLoss()

        # initialize lists
        self.text_embs = []
        self.text_logits = []
        self.text_binaries = []

    def load_pretrained(self, load_path):
        # get pretrained state_dict
        S = torch.load(load_path)['state_dict']
        NS = {}
        for key in S.keys():
            if key[:5] == 'model':
                NS[key[6:]] = S[key]
        # update current model
        S = self.model.state_dict()
        for key in S.keys():
            S[key] = NS[key]
        self.model.load_state_dict(S)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        return optimizer

    def train_dataloader(self):
        return DataLoader(dataset=MyDataset(self.data_path, self.dataset, split='TRAIN'), 
                          batch_size=self.batch_size, shuffle=True, drop_last=False, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(dataset=MyDataset(self.data_path, self.dataset, split='VALID'),
                          batch_size=self.batch_size, 
                          shuffle=False, drop_last=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(dataset=MyDataset(self.data_path, self.dataset, split='TEST'),
                          batch_size=self.batch_size, 
                          shuffle=False, drop_last=False, num_workers=self.num_workers)

    def training_step(self, batch, batch_idx):
        token, mask, binary = batch
        emb, logits = self.model.forward(token, mask)
        loss = self.loss_function(logits, binary.argmax(dim=1))
        self.log('train_loss', loss)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log('avg_loss', avg_loss)

    def validation_step(self, batch, batch_idx):
        token, mask, binary = batch
        emb, logits = self.model.forward(token, mask)
        loss = self.loss_function(logits, binary.argmax(dim=1))
        self.log('valid_loss', loss)

        self.text_embs.append(emb.detach().cpu())
        self.text_logits.append(logits.detach().cpu())
        self.text_binaries.append(binary.detach().cpu())
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        text_embs = torch.cat(self.text_embs, dim=0)
        text_logits = torch.cat(self.text_logits, dim=0)
        text_binaries = torch.cat(self.text_binaries, dim=0)

        # ignore unit test
        if text_embs.size(0) <= self.batch_size*2:
            overall = torch.tensor(0)
        else:
            overall = self.get_scores(text_embs, text_logits, text_binaries)

        # logs
        self.log('monitor', overall)

        self.text_embs = []
        self.text_logits = []
        self.text_binaries = []

    def test_step(self, batch, batch_idx):
        token, mask, binary = batch
        emb, logits = self.model.forward(token, mask)
        loss = self.loss_function(logits, binary.argmax(dim=1))
        self.log('test_loss', loss)

        self.text_embs.append(emb.detach().cpu())
        self.text_logits.append(logits.detach().cpu())
        self.text_binaries.append(binary.detach().cpu())
        return {'loss': loss}

    def test_epoch_end(self, outputs):
        text_embs = torch.cat(self.text_embs, dim=0)
        text_logits = torch.cat(self.text_logits, dim=0)
        text_binaries = torch.cat(self.text_binaries, dim=0)

        # ignore unit test
        if text_embs.size(0) <= self.batch_size*2:
            overall = torch.tensor(0)
        else:
            overall = self.get_scores(text_embs, text_logits, text_binaries)

        # logs
        self.log('monitor', overall)

    # evaluation metrics
    def get_scores(self, text_embs, text_logits, text_binaries):
        accuracy = metrics.accuracy_score(np.argmax(text_logits, axis=1), np.argmax(text_binaries, axis=1))
        roc_auc = metrics.roc_auc_score(text_binaries, text_logits, average='macro')
        pr_auc = metrics.average_precision_score(text_binaries, text_logits, average='macro')
        f1 = metrics.f1_score(np.argmax(text_binaries, axis=1), np.argmax(text_logits, axis=1), average='macro')

        # print
        print('text_accuracy: %.4f' % accuracy)
        print('text_roc_auc: %.4f' % roc_auc)
        print('text_pr_auc: %.4f' % pr_auc)
        print('text_f1: %.4f' % f1)

        return torch.tensor(accuracy)
