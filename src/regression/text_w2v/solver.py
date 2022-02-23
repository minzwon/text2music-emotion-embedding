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

from src.regression.text_w2v.data_loader import MyDataset
from src.regression.text_w2v.model import MyModel


class Solver(LightningModule):
    def __init__(self, config):
        super().__init__()
        # configuration
        self.lr = config.lr
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.mode = config.mode
        self.dataset = config.dataset
        self.mode = config.mode
        self.moods = np.load('./data/text_%s_moods.npy'%self.dataset)

        # model
        self.model = MyModel(ndim=config.ndim)
        self.loss_function = nn.MSELoss()

        # initialize lists
        self.text_logits = []
        self.text_w2v = []
        self.text_binaries = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        return optimizer

    def train_dataloader(self):
        return DataLoader(dataset=MyDataset(split='TRAIN', dataset=self.dataset), 
                          batch_size=self.batch_size, shuffle=True, drop_last=False, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(dataset=MyDataset(split='VALID', dataset=self.dataset),
                          batch_size=self.batch_size, 
                          shuffle=False, drop_last=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(dataset=MyDataset(split='TEST', dataset=self.dataset),
                          batch_size=self.batch_size, 
                          shuffle=False, drop_last=False, num_workers=self.num_workers)

    def training_step(self, batch, batch_idx):
        token, mask, w2v, binary = batch
        logits = self.model.forward(token, mask)
        loss = self.loss_function(logits, w2v)
        self.log('train_loss', loss)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log('avg_loss', avg_loss)

    def validation_step(self, batch, batch_idx):
        token, mask, w2v, binary = batch
        logits = self.model.forward(token, mask)
        loss = self.loss_function(logits, w2v)
        self.log('valid_loss', loss)

        self.text_w2v.append(w2v.detach().cpu())
        self.text_logits.append(logits.detach().cpu())
        self.text_binaries.append(binary.detach().cpu())
        return {'loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss)

        text_logits = torch.cat(self.text_logits, dim=0)
        text_w2v = torch.cat(self.text_w2v, dim=0)
        text_binaries = torch.cat(self.text_binaries, dim=0)

        self.log('monitor', avg_loss)

        self.text_logits = []
        self.text_w2v = []
        self.text_binaries = []
        print(avg_loss)

    def test_step(self, batch, batch_idx):
        token, mask, w2v, binary = batch
        logits = self.model.forward(token, mask)
        loss = self.loss_function(logits, w2v)
        self.log('valid_loss', loss)

        self.text_w2v.append(w2v.detach().cpu())
        self.text_logits.append(logits.detach().cpu())
        self.text_binaries.append(binary.detach().cpu())
        return {'loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('monitor', avg_loss)

        text_logits = torch.cat(self.text_logits, dim=0)
        text_w2v = torch.cat(self.text_w2v, dim=0)
        text_binaries = torch.cat(self.text_binaries, dim=0)

        self.log('monitor', avg_loss)

