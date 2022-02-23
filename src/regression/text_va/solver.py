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

from src.regression.text_va.data_loader import MyDataset
from src.regression.text_va.model import MyModel


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
        self.nrc_vad = pickle.load(open('./data/tag_to_vad.pkl', 'rb'))

        # model
        self.model = MyModel(ndim=config.ndim)
        self.loss_function = nn.MSELoss()

        # initialize lists
        self.text_logits = []
        self.text_vads = []
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
        token, mask, vad, binary = batch
        logits = self.model.forward(token, mask)
        loss = self.loss_function(logits, vad)
        self.log('train_loss', loss)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log('avg_loss', avg_loss)

    def validation_step(self, batch, batch_idx):
        token, mask, vad, binary = batch
        logits = self.model.forward(token, mask)
        loss = self.loss_function(logits, vad)
        self.log('valid_loss', loss)

        self.text_vads.append(vad.detach().cpu())
        self.text_logits.append(logits.detach().cpu())
        self.text_binaries.append(binary.detach().cpu())
        return {'loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss)

        text_logits = torch.cat(self.text_logits, dim=0)
        text_vads = torch.cat(self.text_vads, dim=0)
        text_binaries = torch.cat(self.text_binaries, dim=0)

        rv, ra = metrics.r2_score(text_vads, text_logits, multioutput='raw_values')
        r2 = np.mean([rv, ra])
        print('R2_scores: %.4f, %.4f' % (rv, ra))
        print('Overall: %.4f' % r2)
        self.log('monitor', r2)

        self.text_logits = []
        self.text_vads = []
        self.text_binaries = []
        print(avg_loss)

    def test_step(self, batch, batch_idx):
        token, mask, vad, binary = batch
        logits = self.model.forward(token, mask)
        loss = self.loss_function(logits, vad)
        self.log('valid_loss', loss)

        self.text_vads.append(vad.detach().cpu())
        self.text_logits.append(logits.detach().cpu())
        self.text_binaries.append(binary.detach().cpu())
        return {'loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('monitor', avg_loss)

        text_logits = torch.cat(self.text_logits, dim=0)
        text_vads = torch.cat(self.text_vads, dim=0)
        text_binaries = torch.cat(self.text_binaries, dim=0)

        r2 = self.get_scores(text_logits, text_vads, text_binaries)
        
        self.log('monitor', r2)

        # save items
        np.save(open('./embs/test_text_%s_logits.npy'%self.dataset, 'wb'), text_logits)
        np.save(open('./embs/test_text_%s_vads.npy'%self.dataset, 'wb'), text_vads)
        np.save(open('./embs/test_text_%s_binaries.npy'%self.dataset, 'wb'), text_binaries)

    # evaluation metrics
    def get_scores(self, logits, vads, binaries):
        # r2 score
        rv, ra = metrics.r2_score(vads, logits, multioutput='raw_values')
        r2 = np.mean([rv, ra])
        print('R2_scores: %.4f, %.4f' % (rv, ra))
        print('Overall: %.4f' % r2)

        # precision @k
        moods = self.moods
        mood_vads = np.array([self.nrc_vad[mood] for mood in moods])[:, :2]
        sim = metrics.pairwise.cosine_similarity(mood_vads, vads)
        k_indice = []
        for i in range(len(sim)):
            k_indice.append(sim[i].argsort()[::-1][:10])
        k_indice = np.array(k_indice)

        binaries[binaries>0] = 1

        for k in [1, 5, 10]:
            ps = []
            for i, mood in enumerate(moods):
                p = metrics.precision_score(np.ones(k), binaries[k_indice[i]][:, i][:k])
                ps.append(p)
            precision = np.mean(ps)
            print('Precision @ %d: %.4f' % (k, precision))

        k = 10
        for i, mood in enumerate(moods):
            p = metrics.precision_score(np.ones(k), binaries[k_indice[i]][:, i][:k])
            print('P@10---%s: %.4f' % (mood, p))

        roc_auc = metrics.roc_auc_score(binaries, sim.T, average='macro')
        pr_auc = metrics.average_precision_score(binaries, sim.T, average='macro')
        print('ROC-AUC: %.4f' % roc_auc)
        print('PR-AUC: %.4f' % pr_auc)

        self.get_tag_wise_score(logits, vads, binaries, moods)

        return r2

    def get_tag_wise_score(self, logits, vads, binaries, moods):
        mses = []
        for i, mood in enumerate(moods):
            indice = np.where(binaries[:,i]>0)
            mse = self.loss_function(logits[indice].clone().detach(), vads[indice].clone().detach())
            mses.append(mse)
            print('MSE---%s: %.4f' % (mood, mse))
        print('Overall MSE: %.4f' % np.mean(mses))
