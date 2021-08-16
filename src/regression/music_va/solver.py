import os
import random
import torch
import torchaudio
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
from augmentations import get_augmentation_sequence


class Solver(LightningModule):
    def __init__(self, config):
        super().__init__()
        # configuration
        self.lr = config.lr
        self.data_path = config.data_path
        self.input_length = config.input_length
        self.num_chunk = config.num_chunk
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.mode = config.mode
        self.moods = np.load('./data/song_moods.npy')
        self.nrc_vad = pickle.load(open('./data/tag_to_vad.pkl', 'rb'))

        # model
        self.aug = torchaudio.transforms.MelSpectrogram(sample_rate=config.sample_rate,
                                                         n_fft=config.n_fft,
                                                         f_min=0.0,
                                                         f_max=8000.0,
                                                         n_mels=config.n_bins)
        self.no_aug = torchaudio.transforms.MelSpectrogram(sample_rate=config.sample_rate,
                                                         n_fft=config.n_fft,
                                                         f_min=0.0,
                                                         f_max=8000.0,
                                                         n_mels=config.n_bins)
        self.model = MyModel(ndim=config.ndim)
        self.loss_function = nn.MSELoss()

        # initialize lists
        self.song_logits = []
        self.song_vads = []
        self.song_binaries = []

    def load_pretrained(self, load_path):
        S = torch.load(load_path)['state_dict']
        NS = {key[6:]: S[key] for key in S.keys() if ((key[:5] == 'model') and (key[:13]!='model.song_fc'))}
        S = self.model.state_dict()
        for key in NS.keys():
            if key in S.keys():
                S[key] = NS[key]
        self.model.load_state_dict(S)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        return optimizer

    def train_dataloader(self):
        return DataLoader(dataset=MyDataset(data_path=self.data_path, split='TRAIN', 
                                            input_length=self.input_length, num_chunk=self.num_chunk), 
                          batch_size=self.batch_size, shuffle=True, drop_last=False, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(dataset=MyDataset(data_path=self.data_path, split='VALID',
                                            input_length=self.input_length, num_chunk=self.num_chunk),
                          batch_size=self.batch_size//self.num_chunk, 
                          shuffle=False, drop_last=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(dataset=MyDataset(data_path=self.data_path, split='TEST',
                                            input_length=self.input_length, num_chunk=self.num_chunk),
                          batch_size=self.batch_size//self.num_chunk, 
                          shuffle=False, drop_last=False, num_workers=self.num_workers)

    def training_step(self, batch, batch_idx):
        raw, vad, binary = batch
        with torch.no_grad():
            spec = self.aug(raw)
        logits = self.model.forward(spec)
        loss = self.loss_function(logits, vad)
        self.log('train_loss', loss)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('avg_loss', avg_loss)

    def validation_step(self, batch, batch_idx):
        raw, vad, binary = batch
        b, c, t = raw.size()
        with torch.no_grad():
            spec = self.no_aug(raw.view(-1, t))
        logits = self.model.forward(spec)
        logits = logits.view(b, c, -1).mean(dim=1)
        loss = self.loss_function(logits, vad)
        self.log('valid_loss', loss)

        self.song_vads.append(vad.detach().cpu())
        self.song_logits.append(logits.detach().cpu())
        self.song_binaries.append(binary.detach().cpu())
        return {'loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        song_logits = torch.cat(self.song_logits, dim=0)
        song_vads = torch.cat(self.song_vads, dim=0)
        song_binaries = torch.cat(self.song_binaries, dim=0)

        rv, ra = metrics.r2_score(song_vads, song_logits, multioutput='raw_values')
        r2 = np.mean([rv, ra])
        print('R2_scores: %.4f, %.4f' % (rv, ra))
        print('Overall: %.4f' % r2)
        self.log('monitor', r2)

        self.song_logits = []
        self.song_vads = []
        self.song_binaries = []
        print(avg_loss)

    def test_step(self, batch, batch_idx):
        raw, vad, binary = batch
        b, c, t = raw.size()
        with torch.no_grad():
            spec = self.no_aug(raw.view(-1, t))
        logits = self.model.forward(spec)
        logits = logits.view(b, c, -1).mean(dim=1)
        loss = self.loss_function(logits, vad)
        self.log('test_loss', loss)

        self.song_vads.append(vad.detach().cpu())
        self.song_logits.append(logits.detach().cpu())
        self.song_binaries.append(binary.detach().cpu())
        return {'loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        print(avg_loss)

        song_logits = torch.cat(self.song_logits, dim=0)
        song_vads = torch.cat(self.song_vads, dim=0)
        song_binaries = torch.cat(self.song_binaries, dim=0)

        # get metrics
        r2 = self.get_scores(song_logits, song_vads, song_binaries)

        self.log('monitor', r2)


    # evaluation metrics
    def get_scores(self, logits, vads, binaries):
        # r2 score
        rv, ra = metrics.r2_score(vads, logits, multioutput='raw_values')
        r2 = np.mean([rv, ra])
        print('R2_scores: %.4f, %.4f' % (rv, ra))
        print('Overall: %.4f' % r2)

        # precision @k
        moods = self.moods
        if self.song_dataset == 'msd':
            binaries = np.delete(binaries, 8, 1)
            moods = np.delete(self.moods, 8, 0)
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

        # roc-auc and pr-auc
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
