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

from src.regression.music_w2v.data_loader import MyDataset
from src.regression.music_w2v.model import MyModel
from src.regression.music_w2v.augmentations import get_augmentation_sequence


class Solver(LightningModule):
    def __init__(self, config):
        super().__init__()
        # configuration
        self.lr = config.lr
        self.input_length = config.input_length
        self.num_chunk = config.num_chunk
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.data_path = config.data_path
        self.mode = config.mode
        self.moods = np.load('./data/song_moods.npy')

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
        self.song_w2vs = []
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
        raw, w2v, binary = batch
        with torch.no_grad():
            spec = self.aug(raw)
        logits = self.model.forward(spec)
        loss = self.loss_function(logits, w2v)
        self.log('train_loss', loss)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('avg_loss', avg_loss)

    def validation_step(self, batch, batch_idx):
        raw, w2v, binary = batch
        b, c, t = raw.size()
        with torch.no_grad():
            spec = self.no_aug(raw.view(-1, t))
        logits = self.model.forward(spec)
        logits = logits.view(b, c, -1).mean(dim=1)
        loss = self.loss_function(logits, w2v)
        self.log('valid_loss', loss)

        self.song_w2vs.append(w2v.detach().cpu())
        self.song_logits.append(logits.detach().cpu())
        self.song_binaries.append(binary.detach().cpu())
        return {'loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        song_logits = torch.cat(self.song_logits, dim=0)
        song_w2vs = torch.cat(self.song_w2vs, dim=0)
        song_binaries = torch.cat(self.song_binaries, dim=0)

        print('Overall: %.4f' % avg_loss)
        self.log('monitor', avg_loss)

        self.song_logits = []
        self.song_w2vs = []
        self.song_binaries = []
        print(avg_loss)

    def test_step(self, batch, batch_idx):
        raw, w2v, binary = batch
        b, c, t = raw.size()
        with torch.no_grad():
            spec = self.no_aug(raw.view(-1, t))
        logits = self.model.forward(spec)
        logits = logits.view(b, c, -1).mean(dim=1)
        loss = self.loss_function(logits, w2v)
        self.log('test_loss', loss)

        self.song_w2vs.append(w2v.detach().cpu())
        self.song_logits.append(logits.detach().cpu())
        self.song_binaries.append(binary.detach().cpu())
        return {'loss': loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        print(avg_loss)

        song_logits = torch.cat(self.song_logits, dim=0)
        song_w2vs = torch.cat(self.song_w2vs, dim=0)
        song_binaries = torch.cat(self.song_binaries, dim=0)

        self.log('monitor', avg_loss)

