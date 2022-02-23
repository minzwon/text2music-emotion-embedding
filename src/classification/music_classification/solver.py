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

from src.classification.music_classification.data_loader import MyDataset
from src.classification.music_classification.model import MyModel
from src.classification.music_classification.augmentations import get_augmentation_sequence


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)      # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))       # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


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

        # model
        self.aug, self.no_aug = get_augmentation_sequence(config)
        self.model = MyModel(ndim=config.ndim, edim=config.edim, cdim=len(self.moods))
        self.load_pretrained('./data/mtat.ckpt')
        self.loss_function = FocalLoss(gamma=2)

        # initialize lists
        self.song_embs = []
        self.song_logits = []
        self.song_binaries = []

    def load_pretrained(self, load_path):
        # get pretrained state_dict
        S = torch.load(load_path)['state_dict']
        NS = {key[6:]: S[key] for key in S.keys() if ((key[:5]=='model') and (key[:13]!='model.song_fc'))}

        # update current model
        S = self.model.state_dict()
        for key in NS.keys():
            S[key] = NS[key]
        self.model.load_state_dict(S)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        return optimizer

    def train_dataloader(self):
        return DataLoader(dataset=MyDataset(self.data_path, split='TRAIN', 
                                            input_length=self.input_length, num_chunk=self.num_chunk), 
                          batch_size=self.batch_size, shuffle=True, drop_last=False, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(dataset=MyDataset(self.data_path, split='VALID',
                                            input_length=self.input_length, num_chunk=self.num_chunk),
                          batch_size=self.batch_size//self.num_chunk, 
                          shuffle=False, drop_last=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(dataset=MyDataset(self.data_path, split='TEST',
                                            input_length=self.input_length, num_chunk=self.num_chunk),
                          batch_size=self.batch_size//self.num_chunk, 
                          shuffle=False, drop_last=False, num_workers=self.num_workers)

    def training_step(self, batch, batch_idx):
        raw, binary = batch
        spec = self.aug(raw)
        emb, logits = self.model.forward(spec)
        loss = self.loss_function(logits, binary.argmax(dim=1))
        self.log('train_loss', loss)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log('avg_loss', avg_loss)

    def validation_step(self, batch, batch_idx):
        raw, binary = batch
        b, c, t = raw.size()
        with torch.no_grad():
            spec = self.no_aug(raw.view(-1, t))
        emb, logits = self.model.forward(spec)
        emb = emb.view(b, c, -1).mean(dim=1)
        logits = logits.view(b, c, -1).mean(dim=1)

        self.song_embs.append(emb.detach().cpu())
        self.song_logits.append(logits.detach().cpu())
        self.song_binaries.append(binary.detach().cpu())

    def validation_epoch_end(self, outputs):
        song_embs = torch.cat(self.song_embs, dim=0)
        song_logits = torch.cat(self.song_logits, dim=0)
        song_binaries = torch.cat(self.song_binaries, dim=0)

        # ignore unit test
        if song_embs.size(0) <= self.batch_size*2:
            overall = torch.tensor(0)
        else:
            overall = self.get_scores(song_embs, song_logits, song_binaries)

        # logs
        self.log('monitor', overall)

        self.song_embs = []
        self.song_logits = []
        self.song_binaries = []

    def test_step(self, batch, batch_idx):
        raw, binary = batch
        b, c, t = raw.size()
        with torch.no_grad():
            spec = self.no_aug(raw.view(-1, t))
        emb, logits = self.model.forward(spec)
        emb = emb.view(b, c, -1).mean(dim=1)
        logits = logits.view(b, c, -1).mean(dim=1)

        self.song_embs.append(emb.detach().cpu())
        self.song_logits.append(logits.detach().cpu())
        self.song_binaries.append(binary.detach().cpu())

    def test_epoch_end(self, outputs):
        song_embs = torch.cat(self.song_embs, dim=0)
        song_logits = torch.cat(self.song_logits, dim=0)
        song_binaries = torch.cat(self.song_binaries, dim=0)

        overall = self.get_scores(song_embs, song_logits, song_binaries)

        # logs
        self.log('monitor', overall)

    # evaluation metrics
    def get_scores(self, embs, logits, binaries):
        accuracy = metrics.accuracy_score(np.argmax(binaries, axis=1), np.argmax(logits, axis=1))
        roc_auc = metrics.roc_auc_score(binaries, logits, average='macro')
        pr_auc = metrics.average_precision_score(binaries, logits, average='macro')
        f1 = metrics.f1_score(np.argmax(binaries, axis=1), np.argmax(logits, axis=1), average='macro')
        cm = metrics.confusion_matrix(np.argmax(binaries, axis=1), np.argmax(logits, axis=1))

        # print
        print('accuracy: %.4f' % accuracy)
        print('roc_auc: %.4f' % roc_auc)
        print('pr_auc: %.4f' % pr_auc)
        print('f1-score: %.4f' % f1)
        print('confusion matrix')
        print(cm)
        print(self.moods)

        return torch.tensor(accuracy)
