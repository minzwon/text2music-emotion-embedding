import os
import random
import torch
import time
import pickle
import tqdm
import torchaudio
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


class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.relu = nn.ReLU()

    def forward(self, anchor, positive, negative, size_average=True):
        cosine_positive = nn.CosineSimilarity(dim=-1)(anchor, positive)
        cosine_negative = nn.CosineSimilarity(dim=-1)(anchor, negative)
        losses = self.relu(self.margin - cosine_positive + cosine_negative)
        return losses.mean()


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
        self.is_weighted = config.is_weighted
        self.num_branches = config.num_branches
        self.mode = config.mode
        self.dataset = config.dataset
        self.len_song_dataset = 344
        self.song_moods = ['angry', 'exciting', 'funny', 'happy', 'sad', 'scary', 'tender']

        if self.dataset == 'isear':
            self.len_text_dataset = 1150
            self.text_moods = ['anger', 'disgust', 'fear', 'guilt', 'joy', 'sadness', 'shame']
        elif self.dataset == 'alm':
            self.len_text_dataset = 167
            self.text_moods = ['anger', 'fearful', 'happy', 'sad', 'surprised']

        approx_dict_manual = {'anger': ['angry'],
                      'disgust': ['angry', 'scary'],
                      'fear': ['scary'],
                      'fearful': ['scary'],
                      'guilt': ['sad', 'angry'],
                      'happy': ['happy', 'exciting', 'funny'],
                      'joy': ['happy', 'exciting', 'funny'],
                      'sad': ['sad'],
                      'sadness': ['sad'],
                      'shame': ['angry', 'sad'],
                      'surprised': ['exciting']}

        approx_dict_vad = {'anger': ['angry'],
                    'disgust': ['angry'],
                    'fear': ['angry'],
                    'fearful': ['sad'],
                    'guilt': ['sad'],
                    'happy': ['happy'],
                    'joy': ['exciting'],
                    'sad': ['sad'],
                    'sadness': ['sad'],
                    'shame': ['angry'],
                    'surprised': ['exciting']}

        approx_dict_w2v = {'anger': ['angry'],
                    'disgust': ['angry'],
                    'fear': ['angry'],
                    'fearful': ['scary'],
                    'guilt': ['angry'],
                    'happy': ['happy'],
                    'joy': ['tender'],
                    'sad': ['sad'],
                    'sadness': ['tender'],
                    'shame': ['sad'],
                    'surprised': ['happy']}
        self.approx_dict = approx_dict_w2v

        # model
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=16000,
                                 n_fft=512,
                                 f_min=0.0,
                                 f_max=8000.0,
                                 n_mels=128)
        self.model = MyModel(ndim=config.ndim, edim=config.edim)
        self.loss_function = TripletLoss(config.margin)

        # initialize lists
        self.song_tag_embs = []
        self.song_tag_binaries = []
        self.text_tag_embs = []
        self.text_tag_binaries = []
        self.song_embs = []
        self.song_binaries = []
        self.text_embs = []
        self.text_binaries = []

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
        return DataLoader(dataset=MyDataset(self.data_path, self.dataset, split='TRAIN',
                            input_length=self.input_length, num_chunk=self.num_chunk), 
                  batch_size=self.batch_size//2, shuffle=True, drop_last=False, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(dataset=MyDataset(self.data_path, self.dataset, split='VALID',
                            input_length=self.input_length, num_chunk=self.num_chunk),
                  batch_size=self.batch_size//self.num_chunk, 
                  shuffle=False, drop_last=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(dataset=MyDataset(self.data_path, self.dataset, split='TEST',
                            input_length=self.input_length, num_chunk=self.num_chunk),
                  batch_size=self.batch_size//self.num_chunk, 
                  shuffle=False, drop_last=False, num_workers=self.num_workers)

    def training_step(self, batch, batch_idx):
        # load items
        ttt_tag, ttt_token, ttt_mask, ttt_binary,\
            tts_tag, tts_song, tts_binary,\
            common_token, common_mask, common_song, common_binary = batch
        batch_size = len(ttt_tag)

        # concatenate
        tag = torch.cat([ttt_tag, tts_tag])
        token = torch.cat([ttt_token, common_token])
        mask = torch.cat([ttt_mask, common_mask])
        song = torch.cat([tts_song, common_song])

        # forward
        spec = self.spec(song)
        tag_emb, song_emb, text_emb = self.model.forward(tag, spec, token, mask)

        # split back
        ttt_tag_emb = tag_emb[:batch_size]
        tts_tag_emb = tag_emb[batch_size:]
        ttt_text_emb = text_emb[:batch_size]
        common_text_emb = text_emb[batch_size:]
        tts_song_emb = song_emb[:batch_size]
        common_song_emb = song_emb[batch_size:]

        # triplet sampling
        ttt_anchor, ttt_pos, ttt_neg = self.triplet_sampling(ttt_tag_emb, ttt_text_emb, ttt_binary)
        tts_anchor, tts_pos, tts_neg = self.triplet_sampling(tts_tag_emb, tts_song_emb, tts_binary)
        common_anchor, common_pos, common_neg = self.triplet_sampling(common_text_emb, common_song_emb, common_binary, is_weighted=True)

        # loss functions
        ttt_loss = self.loss_function(ttt_anchor, ttt_pos, ttt_neg)
        tts_loss = self.loss_function(tts_anchor, tts_pos, tts_neg)
        common_loss = self.loss_function(common_anchor, common_pos, common_neg)

        if self.num_branches == 2:
            loss = common_loss
        elif self.num_branches == 3:
            a, b, c = 0.3, 0.3, 0.4
            loss = (a * ttt_loss) + (b * tts_loss) + (c * common_loss)
        self.log('train_loss', loss)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log('avg_tr_loss', avg_loss)

    def validation_step(self, batch, batch_idx):
        song_tag, song_tag_binary, text_tag, text_tag_binary, song, song_binary, token, mask, text_binary = batch
        b, c, t = song.size()
        with torch.no_grad():
            spec = self.spec(song.view(-1, t))
            song_tag_emb = self.model.tag_to_embedding(song_tag)
            text_tag_emb = self.model.tag_to_embedding(text_tag)
            song_emb = self.model.spec_to_embedding(spec)
            text_emb = self.model.text_to_embedding(token, mask)
        song_emb = song_emb.view(b, c, -1).mean(dim=1)

        self.song_tag_embs.append(song_tag_emb.detach().cpu())
        self.song_tag_binaries.append(song_tag_binary.detach().cpu())
        self.text_tag_embs.append(text_tag_emb.detach().cpu())
        self.text_tag_binaries.append(text_tag_binary.detach().cpu())
        self.song_embs.append(song_emb.detach().cpu())
        self.song_binaries.append(song_binary.detach().cpu())
        self.text_embs.append(text_emb.detach().cpu())
        self.text_binaries.append(text_binary.detach().cpu())

    def validation_epoch_end(self, outputs):
        song_tag_embs = torch.cat(self.song_tag_embs, dim=0)[:len(self.song_moods)]
        song_tag_binaries = torch.cat(self.song_tag_binaries, dim=0)[:len(self.song_moods)]
        text_tag_embs = torch.cat(self.text_tag_embs, dim=0)[:len(self.text_moods)]
        text_tag_binaries = torch.cat(self.text_tag_binaries, dim=0)[:len(self.text_moods)]
        song_embs = torch.cat(self.song_embs, dim=0)[:self.len_song_dataset]
        song_binaries = torch.cat(self.song_binaries, dim=0)[:self.len_song_dataset]
        text_embs = torch.cat(self.text_embs, dim=0)[:self.len_text_dataset]
        text_binaries = torch.cat(self.text_binaries, dim=0)[:self.len_text_dataset]

        # ignore unit test
        if song_embs.size(0) <= self.batch_size*2:
            overall = torch.tensor(0)
        else:
            overall = self.get_scores(song_tag_embs, text_tag_embs, song_embs, text_embs, song_tag_binaries, text_tag_binaries, song_binaries, text_binaries)

        # logs
        self.log('monitor', overall)

        self.song_tag_embs = []
        self.song_tag_binaries = []
        self.text_tag_embs = []
        self.text_tag_binaries = []
        self.song_embs = []
        self.song_binaries = []
        self.text_embs = []
        self.text_binaries = []

    def test_step(self, batch, batch_idx):
        song_tag, song_tag_binary, text_tag, text_tag_binary, song, song_binary, token, mask, text_binary = batch
        b, c, t = song.size()
        with torch.no_grad():
            spec = self.spec(song.view(-1, t))
            song_tag_emb = self.model.tag_to_embedding(song_tag)
            text_tag_emb = self.model.tag_to_embedding(text_tag)
            song_emb = self.model.spec_to_embedding(spec)
            text_emb = self.model.text_to_embedding(token, mask)
        song_emb = song_emb.view(b, c, -1).mean(dim=1)

        self.song_tag_embs.append(song_tag_emb.detach().cpu())
        self.song_tag_binaries.append(song_tag_binary.detach().cpu())
        self.text_tag_embs.append(text_tag_emb.detach().cpu())
        self.text_tag_binaries.append(text_tag_binary.detach().cpu())
        self.song_embs.append(song_emb.detach().cpu())
        self.song_binaries.append(song_binary.detach().cpu())
        self.text_embs.append(text_emb.detach().cpu())
        self.text_binaries.append(text_binary.detach().cpu())

    def test_epoch_end(self, outputs):
        song_tag_embs = torch.cat(self.song_tag_embs, dim=0)[:len(self.song_moods)]
        song_tag_binaries = torch.cat(self.song_tag_binaries, dim=0)[:len(self.song_moods)]
        text_tag_embs = torch.cat(self.text_tag_embs, dim=0)[:len(self.text_moods)]
        text_tag_binaries = torch.cat(self.text_tag_binaries, dim=0)[:len(self.text_moods)]
        song_embs = torch.cat(self.song_embs, dim=0)[:self.len_song_dataset]
        song_binaries = torch.cat(self.song_binaries, dim=0)[:self.len_song_dataset]
        text_embs = torch.cat(self.text_embs, dim=0)[:self.len_text_dataset]
        text_binaries = torch.cat(self.text_binaries, dim=0)[:self.len_text_dataset]

        # ignore unit test
        if song_embs.size(0) <= self.batch_size*2:
            overall = torch.tensor(0)
        else:
            overall = self.get_scores(song_tag_embs, text_tag_embs, song_embs, text_embs, song_tag_binaries, text_tag_binaries, song_binaries, text_binaries)

    # evaluation metrics
    def get_scores(self, song_tag_embs, text_tag_embs, song_embs, text_embs, song_tag_binaries, text_tag_binaries, song_binaries, text_binaries):
        # each modality metrics
        t2s_sim = self.get_similarity(song_tag_embs, song_embs)
        t2t_sim = self.get_similarity(text_tag_embs, text_embs)
        song_roc_auc = metrics.roc_auc_score(song_binaries, t2s_sim.T, average='macro')
        song_pr_auc = metrics.average_precision_score(song_binaries, t2s_sim.T, average='macro')
        text_roc_auc = metrics.roc_auc_score(text_binaries, t2t_sim.T, average='macro')
        text_pr_auc = metrics.average_precision_score(text_binaries, t2t_sim.T, average='macro')

        # print
        print('song_roc_auc: %.4f' % song_roc_auc)
        print('song_pr_auc: %.4f' % song_pr_auc)
        print('text_roc_auc: %.4f' % text_roc_auc)
        print('text_pr_auc: %.4f' % text_pr_auc)

        text2song_sim = self.get_similarity(text_embs, song_embs)
        sorted_indice = np.array(np.argsort(text2song_sim, axis=1)[:, ::-1])

        # count retrieval success with approximation
        approx_success = []
        for i in range(len(sorted_indice)):
            success = []
            for song_ix in sorted_indice[i]:
                key = self.text_moods[text_binaries[i].argmax()]
                values = self.approx_dict[key]
                song_tag = self.song_moods[song_binaries[song_ix].argmax()]
                if song_tag in values:
                    success.append(1)
                else:
                    success.append(0)
            approx_success.append(success)
        approx_success = np.array(approx_success)

        # get precision
        p_5 = approx_success[:, :5].mean(axis=1)
        p_5 = np.mean(p_5)
        print('P@5: %.4f' % p_5)

        eval_metric = p_5

        return torch.tensor(eval_metric)

    def get_similarity(self, embs_a, embs_b):
        sim_scores = np.zeros((len(embs_a), len(embs_b)))
        embs_a = embs_a.detach().cpu()
        embs_b = embs_b.detach().cpu()
        for i in range(len(embs_a)):
            sim_scores[i] = np.array(nn.CosineSimilarity(dim=-1)(embs_a[i], embs_b))
        return sim_scores

    def triplet_sampling(self, anchor_emb, positive_emb, binary, is_weighted=True):
        num_batch = len(anchor_emb)
        if is_weighted:
            # get distance weights
            anchor_norm = anchor_emb / anchor_emb.norm(dim=1)[:, None]
            positive_norm = positive_emb / positive_emb.norm(dim=1)[:, None]
            dot_sim = torch.matmul(anchor_norm, positive_norm.T)
            weights = (dot_sim + 1) / 2

            # masking
            mask = 1 - torch.matmul(binary, binary.T)
            masked_weights = weights * mask

            # sampling
            index_array = torch.arange(num_batch)
            negative_ix = [random.choices(index_array, weights=masked_weights[i], k=1)[0].item() for i in range(num_batch)]
            negative_emb = positive_emb[negative_ix]

        else:
            num_batch = len(anchor_emb)

            # masking
            mask = 1 - torch.matmul(binary, binary.T)

            # sampling
            index_array = torch.arange(num_batch)
            negative_ix = [random.choices(index_array, weights=mask[i], k=1)[0].item() for i in range(num_batch)]
            negative_emb = positive_emb[negative_ix]
        return anchor_emb, positive_emb, negative_emb

