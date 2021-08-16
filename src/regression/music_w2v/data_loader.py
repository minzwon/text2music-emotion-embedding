import os
import sys
import pickle
import tqdm
import numpy as np
import pandas as pd
import random
from torch.utils import data


class MyDataset(data.Dataset):
    def __init__(self, data_path, split='TRAIN', input_length=None, num_chunk=16):
        self.split = split
        self.data_path = data_path
        self.input_length = input_length
        self.num_chunk = num_chunk
        self.song_tags = np.load('./data/song_moods.npy')

        if split == 'TRAIN':
            # song
            self.song_ids = np.load('./data/song_train_fn.npy')
            self.song_w2v = np.load('./data/song_train_w2v.npy')
            self.song_binaries = np.load('./data/song_train_binaries.npy', allow_pickle=True)
        elif split == 'VALID':
            # song
            self.song_ids = np.load('./data/song_valid_fn.npy')
            self.song_w2v = np.load('./data/song_valid_w2v.npy')
            self.song_binaries = np.load('./data/song_valid_binaries.npy', allow_pickle=True)
        elif split == 'TEST':
            # song
            self.song_ids = np.load('./data/song_test_fn.npy')
            self.song_w2v = np.load('./data/song_test_w2v.npy')
            self.song_binaries = np.load('./data/song_test_binaries.npy')

    def load_audio(self, fn):
        length = self.input_length
        raw = np.load(fn, mmap_mode='r')
        if len(raw) < length:
            nraw = np.zeros(length)
            nraw[:len(raw)] = raw
            raw = nraw

        # multiple chunks for eval loader
        if self.split == 'TRAIN':
            time_ix = int(np.floor(np.random.random(1) * (len(raw) - length)))
            raw = raw[time_ix:time_ix+length]
        elif (self.split=='VALID') or (self.split=='TEST'):
            hop = (len(raw) - self.input_length) // self.num_chunk
            raw = np.array([raw[i*hop:i*hop+length] for i in range(self.num_chunk)])
        return raw

    def __getitem__(self, index):
        # load audio and w2v
        ix, fn = self.song_ids[index].split('---')
        fn = os.path.join(self.data_path, fn)
        song = self.load_audio(fn)
        w2v = self.song_w2v[int(ix)]

        # get binaries
        binary = self.song_binaries[int(ix)]
        return song.astype('float32'), w2v.astype('float32'), binary.astype('float32')

    def __len__(self):
        return len(self.song_ids)


