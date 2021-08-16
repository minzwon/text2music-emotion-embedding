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
    def __init__(self, data_path, text_dataset, split='TRAIN', input_length=None, num_chunk=16):
        self.data_path = data_path
        self.dataset = text_dataset
        self.split = split
        self.input_length = input_length
        self.num_chunk = num_chunk
        self.text_tags = np.load('./data/text_%s_moods.npy' % text_dataset)
        self.song_tags = np.load('./data/song_moods.npy')

        self.w2v = pickle.load(open('./data/w2v.pkl', 'rb'))

        # mapping
        w2v_mapper = {'anger': 'angry',
                       'fearful': 'scary',
                       'happy': 'happy',
                       'sad': 'sad',
                       'surprised': 'happy',
                       'disgust': 'angry',
                       'fear': 'angry',
                       'guilt': 'angry',
                       'joy': 'tender',
                       'sadness': 'tender',
                       'shame': 'sad'}
        va_mapper = {'anger': 'angry',
                       'disgust': 'angry',
                       'fear': 'angry',
                       'fearful': 'sad',
                       'guilt': 'sad',
                       'happy': 'happy',
                       'joy': 'exciting',
                       'sad': 'sad',
                       'sadness': 'sad',
                       'shame': 'angry',
                       'surprised': 'exciting'}
        self.tts_mapper = w2v_mapper

        if split == 'TRAIN':
            # text
            self.tag_to_text = pickle.load(open('./data/text_%s_tag_to_ix.pkl' % text_dataset, 'rb'))
            self.text_ix = np.load('./data/text_%s_train_ix.npy' % text_dataset)
            # song
            self.tag_to_song = pickle.load(open('./data/song_tag_to_ids.pkl', 'rb'))
            self.song_ids = np.load('./data/song_train_fn.npy')
            self.song_binaries = np.load('./data/song_train_binaries.npy')
        elif split == 'VALID':
            # text
            self.text_ix = np.load('./data/text_%s_valid_ix.npy' % text_dataset)
            # song
            self.song_ids = np.load('./data/song_valid_fn.npy')
            self.song_binaries = np.load('./data/song_valid_binaries.npy')
        elif split == 'TEST':
            # text
            self.text_ix = np.load('./data/text_%s_test_ix.npy' % text_dataset)
            # song
            self.song_ids = np.load('./data/song_test_fn.npy')
            self.song_binaries = np.load('./data/song_test_binaries.npy')

        # text
        corpus = np.load('./data/text_%s_corpus.npy' % text_dataset, allow_pickle=True)
        self.text_binaries = np.load('./data/text_%s_binaries.npy' % text_dataset)

        # tokenize
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        encoding = tokenizer(list(corpus), return_tensors='pt', padding=True, truncation=True)
        self.text_tokens = np.concatenate([encoding['input_ids'].unsqueeze(0), encoding['attention_mask'].unsqueeze(0)])

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

    def get_train_item(self, index):
        ttt_tag_emb, ttt_token, ttt_mask, ttt_binary = self.get_tag_to_text()
        tts_tag_emb, tts_song, tts_binary = self.get_tag_to_song()
        common_token, common_mask, common_song, common_binary = self.get_text_to_song()
        return ttt_tag_emb, ttt_token, ttt_mask, ttt_binary, \
                tts_tag_emb, tts_song, tts_binary, \
                common_token, common_mask, common_song, common_binary

    def get_tag_to_text(self):
        # get random tag
        i = random.randrange(len(self.text_tags))
        tag = self.text_tags[i]
        tag_emb = self.w2v[tag]

        # get text token/mask
        ix = random.choice(self.tag_to_text[tag])
        token = self.text_tokens[0][ix]
        mask = self.text_tokens[1][ix]
        text_binary = self.text_binaries[ix]

        return tag_emb.astype('float32'), token, mask, text_binary.astype('float32')

    def get_tag_to_song(self):
        # get random tag
        i = random.randrange(len(self.song_tags))
        tag = self.song_tags[i]
        tag_emb = self.w2v[tag]

        # get song audio
        ix, fn = random.choice(self.tag_to_song[tag]).split('---')
        fn = os.path.join(self.data_path, fn)
        song = self.load_audio(fn)
        song_binary = self.song_binaries[int(ix)]
        return tag_emb.astype('float32'), song.astype('float32'), song_binary.astype('float32')

    def get_text_to_song(self):
        i = random.randrange(len(self.text_tags))
        text_tag = self.text_tags[i]
        
        # get text token/mask
        ix = random.choice(self.tag_to_text[text_tag])
        token = self.text_tokens[0][ix]
        mask = self.text_tokens[1][ix]
        text_binary = self.text_binaries[ix]

        # get song audio
        song_tag = self.tts_mapper[text_tag]
        ix, fn = random.choice(self.tag_to_song[song_tag]).split('---')
        fn = os.path.join(self.data_path, fn)
        song = self.load_audio(fn)
        song_binary = self.song_binaries[int(ix)]
        return token, mask, song.astype('float32'), text_binary.astype('float32')

    def get_eval_item(self, index):
        # text tag
        text_tag = self.text_tags[index % len(self.text_tags)]
        text_tag_emb = self.w2v[text_tag]
        text_tag_binary = np.zeros(len(self.text_tags))
        text_tag_binary[np.where(self.text_tags==text_tag)[0]] = 1

        # song tag
        song_tag = self.song_tags[index % len(self.song_tags)]
        song_tag_emb = self.w2v[song_tag]
        song_tag_binary = np.zeros(len(self.song_tags))
        song_tag_binary[np.where(self.song_tags==song_tag)[0]] = 1

        # text
        ix = self.text_ix[index % len(self.text_ix)]
        token = self.text_tokens[0][ix]
        mask = self.text_tokens[1][ix]
        text_binary = self.text_binaries[ix]

        # song
        ix, fn = self.song_ids[index % len(self.song_ids)].split('---')
        song = self.load_audio(fn)
        song_binary = self.song_binaries[int(ix)]
        return song_tag_emb.astype('float32'), song_tag_binary.astype('float32'), text_tag_emb.astype('float32'), text_tag_binary.astype('float32'), song.astype('float32'), song_binary.astype('float32'), token, mask, text_binary.astype('float32')

    def __getitem__(self, index):
        if self.split=='TRAIN':
            return self.get_train_item(index)
        else:
            return self.get_eval_item(index)

    def __len__(self):
        return max(len(self.song_ids), len(self.text_ix))


