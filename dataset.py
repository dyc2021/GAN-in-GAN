import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import random
import soundfile as sf
import torchaudio

DATASET_FILE_PATH = os.path.join(".", "dataset")
CHUNK_LENGTH = 159 * 159
WIN_SIZE = 318
FFT_NUM = 318
WIN_SHIFT = 159

# CHUNK_LENGTH = 160 * 160
# WIN_SIZE = 320
# FFT_NUM = 320
# WIN_SHIFT = 160


class ToTensor(object):
    def __call__(self, x, type='float'):
        if type == 'float':
            return torch.FloatTensor(x)
        elif type == 'int':
            return torch.IntTensor(x)


class TrainDataset(Dataset):
    def __init__(self, json_dir, batch_size):
        self.json_dir = json_dir
        self.batch_size = batch_size
        json_pos = os.path.join(json_dir, 'train', 'files.json')
        with open(json_pos, 'r') as f:
            json_list = json.load(f)

        minibatch = []
        start = 0
        while True:
            end = min(len(json_list), start + batch_size)
            minibatch.append(json_list[start:end])
            start = end
            if end == len(json_list):
                break
        self.minibatch = minibatch

    def __len__(self):
        return len(self.minibatch)

    def __getitem__(self, index):
        return self.minibatch[index]


class TestDataset(Dataset):
    def __init__(self, json_dir, batch_size):
        self.json_dir = json_dir
        self.batch_size = batch_size
        json_pos = os.path.join(json_dir, 'test', 'files.json')
        with open(json_pos, 'r') as f:
            json_list = json.load(f)

        minibatch = []
        start = 0
        while True:
            end = min(len(json_list), start + batch_size)
            minibatch.append(json_list[start:end])
            start = end
            if end == len(json_list):
                break
        self.minibatch = minibatch

    def __len__(self):
        return len(self.minibatch)

    def __getitem__(self, index):
        return self.minibatch[index]


class TrainDataLoader(object):
    def __init__(self, data_set, **kw):
        self.data_loader = DataLoader(dataset=data_set,
                                      shuffle=True,
                                      collate_fn=self.collate_fn,
                                      **kw)

    @staticmethod
    def collate_fn(batch):
        feats, labels, frame_mask_list = generate_feats_labels(batch)
        return BatchInfo(feats, labels, frame_mask_list)

    def get_data_loader(self):
        return self.data_loader


def generate_feats_labels(batch):
    batch = batch[0]
    feat_list, label_list, frame_mask_list = [], [], []
    to_tensor = ToTensor()
    for id in range(len(batch)):
        clean_file_name = "{}.wav".format(batch[id])
        mix_file_name = "{}.wav".format(batch[id])
        feat_wav, ori_feat_sr = sf.read(os.path.join(DATASET_FILE_PATH, "train", "noisy_trainset_28spk_wav",
                                                     mix_file_name))
        label_wav, ori_label_sr = sf.read(os.path.join(DATASET_FILE_PATH, "train", "clean_trainset_28spk_wav",
                                                       clean_file_name))

        feat_wav = torchaudio.functional.resample(to_tensor(feat_wav), orig_freq=ori_feat_sr, new_freq=16000)
        label_wav = torchaudio.functional.resample(to_tensor(label_wav), orig_freq=ori_label_sr, new_freq=16000)
        # feat_wav = to_tensor(feat_wav)
        # label_wav = to_tensor(label_wav)
        c = torch.sqrt(len(feat_wav) / torch.sum(feat_wav ** 2.0))
        feat_wav, label_wav = feat_wav * c, label_wav * c
        if len(feat_wav) > CHUNK_LENGTH:
            wav_start = random.randint(0, len(feat_wav) - CHUNK_LENGTH)
            feat_wav = feat_wav[wav_start:wav_start + CHUNK_LENGTH]
            label_wav = label_wav[wav_start:wav_start + CHUNK_LENGTH]

        frame_num = (len(feat_wav) - WIN_SIZE + FFT_NUM) // WIN_SHIFT + 1
        frame_mask_list.append(frame_num)
        feat_list.append(feat_wav)
        label_list.append(label_wav)

    feat_list = nn.utils.rnn.pad_sequence(feat_list, batch_first=True)
    label_list = nn.utils.rnn.pad_sequence(label_list, batch_first=True)
    feat_list = nn.ConstantPad1d((0, CHUNK_LENGTH - feat_list.shape[-1]), 0.0)(feat_list)
    label_list = nn.ConstantPad1d((0, CHUNK_LENGTH - label_list.shape[-1]), 0.0)(label_list)
    # feat_list = torch.stft(feat_list, n_fft=FFT_NUM, hop_length=WIN_SHIFT, win_length=WIN_SIZE,
    #                        window=torch.hann_window(FFT_NUM), return_complex=True)
    # label_list = torch.stft(label_list, n_fft=FFT_NUM, hop_length=WIN_SHIFT, win_length=WIN_SIZE,
    #                         window=torch.hann_window(FFT_NUM), return_complex=True)
    return feat_list, label_list, frame_mask_list


class TestDataLoader(object):
    def __init__(self, data_set, **kw):
        self.data_loader = DataLoader(dataset=data_set,
                                      shuffle=True,
                                      collate_fn=self.collate_fn,
                                      **kw)

    @staticmethod
    def collate_fn(batch):
        feats, labels, frame_mask_list = test_generate_feats_labels(batch)
        return BatchInfo(feats, labels, frame_mask_list)

    def get_data_loader(self):
        return self.data_loader


def test_generate_feats_labels(batch):
    batch = batch[0]
    feat_list, label_list, frame_mask_list = [], [], []
    to_tensor = ToTensor()
    for id in range(len(batch)):
        clean_file_name = "{}.wav".format(batch[id])
        mix_file_name = "{}.wav".format(batch[id])
        feat_wav, ori_feat_sr = sf.read(os.path.join(DATASET_FILE_PATH, "test", "noisy_testset_wav", mix_file_name))
        label_wav, ori_label_sr = sf.read(os.path.join(DATASET_FILE_PATH, "test", "clean_testset_wav", clean_file_name))

        feat_wav = torchaudio.functional.resample(to_tensor(feat_wav), orig_freq=ori_feat_sr, new_freq=16000)
        label_wav = torchaudio.functional.resample(to_tensor(label_wav), orig_freq=ori_label_sr, new_freq=16000)

        c = torch.sqrt(len(feat_wav) / torch.sum(feat_wav ** 2.0))
        feat_wav, label_wav = feat_wav * c, label_wav * c
        if len(feat_wav) > CHUNK_LENGTH:
            wav_start = random.randint(0, len(feat_wav) - CHUNK_LENGTH)
            feat_wav = feat_wav[wav_start:wav_start + CHUNK_LENGTH]
            label_wav = label_wav[wav_start:wav_start + CHUNK_LENGTH]

        frame_num = (len(feat_wav) - WIN_SIZE + FFT_NUM) // WIN_SHIFT + 1
        frame_mask_list.append(frame_num)
        feat_list.append(feat_wav)
        label_list.append(label_wav)

    feat_list = nn.utils.rnn.pad_sequence(feat_list, batch_first=True)
    label_list = nn.utils.rnn.pad_sequence(label_list, batch_first=True)
    feat_list = nn.ConstantPad1d((0, CHUNK_LENGTH - feat_list.shape[-1]), 0.0)(feat_list)
    label_list = nn.ConstantPad1d((0, CHUNK_LENGTH - label_list.shape[-1]), 0.0)(label_list)
    # feat_list = torch.stft(feat_list, n_fft=FFT_NUM, hop_length=WIN_SHIFT, win_length=WIN_SIZE,
    #                        window=torch.hann_window(FFT_NUM), return_complex=True)
    # label_list = torch.stft(label_list, n_fft=FFT_NUM, hop_length=WIN_SHIFT, win_length=WIN_SIZE,
    #                         window=torch.hann_window(FFT_NUM), return_complex=True)
    return feat_list, label_list, frame_mask_list


class BatchInfo(object):
    def __init__(self, feats, labels, frame_mask_list):
        self.feats = feats
        self.labels = labels
        self.frame_mask_list = frame_mask_list
