# from utils import set_seed

import numpy as np
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from typing import Callable, Optional
import torch.nn as nn
import torchvision.transforms as transforms

from spikingjelly.datasets.shd import SpikingHeidelbergDigits
from spikingjelly.datasets.shd import SpikingSpeechCommands
from spikingjelly.datasets import pad_sequence_collate,padded_sequence_mask

import torch
import torchaudio
from torchaudio.transforms import Spectrogram, MelScale, AmplitudeToDB, Resample
from torchaudio.datasets.speechcommands import SPEECHCOMMANDS
from torchaudio.datasets.librispeech import LIBRISPEECH
from torchvision import transforms
import augmentations
import os
import torch.distributed as dist
import random
import requests
import zipfile
import pandas as pd
# from pytorch_lightning import LightningDataModule
from scipy.io import wavfile
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

class TextTransform:
    """Maps characters to integers and vice versa"""
    def __init__(self):
        char_map_str = """
            ' 0
            <SPACE> 1
            a 2
            b 3
            c 4
            d 5
            e 6
            f 7
            g 8
            h 9
            i 10
            j 11
            k 12
            l 13
            m 14
            n 15
            o 16
            p 17
            q 18
            r 19
            s 20
            t 21
            u 22
            v 23
            w 24
            x 25
            y 26
            z 27
            """
        self.char_map = {}
        for line in char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer array """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence


class SpecAugment(nn.Module):
    """Spectrogram Augmentation

    Args:
        spec_augment: whether to apply spec augment
        mF: number of frequency masks
        F: maximum frequency mask size
        mT: number of time masks
        pS: adaptive maximum time mask size in %

    References:
        SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition, Park et al.
        https://arxiv.org/abs/1904.08779

        SpecAugment on Large Scale Datasets, Park et al.
        https://arxiv.org/abs/1912.05533

    """

    def __init__(self, config):
        super(SpecAugment, self).__init__()
        self.mF = config.mF
        self.F = config.F
        self.mT = config.mT
        self.pS = config.pS

    def forward(self, x, x_len):

        x = x.transpose(1, 2)
        # Frequency Masking
        for _ in range(self.mF):
            x = torchaudio.transforms.FrequencyMasking(freq_mask_param=self.F, iid_masks=False).forward(x)
        # Time Masking
        for b in range(x.size(0)):
            T = int(self.pS * x_len[b])
            for _ in range(self.mT):
                x[b:b + 1, :, :x_len[b]] = torchaudio.transforms.TimeMasking(time_mask_param=T).forward(
                    x[b:b + 1, :, :x_len[b]])

        x = x.transpose(1, 2)
        return x


class SpecAugmenter:
    def __init__(self, config):
        """
        Class to perform spectral augmentation on mel spectrograms.
        n_time_masks: int, time_mask_width: int, n_freq_masks: int, freq_mask_width: int
        Args:
            n_time_masks (int): Number of time bands to mask.
            time_mask_width (int): Maximum width of each time band to mask.
            n_freq_masks (int): Number of frequency bands to mask.
            freq_mask_width (int): Maximum width of each frequency band to mask.
        """
        self.n_time_masks = config.n_time_masks
        self.time_mask_width = config.time_mask_width
        self.n_freq_masks = config.n_freq_masks
        self.freq_mask_width = config.freq_mask_width

    def __call__(self, mel_spec):
        """
        Apply spectral augmentation to a mel spectrogram.

        Args:
            mel_spec (np.ndarray): Mel spectrogram, array of shape (n_mels, T).

        Returns:
            np.ndarray: Spectrogram with random time and frequency bands masked out.
        """
        mel_spec = mel_spec.transpose(1, 2)

        # Apply time masks
        for _ in range(self.n_time_masks):
            if mel_spec.shape[1] > self.time_mask_width:
                offset = np.random.randint(0, self.time_mask_width)
                begin = np.random.randint(0, mel_spec.shape[1] - offset)
                mel_spec[:, begin: begin + offset] = 0.0

        # Apply frequency masks
        for _ in range(self.n_freq_masks):
            if mel_spec.shape[0] > self.freq_mask_width:
                offset = np.random.randint(0, self.freq_mask_width)
                begin = np.random.randint(0, mel_spec.shape[0] - offset)
                mel_spec[begin: begin + offset, :] = 0.0

        mel_spec = mel_spec.transpose(1, 2)

        return mel_spec




class TimeNeurons_mask_aug(object):

  def __init__(self, config):
    self.config = config

  def __call__(self, x, y):

    if np.random.uniform() < self.config.TN_mask_aug_proba:
        mask_size = int(self.config.time_mask_proportion * x.shape[0])  # Use pS to determine the mask size proportionally
        ind = np.random.randint(0, x.shape[0] - mask_size)
        x[ind:ind + mask_size, :] = 0

    # Neuron mask
    if np.random.uniform() < self.config.TN_mask_aug_proba:
        mask_size = np.random.randint(0, self.config.neuron_mask_size)
        ind = np.random.randint(0, x.shape[1] - self.config.neuron_mask_size)
        x[:, ind:ind+mask_size] = 0

    return x, y




class Augs(object):

    def __init__(self, config):
        self.config = config
        self.augs = [TimeNeurons_mask_aug(config)] # ,TimeJitterAug(config)
        # self.augs = [TimeNeurons_mask_aug(config)]

    def __call__(self, x, y):
        for aug in self.augs:
            x, y = aug(x, y)

        return x, y

    def list_augs(self):
        # This will return the names of the augmentation classes used
        return [aug.__class__.__name__ for aug in self.augs]

def SHD_dataloaders(config):

    set_seed(config.seed)

    if config.use_aug:
        augs = Augs(config)
        print("Data Augmentations used:", augs.list_augs())
        train_dataset = BinnedSpikingHeidelbergDigits(config.datasets_path, config.n_bins, train=True, data_type='frame', duration=config.time_step, transform=augs)
    else:
        train_dataset = BinnedSpikingHeidelbergDigits(config.datasets_path, config.n_bins, train=True, data_type='frame', duration=config.time_step, transform=None)

    test_dataset= BinnedSpikingHeidelbergDigits(config.datasets_path, config.n_bins, train=False, data_type='frame', duration=config.time_step)

    train_loader = DataLoader(train_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size, shuffle=True, num_workers=4)

    test_loader = DataLoader(test_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size, num_workers=4)

    return train_loader, test_loader



def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def SSC_dataloaders(config):
  set_seed(config.seed)

  if config.use_aug:
      augs = Augs(config)
      print("Data Augmentations used:", augs.list_augs())
      train_dataset = BinnedSpikingSpeechCommands(config.datasets_path, config.n_bins, split='train', data_type='frame', duration=config.time_step,transform=augs)

  else:
      train_dataset = BinnedSpikingSpeechCommands(config.datasets_path, config.n_bins, split='train', data_type='frame', duration=config.time_step, transform=None)
  valid_dataset = BinnedSpikingSpeechCommands(config.datasets_path, config.n_bins, split='valid', data_type='frame', duration=config.time_step)
  test_dataset = BinnedSpikingSpeechCommands(config.datasets_path, config.n_bins, split='test', data_type='frame', duration=config.time_step)


  train_loader = DataLoader(train_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size, shuffle=True, num_workers=4)
  valid_loader = DataLoader(valid_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size, num_workers=4)
  test_loader = DataLoader(test_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size, num_workers=4)

  return train_loader, valid_loader, test_loader




def GSC_dataloaders(config,is_transform=False):
  set_seed(config.seed)

  train_dataset = GSpeechCommands(config.datasets_path, 'training', transform=build_transform(config, is_transform), target_transform=target_transform)
  valid_dataset = GSpeechCommands(config.datasets_path, 'validation', transform=build_transform(config, is_transform), target_transform=target_transform)
  test_dataset = GSpeechCommands(config.datasets_path, 'testing', transform=build_transform(config, is_transform), target_transform=target_transform)


  train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
  valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, num_workers=4)
  test_loader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=4)

  return train_loader, valid_loader, test_loader



class BinnedSpikingHeidelbergDigits(SpikingHeidelbergDigits):
    def __init__(
            self,
            root: str,
            n_bins: int,
            train: bool = None,
            data_type: str = 'event',
            frames_number: int = None,
            split_by: str = None,
            duration: int = None,
            custom_integrate_function: Callable = None,
            custom_integrated_frames_dir_name: str = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        """
        The Spiking Heidelberg Digits (SHD) dataset, which is proposed by `The Heidelberg Spiking Data Sets for the Systematic Evaluation of Spiking Neural Networks <https://doi.org/10.1109/TNNLS.2020.3044364>`_.

        Refer to :class:`spikingjelly.datasets.NeuromorphicDatasetFolder` for more details about params information.

        .. admonition:: Note
            :class: note

            Events in this dataset are in the format of ``(x, t)`` rather than ``(x, y, t, p)``. Thus, this dataset is not inherited from :class:`spikingjelly.datasets.NeuromorphicDatasetFolder` directly. But their procedures are similar.

        :class:`spikingjelly.datasets.shd.custom_integrate_function_example` is an example of ``custom_integrate_function``, which is similar to the cunstom function for DVS Gesture in the ``Neuromorphic Datasets Processing`` tutorial.
        """
        super().__init__(root, train, data_type, frames_number, split_by, duration, custom_integrate_function, custom_integrated_frames_dir_name, transform, target_transform)
        self.n_bins = n_bins

    def __getitem__(self, i: int):
        if self.data_type == 'event':
            events = {'t': self.h5_file['spikes']['times'][i], 'x': self.h5_file['spikes']['units'][i]}
            label = self.h5_file['labels'][i]
            # if self.transform is not None:
            #     events = self.transform(events)
            # if self.target_transform is not None:
            #     label = self.target_transform(label)
            if self.transform is not None:
                events, label = self.transform(events,label)

            return events, label

        elif self.data_type == 'frame':
            frames = np.load(self.frames_path[i], allow_pickle=True)['frames'].astype(np.float32)
            label = self.frames_label[i]

            binned_len = frames.shape[1]//self.n_bins
            binned_frames = np.zeros((frames.shape[0], binned_len))
            for i in range(binned_len):
                binned_frames[:,i] = frames[:, self.n_bins*i : self.n_bins*(i+1)].sum(axis=1)

            # if self.transform is not None:
            #     binned_frames = self.transform(binned_frames)
            # if self.target_transform is not None:
            #     label = self.target_transform(label)
            if self.transform is not None:
                binned_frames, label = self.transform(binned_frames,label)
            return binned_frames, label



class BinnedSpikingSpeechCommands(SpikingSpeechCommands):
    def __init__(
            self,
            root: str,
            n_bins: int,
            split: str = 'train',
            data_type: str = 'event',
            frames_number: int = None,
            split_by: str = None,
            duration: int = None,
            custom_integrate_function: Callable = None,
            custom_integrated_frames_dir_name: str = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        """
        The Spiking Speech Commands (SSC) dataset, which is proposed by `The Heidelberg Spiking Data Sets for the Systematic Evaluation of Spiking Neural Networks <https://doi.org/10.1109/TNNLS.2020.3044364>`_.

        Refer to :class:`spikingjelly.datasets.NeuromorphicDatasetFolder` for more details about params information.

        .. admonition:: Note
            :class: note

            Events in this dataset are in the format of ``(x, t)`` rather than ``(x, y, t, p)``. Thus, this dataset is not inherited from :class:`spikingjelly.datasets.NeuromorphicDatasetFolder` directly. But their procedures are similar.

        :class:`spikingjelly.datasets.shd.custom_integrate_function_example` is an example of ``custom_integrate_function``, which is similar to the cunstom function for DVS Gesture in the ``Neuromorphic Datasets Processing`` tutorial.
        """
        super().__init__(root, split, data_type, frames_number, split_by, duration, custom_integrate_function, custom_integrated_frames_dir_name, transform, target_transform)
        self.n_bins = n_bins

    def __getitem__(self, i: int):
        if self.data_type == 'event':
            events = {'t': self.h5_file['spikes']['times'][i], 'x': self.h5_file['spikes']['units'][i]}
            label = self.h5_file['labels'][i]
            # if self.transform is not None:
            #     events = self.transform(events)
            # if self.target_transform is not None:
            #     label = self.target_transform(label)
            if self.transform is not None:
                events, label = self.transform(events,label)
            print(label)
            return events, label

        elif self.data_type == 'frame':
            frames = np.load(self.frames_path[i], allow_pickle=True)['frames'].astype(np.float32)
            label = self.frames_label[i]

            binned_len = frames.shape[1]//self.n_bins
            binned_frames = np.zeros((frames.shape[0], binned_len))
            for i in range(binned_len):
                binned_frames[:,i] = frames[:, self.n_bins*i : self.n_bins*(i+1)].sum(axis=1)

            # if self.transform is not None:
            #     binned_frames = self.transform(binned_frames)
            # if self.target_transform is not None:
            #     label = self.target_transform(label)
            if self.transform is not None:
                binned_frames, label = self.transform(binned_frames,label)
            # label 为 0-34
            # print(type(label)) # int
            # print(label)
            return binned_frames, label


def build_transform(config, is_train):
    sample_rate = 16000
    window_size = config.window_size
    hop_length = config.hop_length
    n_mels = config.n_mels
    f_min = 50
    f_max = 14000

    t = [augmentations.PadOrTruncate(sample_rate),
         Resample(sample_rate, sample_rate // 2)]
    pass
    if is_train:
        t.extend([augmentations.RandomRoll(dims=(1,)),
                  augmentations.SpeedPerturbation(rates=(0.5, 1.5), p=0.5)
                 ])

    t.append(Spectrogram(n_fft=window_size, hop_length=hop_length, power=2))

    if is_train:
        pass

    t.extend([MelScale(n_mels=n_mels,
                       sample_rate=sample_rate // 2,
                       f_min=f_min,
                       f_max=f_max,
                       n_stft=window_size // 2 + 1),
              AmplitudeToDB()
             ])

    return transforms.Compose(t)

labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']

target_transform = lambda word : torch.tensor(labels.index(word))

class GSpeechCommands(Dataset):
    def __init__(self, root, split_name, transform=None, target_transform=None, download=True):

        self.split_name = split_name
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = SPEECHCOMMANDS(root, download=download, subset=split_name)


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index):
        # Return Tuple of the following item: Waveform, Sample rate, Label, Speaker ID, Utterance number
        waveform, _,label,_,_ = self.dataset.__getitem__(index)

        if self.transform is not None:
            waveform = self.transform(waveform).squeeze().t()

        target = label

        if self.target_transform is not None:
            target = self.target_transform(target)

        # number = torch.sum(waveform.ne(-100.0000))
        mask = waveform.ne(-100.0000)
        valid_rows = mask.all(dim=1)  # 在频率维度上检查，确定没有一行完全是 -100.0000
        # number = torch.sum(valid_rows)  # 计算不全为 -100.0000 的行数
        number = len(valid_rows)

        # return waveform, target, torch.zeros(1)
        return waveform, target, number


def download_data():
    url = 'https://github.com/karoldvl/ESC-50/archive/master.zip'
    r = requests.get(url, allow_redirects=True)

    if not os.path.exists("data/ESC50"):
        os.mkdir("data/ESC50")

    open('data/ESC50/data.zip', 'wb').write(r.content)
    with zipfile.ZipFile('data/ESC50/data.zip', 'r') as zip_ref:
        zip_ref.extractall("data/ESC50/")


def esc_collate_fn(batch):

    batch = pd.DataFrame(batch).to_dict(orient="list")
    batch["input"] = torch.from_numpy(np.stack(batch["input"], axis=0))
    batch["target"] = torch.LongTensor(batch["target"])
    return batch