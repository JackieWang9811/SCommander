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


# def SpecAugment(mel_spec: np.ndarray, n_time_masks: int, time_mask_width: int, n_freq_masks: int,
#                  freq_mask_width: int):
#     """Numpy implementation of spectral augmentation.
#
#     Args:
#         mel_spec (np.ndarray): Mel spectrogram, array of shape (n_mels, T).
#         n_time_masks (int): Number of time bands.
#         time_mask_width (int): Max width of each time band.
#         n_freq_masks (int): Number of frequency bands.
#         freq_mask_width (int): Max width of each frequency band.
#
#     Returns:
#         mel_spec (np.ndarray): Spectrogram with random time bands and freq bands masked out.
#     """
#
#     offset, begin = 0, 0
#
#     for _ in range(n_time_masks):
#         offset = np.random.randint(0, time_mask_width)
#         begin = np.random.randint(0, mel_spec.shape[1] - offset)
#         mel_spec[:, begin: begin + offset] = 0.0
#
#     for _ in range(n_freq_masks):
#         offset = np.random.randint(0, freq_mask_width)
#         begin = np.random.randint(0, mel_spec.shape[0] - offset)
#         mel_spec[begin: begin + offset, :] = 0.0
#
#     return mel_spec


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


# class RNoise(object):
#
#   def __init__(self, config):
#     self.config = config
#     self.sig = config.sig
#
#   def __call__(self, sample, label):
#     if np.random.uniform() < self.config.noise_proba:
#         # loc:随机数中心，scale，标准差，小瘦高，大矮宽
#         noise = np.abs(np.random.normal(1, self.sig, size=sample.shape).round())
#         return sample + noise, label
#     else:
#         return sample, label

class RNoise(object):
    def __init__(self, config):
        self.config = config
        self.sig = config.sig
        # 噪声值和它们的比例
        self.noise_values = np.arange(6)  # 假设噪声值从0到5
        self.noise_proportions = [87, 6, 3, 2, 1, 1]  # 例如噪声比例
        self.noise_proportions = np.array(self.noise_proportions) / np.sum(self.noise_proportions)  # 归一化比例

    def __call__(self, sample, label):
        if np.random.uniform() < self.config.noise_proba:
            # 根据比例生成噪声
            noise = np.random.choice(self.noise_values, size=sample.shape, p=self.noise_proportions)
            # 添加噪声到sample，确保结果仍在0到5之间
            noisy_sample = np.clip(sample + noise, 0, 5)
            return noisy_sample, label
        else:
            return sample, label


class TimeJitterAug(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, x, y):
        # Apply time jitter if probability check passes
        T,N = x.shape
        if np.random.uniform() < self.config.time_jitter_proba:
            for i in range(T):  # iterating over each neuron
                jitter = np.random.randint(-self.config.max_jitter, self.config.max_jitter+1)
                # Apply jitter within the bounds of the tensor
                x[i, :] = np.roll(x[i, :], jitter)
        return x, y

class NeuronJitterAug(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, x, y):
        # Apply channel jitter if probability check passes
        T, N = x.shape
        if np.random.uniform() < self.config.channel_jitter_proba:
            # n_neurons = x.shape[1]
            # indices = np.arange(n_neurons)
            # np.random.shuffle(indices)
            # x = x[:, indices]
            for j in range(N):
                neuron_jitter = np.random.randint(-self.config.max_jitter, self.config.max_jitter+1)
                x[:, j] = np.roll(x[:, j], neuron_jitter, axis=0)
        return x, y


class DropEventAug(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, x, y):
        # Apply drop event if probability check passes
        # if np.random.uniform() < self.config.drop_event_proba:
        num_events_to_drop = np.random.randint(0, self.config.max_drop_events)
        for _ in range(num_events_to_drop):
            t = np.random.randint(0, x.shape[0])  # random time
            n = np.random.randint(0, x.shape[1])  # random neuron
            x[t, n] = 0  # drop the event
        return x, y

class TimeNeurons_mask_aug(object):

  def __init__(self, config):
    self.config = config

  def __call__(self, x, y):
    # Sample shape: (time, neurons)

    # if np.random.uniform() < self.config.TN_mask_aug_proba:
    #     mask_size = np.random.randint(0, self.config.time_mask_size)
    #     ind = np.random.randint(0, x.shape[0] - self.config.time_mask_size)
    #     x[ind:ind+mask_size, :] = 0

    # Time mask with adaptive size based on pS
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



# class CutMix(object):
#     """
#     Apply Spectrogram-CutMix augmentaiton which only cuts patch across time axis unlike
#     typical Computer-Vision CutMix. Applies CutMix to one batch and its shifted version.
#
#     """
#
#     def __init__(self, config):
#         self.config = config
#
#     def __call__(self, x, y):
#
#         # x shape: (batch, time, neurons)
#         # Go to L-1, no need to augment last sample in batch (for ease of coding)
#         # Arrays to store the mixed targets
#         batch_size, time, neurons = x.shape
#         target_yi = y.clone()  # 克隆原始目标以便修改
#         target_yj = y.clone()
#
#         for i in range(batch_size - 1):
#             j = i + 1
#
#             if np.random.uniform() < self.config.cutmix_aug_proba:
#                 lam = self.config.cut_size_proba
#                 cut_size = int(lam * x[j].shape[0])
#                 ind = np.random.randint(0, x[i].shape[0] - cut_size)
#
#                 x[i, ind:ind + cut_size, :] = x[j, ind:ind + cut_size, :]
#                 target_yi[i] = (1 - lam) * y[i] + lam * y[j]
#                 target_yj[i] = y[j]
#             else:
#                 lam = self.config.cut_size_proba
#                 target_yi[i] = y[i]
#                 target_yj[i] = y[i]
#
#         return x, target_yi, target_yj, lam

class CutMix(object):
    """
    Apply Spectrogram-CutMix augmentation which only cuts patch across time axis unlike
    typical Computer-Vision CutMix. Randomly selects two samples in the batch for augmentation.
    """

    def __init__(self, config):
        self.config = config

    def __call__(self, x, y):
        # Ensure y is a float array to handle weighted labels
        # y = y.float()
        n_samples = x.shape[0]

        # Determine if CutMix is to be applied
        if np.random.uniform() < self.config.cutmix_aug_proba:
            # Generate one lambda for the whole batch
            lam = self.config.cut_size_proba

            # Arrays to store the mixed targets
            target_as = []
            target_bs = []

            for i in range(n_samples):
                # Randomly select another sample from the batch, ensuring it's not the same sample
                j = np.random.choice([k for k in range(n_samples) if k != i])
                cut_size = int(lam * x[j].shape[0])
                ind = np.random.randint(0, x[i].shape[0] - cut_size)

                # Perform the cut and mix operation
                x[i][ind:ind + cut_size, :] = x[j][ind:ind + cut_size, :]

                # Store the original targets
                target_as.append(y[i])
                target_bs.append(y[j])
        else:
            # No CutMix, use original labels and lambda=1
            lam = 1.0
            target_as = y.clone()
            target_bs = y.clone()

        # Convert lists to tensors if needed
        if isinstance(target_as, list):
            target_as = torch.stack(target_as)
            target_bs = torch.stack(target_bs)

        return x, target_as, target_bs, lam


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


def SHD_dataloaders_teacher_student(config, teacher_config):
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def set_seed(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    set_seed(config.seed)
    g = torch.Generator()
    g.manual_seed(config.seed)

    if config.use_aug:
        augs = Augs(config)
        print("Data Augmentations used:", augs.list_augs())
        train_dataset = BinnedSpikingHeidelbergDigits(config.datasets_path, config.n_bins, train=True,
                                                      data_type='frame', duration=config.time_step, transform=augs)
    else:
        train_dataset = BinnedSpikingHeidelbergDigits(config.datasets_path, config.n_bins, train=True,
                                                      data_type='frame', duration=config.time_step, transform=None)

    train_dataset_teacher = BinnedSpikingHeidelbergDigits(config.datasets_path, teacher_config.n_bins, train=True,
                                                          data_type='frame', duration=teacher_config.time_step,
                                                          transform=None)

    test_dataset = BinnedSpikingHeidelbergDigits(config.datasets_path, config.n_bins, train=False, data_type='frame',
                                                 duration=config.time_step)

    # 创建索引列表
    indices = torch.randperm(len(train_dataset)).tolist()

    # 创建两个 Subset 数据集，确保它们使用相同的索引
    train_subset = Subset(train_dataset, indices)
    train_subset_teacher = Subset(train_dataset_teacher, indices)

    train_loader = DataLoader(train_subset, collate_fn=pad_sequence_collate, batch_size=config.batch_size,
                              shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)
    teacher_train_loader = DataLoader(train_subset_teacher, collate_fn=pad_sequence_collate,
                                      batch_size=config.batch_size, shuffle=False, num_workers=4,
                                      worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size, num_workers=4)

    return teacher_train_loader, train_loader, test_loader

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


def SSC_dataloaders_teacher_student(config, teacher_config):
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def set_seed(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    set_seed(config.seed)
    g = torch.Generator()
    g.manual_seed(config.seed)

    if config.use_aug:
        augs = Augs(config)
        print("Data Augmentations used:", augs.list_augs())
        train_dataset = BinnedSpikingSpeechCommands(config.datasets_path, config.n_bins, split='train',
                                                    data_type='frame', duration=config.time_step, transform=augs)
    else:
        train_dataset = BinnedSpikingSpeechCommands(config.datasets_path, config.n_bins, split='train',
                                                    data_type='frame', duration=config.time_step, transform=None)

    train_dataset_teacher = BinnedSpikingSpeechCommands(config.datasets_path, teacher_config.n_bins, split='train',
                                                        data_type='frame', duration=teacher_config.time_step,
                                                        transform=None)

    valid_dataset = BinnedSpikingSpeechCommands(config.datasets_path, config.n_bins, split='valid', data_type='frame',
                                                duration=config.time_step)
    test_dataset = BinnedSpikingSpeechCommands(config.datasets_path, config.n_bins, split='test', data_type='frame',
                                               duration=config.time_step)

    # 创建索引列表
    indices = torch.randperm(len(train_dataset)).tolist()

    # 创建两个 Subset 数据集，确保它们使用相同的索引
    train_subset = Subset(train_dataset, indices)
    train_subset_teacher = Subset(train_dataset_teacher, indices)

    train_loader = DataLoader(train_subset, collate_fn=pad_sequence_collate, batch_size=config.batch_size,
                              shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)
    teacher_train_loader = DataLoader(train_subset_teacher, collate_fn=pad_sequence_collate,
                                      batch_size=config.batch_size, shuffle=False, num_workers=4,
                                      worker_init_fn=seed_worker, generator=g)
    valid_loader = DataLoader(valid_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size,
                              num_workers=4)
    test_loader = DataLoader(test_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size, num_workers=4)

    return teacher_train_loader, train_loader, valid_loader, test_loader



def GSC_dataloaders(config,is_transform=False):
  set_seed(config.seed)

  train_dataset = GSpeechCommands(config.datasets_path, 'training', transform=build_transform(config, is_transform), target_transform=target_transform)
  valid_dataset = GSpeechCommands(config.datasets_path, 'validation', transform=build_transform(config, is_transform), target_transform=target_transform)
  test_dataset = GSpeechCommands(config.datasets_path, 'testing', transform=build_transform(config, is_transform), target_transform=target_transform)


  train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
  valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, num_workers=4)
  test_loader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=4)

  return train_loader, valid_loader, test_loader


def GSC_dataloaders_teacher_student(config, teacher_config):
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def set_seed(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    set_seed(config.seed)
    g = torch.Generator()
    g.manual_seed(config.seed)

    # Load datasets with transformations
    train_dataset = GSpeechCommands(config.datasets_path, 'training', transform=build_transform(config, False), target_transform=target_transform)
    train_dataset_teacher = GSpeechCommands(config.datasets_path, 'training', transform=build_transform(teacher_config, False), target_transform=target_transform)

    valid_dataset = GSpeechCommands(config.datasets_path, 'validation', transform=build_transform(config, False), target_transform=target_transform)
    test_dataset = GSpeechCommands(config.datasets_path, 'testing', transform=build_transform(config, False), target_transform=target_transform)

    # Create index list for ensuring both datasets use the same subset of data
    indices = torch.randperm(len(train_dataset)).tolist()

    # Create two Subset datasets using the same indices
    train_subset = Subset(train_dataset, indices)
    train_subset_teacher = Subset(train_dataset_teacher, indices)

    # Dataloaders with specified settings for parallel loading and specific seed initialization
    train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)
    teacher_train_loader = DataLoader(train_subset_teacher, batch_size=config.batch_size, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)

    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=4)

    return teacher_train_loader, train_loader, valid_loader, test_loader


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


# def build_transform_change(config, is_train):
#     sample_rate = 16000
#     # window_size = config.window_size
#     # hop_length = config.hop_length
#     window_size = config.window_size // 2  # 将 window_size 缩小一半
#     hop_length = config.hop_length // 2  # 将 hop_length 缩小一半
#     n_mels = config.n_mels
#     f_min = 50
#     f_max = 14000
#
#     t = [augmentations.PadOrTruncate(sample_rate),
#          Resample(sample_rate, sample_rate // 4)]  # 修改为 sample_rate // 4
#     pass
#     if is_train:
#         t.extend([augmentations.RandomRoll(dims=(1,)),
#                   augmentations.SpeedPerturbation(rates=(0.5, 1.5), p=0.5)
#                  ])
#
#     t.append(Spectrogram(n_fft=window_size, hop_length=hop_length, power=2))
#
#     if is_train:
#         pass
#
#     t.extend([MelScale(n_mels=n_mels,
#                        sample_rate=sample_rate // 4,  # 修改为 sample_rate // 4
#                        f_min=f_min,
#                        f_max=f_max,
#                        n_stft=window_size // 2 + 1),
#               AmplitudeToDB()
#              ])
#
#     return transforms.Compose(t)

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


class ESC50Dataset(Dataset):

    def __init__(self, root, split=None, verbose=False):
        super().__init__()

        self.root = root
        if self.root[-1] != "/":
            self.root = self.root + "/"
        self.split = split
        self.verbose = verbose

        self.index = pd.read_csv(root + "/meta/esc50.csv")

        if split is not None:
            if split == "train":
                self.index = self.index[self.index.fold.isin([1, 2, 3])]
            elif split == "val":
                self.index = self.index[self.index.fold.isin([4])]
            elif split == "test":
                self.index = self.index[self.index.fold.isin([5])]

        self.data = []
        if self.verbose:
            loop = tqdm(range(len(self.index)))
        else:
            loop = range(len(self.index))
        for i in loop:
            filename = self.index.iloc[i].filename
            rate, sample = wavfile.read(self.root + "audio/" + filename)
            # breakpoint()
            target = self.index.iloc[i].target
            self.data.append({"input": np.expand_dims(sample, axis=0).astype(np.float32),
                              "target": target,
                              "len": len(sample)})

    def __getitem__(self, ind):
        # item = self.data[ind]
        # waveform, target, number = item["input"], item["target"], item["len"]
        # return waveform, target, number
        return self.data[ind]

    def __len__(self):
        return len(self.data)


def ESC_dataloaders(config):

    set_seed(config.seed)

    train_dataset = ESC50Dataset(root=config.datasets_path, split="train")
    valid_dataset = ESC50Dataset(root=config.datasets_path, split="val")

    test_dataset  = ESC50Dataset(root=config.datasets_path, split="test")


    train_loader = DataLoader(train_dataset, collate_fn=esc_collate_fn, batch_size=config.batch_size, shuffle=True,
                          num_workers=4)

    valid_loader = DataLoader(valid_dataset, collate_fn=esc_collate_fn, batch_size=config.batch_size, num_workers=4)

    test_loader = DataLoader(test_dataset, collate_fn=esc_collate_fn, batch_size=config.batch_size, num_workers=4)

    return train_loader, valid_loader, test_loader

def esc_collate_fn(batch):

    batch = pd.DataFrame(batch).to_dict(orient="list")
    batch["input"] = torch.from_numpy(np.stack(batch["input"], axis=0))
    batch["target"] = torch.LongTensor(batch["target"])
    return batch