#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import MutableSequence, List, Union, MutableMapping

import numpy as np
import torch
import librosa
import plotting as plt
import pickle


__docformat__ = 'reStructuredText'
__all__ = [
    'get_audio_file_data',
    'extract_mel_band_energies',
    'serialize_features_and_classes',
    'dataset_iteration'
]


def get_audio_file_data(audio_file: str) -> np.ndarray:
    """Loads and returns the audio data from the `audio_file`.

    :param audio_file: Path of the `audio_file` audio file.
    :type audio_file: str
    :return: Data of the `audio_file` audio file.
    :rtype: numpy.ndarray
    """
    audio, sr = librosa.load(audio_file, sr=22050)
    duration = len(audio) / sr
    print("Audio durations in seconds for file:", audio_file)
    print(duration)
    return audio[0]



def extract_mel_band_energies(audio_file: str) -> np.ndarray:
    """Extracts and returns the mel-band energies from the `audio_file` audio file.

    :param audio_file: Path of the audio file.
    :type audio_file: str
    :return: Mel-band energies of the `audio_file` audio file.
    :rtype: numpy.ndarray
    """
    audio, sr = librosa.load(audio_file, sr=22050)
    mel_spec = librosa.feature.melspectrogram(y = audio, sr = sr, window='hann', hop_length= int(0.01*sr), n_fft=int(0.02*sr), n_mels=40)
    print(mel_spec.shape)
    plt.plot_audio_signal(audio)
    plt.plot_mel_band_energies(mel_spec)
    return mel_spec



def serialize_features_and_classes(features_and_classes: MutableMapping[str, Union[np.ndarray, int]]) -> None:
    """Serializes the features and classes.

    :param features_and_classes: Features and classes.
    :type features_and_classes: dict[str, numpy.ndarray|int]
    """
    file_path = features_and_classes['file_path']
    with open(file_path, 'wb') as f:
        pickle.dump(features_and_classes, f)

def dataset_iteration(dataset: torch.utils.data.Dataset) -> None:
    """Iterates over the dataset using the DataLoader of PyTorch.

    :param dataset: Dataset to iterate over.
    :type dataset: torch.utils.data.Dataset
    """
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=True, drop_last=True)
    for features, labels in data_loader:
        print(features, labels)

# EOF
