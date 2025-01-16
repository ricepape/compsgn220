#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Tuple

from torch.utils.data import Dataset
import numpy as np
import pickle
import file_io as fio



__docformat__ = 'reStructuredText'
__all__ = ['MyDataset']


class MyDataset(Dataset):

    def __init__(self, dataset_dir: str) -> None:
        super().__init__()
        self.data = []
        file_paths = fio.get_files_from_dir_with_pathlib(dataset_dir)
        for file_path in file_paths:
            with open(file_path, 'rb') as f:  # Use 'rb' mode for reading binary files
                self.data.append(pickle.load(f))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        item = self.data[idx]
        mel_spectrogram = item['features']
        label = item['class']
        features = np.mean(mel_spectrogram, axis=1)  # Temporal average of the spectrogram
        return features, label

# EOF

