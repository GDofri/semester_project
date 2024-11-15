import os

import pandas as pd
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from src import utils


def pad_sequences(sequences):
    """
    sequences: List of tensors with shape (seq_len_i, input_dim)
    Returns:
        padded_sequences: Tensor of shape (batch_size, max_seq_len, input_dim)
        attention_mask: Boolean tensor of shape (batch_size, max_seq_len)
    """
    # Pad sequences
    padded_sequences = pad_sequence(sequences, batch_first=True)  # (batch_size, max_seq_len, input_dim)
    # Create attention mask
    attention_mask = torch.zeros(padded_sequences.size(0), padded_sequences.size(1), dtype=torch.bool)
    for i, seq in enumerate(sequences):
        attention_mask[i, :seq.size(0)] = True
    return padded_sequences, attention_mask


class StreaksDataset(Dataset):
    def __init__(self, images, numeric_features, targets, images_mean=None, images_std=None, numeric_features_mean=None,
                 numeric_features_std=None, targets_mean=None, targets_std=None):

        images = [torch.log(image + 1) for image in images]
        images = [image.transpose(0, 1) for image in images]

        targets = torch.log(targets + 10e-6)  # List of targets

        # images list of tensors of shape (,)
        # numeric_features (num_images, num_numeric_features)
        # targets (num_images, 1)
        # images_mean, images_std, targets_mean, targets_std: Tensors of shape (1,)
        # numeric_features_mean, numeric_features_std: Tensors of shape (num_numeric_features,)

        # Check if scaling data was provided
        scaling_data = [images_mean, images_std,
                        numeric_features_mean, numeric_features_std,
                        targets_mean, targets_std]

        provided_data = [x is not None for x in scaling_data]
        if any(provided_data) and not all(provided_data):
            raise ValueError("If any of the scaling data is provided, all of them should be provided.")
        if all(provided_data):
            # Scaling data provided
            self.images_mean, self.images_std, self.numeric_features_mean, self.numeric_features_std, self.targets_mean, self.targets_std = scaling_data
        else:
            # Scaling data not provided, calculate it
            flat_images = torch.cat( [image.flatten() for image in images] )
            self.images_mean = flat_images.mean()
            self.images_std = flat_images.std()
            self.numeric_features_mean = numeric_features.mean(dim=0)
            self.numeric_features_std = numeric_features.std(dim=0)
            self.targets_mean = targets.mean()
            self.targets_std = targets.std()

        # Scale data
        self.images = [(image - self.images_mean) / self.images_std for image in images]
        self.numeric_features = (numeric_features - self.numeric_features_mean) / self.numeric_features_std
        self.targets = (targets - self.targets_mean) / self.targets_std

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.images[idx], self.numeric_features[idx], self.targets[idx]


def collate_fn(batch):
    sequences, numeric_features, targets = zip(*batch)
    padded_sequences, attention_mask = pad_sequences(sequences)
    numeric_features = torch.stack(numeric_features)
    targets = torch.stack(targets)

    return padded_sequences, attention_mask, numeric_features, targets


def split_data_into_datasets(train=0.8, val=0.1, test=0.1, seed=1104, device='cpu', best_nth = 100):

    good_data_paths = {
        70: 'good_data_70th.csv',
        80: 'good_data_80th.csv',
        90: 'good_data_90th.csv',
        100: 'good_data.csv'
    }
    numerical_data_paths = {
        70: 'datasets/auxiliary_data_70th.csv',
        80: 'datasets/auxiliary_data_80th.csv',
        90: 'datasets/auxiliary_data_90th.csv',
        100: 'datasets/auxiliary_data.csv'
    }
    targets_paths = {
        70: 'datasets/targets_70th.csv',
        80: 'datasets/targets_80th.csv',
        90: 'datasets/targets_90th.csv',
        100: 'datasets/targets.csv'
    }

    # Load the data
    good_data = pd.read_csv(os.path.join(utils.get_project_root(), good_data_paths[best_nth]))
    numerical_data_df = pd.read_csv(os.path.join(utils.get_project_root(), numerical_data_paths[best_nth]))
    targets_df = pd.read_csv(os.path.join(utils.get_project_root(), targets_paths[best_nth]))
    # Split data by filenames
    file_names = good_data['file_name'].unique()
    file_names_temp, file_names_test = train_test_split(file_names, train_size=train+val, random_state=seed)
    file_names_train, file_names_val = train_test_split(file_names_temp, train_size=train/(train+val), random_state=seed+1)

    train_data = good_data[good_data['file_name'].isin(file_names_train)]
    val_data = good_data[good_data['file_name'].isin(file_names_val)]
    test_data = good_data[good_data['file_name'].isin(file_names_test)]

    def get_data(data):
        images = [
            torch.tensor(np.load(utils.get_strip_file_path(row)), dtype=torch.float, device=device) for (_, row) in data.iterrows()
        ]
        numeric = torch.tensor(
            pd.merge(data, numerical_data_df, on='file_name')[numerical_data_df.columns].drop(columns=['file_name']).to_numpy(),
            dtype=torch.float, device=device
        )
        targets = torch.tensor(
            pd.merge(data.drop(columns=['ang_vel[deg/s]']), targets_df, on=['file_name', 'extension', 'ID'])['ang_vel[deg/s]'].to_numpy(),
            dtype=torch.float, device=device
        )
        return images, numeric, targets

    train_dataset = StreaksDataset(*get_data(train_data))

    val_dataset = StreaksDataset(*get_data(val_data),
                                 images_mean=train_dataset.images_mean,
                                 images_std=train_dataset.images_std,
                                 numeric_features_mean=train_dataset.numeric_features_mean,
                                 numeric_features_std=train_dataset.numeric_features_std,
                                 targets_mean=train_dataset.targets_mean,
                                 targets_std=train_dataset.targets_std)
    test_dataset = StreaksDataset(*get_data(test_data),
                                   images_mean=train_dataset.images_mean,
                                   images_std=train_dataset.images_std,
                                   numeric_features_mean=train_dataset.numeric_features_mean,
                                   numeric_features_std=train_dataset.numeric_features_std,
                                   targets_mean=train_dataset.targets_mean,
                                   targets_std=train_dataset.targets_std)

    return train_dataset, val_dataset, test_dataset


















