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


class ArtificialStreaksDataset(Dataset):
    def __init__(self, images, targets,
                 images_mean=None, images_std=None,
                 targets_mean=None, targets_std=None):

        images = [torch.log(image + 1) for image in images]
        images = [image.transpose(0, 1) for image in images]

        # targets = torch.log(targets + 10e-6)  # List of targets
        # images list of tensors of shape (,)
        # targets (num_images, 1)
        # images_mean, images_std, targets_mean, targets_std: Tensors of shape (1,)

        # Check if scaling data was provided
        scaling_data = [images_mean, images_std,
                        targets_mean, targets_std]

        provided_data = [x is not None for x in scaling_data]
        if any(provided_data) and not all(provided_data):
            raise ValueError("If any of the scaling data is provided, all of them should be provided.")
        if all(provided_data):
            # Scaling data provided
            self.images_mean, self.images_std, self.targets_mean, self.targets_std = scaling_data
        else:
            # Scaling data not provided, calculate it
            flat_images = torch.cat( [image.flatten() for image in images] )
            self.images_mean = flat_images.mean()
            self.images_std = flat_images.std()
            self.targets_mean = targets.mean()
            self.targets_std = targets.std()

        # Scale data
        self.images = [(image - self.images_mean) / self.images_std for image in images]
        self.targets = (targets - self.targets_mean) / self.targets_std

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]


def collate_fn(batch):
    sequences, targets = zip(*batch)
    padded_sequences, attention_mask = pad_sequences(sequences)
    targets = torch.stack(targets)

    return padded_sequences, attention_mask, targets


def split_data_into_datasets(train=0.8, val=0.1, test=0.1, seed=1104, device='cpu', data_path = None, no_samples = -1):


    if not data_path:
        data_path = os.path.join(utils.get_project_root(), 'src', 'datasets', 'artificial_strips', 'artificial_xw')

    # Load the data
    data_df = pd.read_csv(os.path.join(data_path, 'image_parameters.csv'))
    if no_samples > 0:
        data_df = data_df.head(no_samples)

    # Split the data into training + validation and testing
    train_val_df, test_df = train_test_split(data_df, test_size=test, random_state=seed)

    # Further split training + validation into separate training and validation sets
    train_df, val_df = train_test_split(train_val_df, test_size=val/(train + val), random_state=seed + 1)

    images_folder_path = data_path
    def get_data(data):
        images = [
            torch.tensor(np.load(os.path.join(images_folder_path, row['filename'])), dtype=torch.float, device=device) for (_, row) in data.iterrows()
        ]
        targets = torch.tensor(data['frequency'].to_numpy(), dtype=torch.float, device=device)
        return images, targets

    train_dataset = ArtificialStreaksDataset(*get_data(train_df))

    val_dataset = ArtificialStreaksDataset(*get_data(val_df),
                                 images_mean=train_dataset.images_mean,
                                 images_std=train_dataset.images_std,
                                 targets_mean=train_dataset.targets_mean,
                                 targets_std=train_dataset.targets_std)
    test_dataset = ArtificialStreaksDataset(*get_data(test_df),
                                  images_mean=train_dataset.images_mean,
                                  images_std=train_dataset.images_std,
                                  targets_mean=train_dataset.targets_mean,
                                  targets_std=train_dataset.targets_std)

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return {"train": train_dataset, "val": val_dataset, "test": test_dataset}, {"train": train_df, "val": val_df, "test": test_df}


















