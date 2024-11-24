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
                 numeric_features_std=None, targets_mean=None, targets_std=None, img_width=None, eval=False):

        self.img_width = img_width
        self.eval = eval

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
        image = self.images[idx]
        image_width = image.shape[0]
        start = 0
        if not self.eval:
            # Cut image to width at a random place
            start = np.random.randint(0, image_width - self.img_width + 1)
        image = image[start:start + self.img_width]
        return image, self.numeric_features[idx], self.targets[idx]


def collate_fn_w_mask(batch):
    sequences, numeric_features, targets = zip(*batch)
    padded_sequences, attention_mask = pad_sequences(sequences)
    numeric_features = torch.stack(numeric_features)
    targets = torch.stack(targets)

    return padded_sequences, attention_mask, numeric_features, targets
def collate_fn_wo_mask(batch):
    sequences, numeric_features, targets = zip(*batch)
    padded_sequences, attention_mask = pad_sequences(sequences)
    numeric_features = torch.stack(numeric_features)
    targets = torch.stack(targets)

    return padded_sequences, numeric_features, targets

def split_data_into_datasets(train=0.8, val=0.1, test=0.1, seed=1104, device='cpu', min_width=None, no_samples = -1, split_on_files=True):

    # good_data_paths = {
    #     70: 'good_data_70th.csv',
    #     80: 'good_data_80th.csv',
    #     90: 'good_data_90th.csv',
    #     100: 'good_data.csv'
    # }
    # numerical_data_paths = {
    #     70: 'datasets/auxiliary_data_70th.csv',
    #     80: 'datasets/auxiliary_data_80th.csv',
    #     90: 'datasets/auxiliary_data_90th.csv',
    #     100: 'datasets/auxiliary_data.csv'
    # }
    # targets_paths = {
    #     70: 'datasets/targets_70th.csv',
    #     80: 'datasets/targets_80th.csv',
    #     90: 'datasets/targets_90th.csv',
    #     100: 'datasets/targets.csv'
    # }

    data_file = 'src/datasets/combined_lc.csv'
    # Load the data
    data_df = pd.read_csv(os.path.join(utils.get_project_root(), data_file))
    if min_width:
        data_df = data_df[data_df['width'] >= min_width]

    if no_samples > 0:
        data_df = data_df.head(no_samples)

    if split_on_files:
        # Split data by filenames
        file_names = data_df['file_name'].unique()
        file_names_temp, file_names_test = train_test_split(file_names, train_size=train+val, random_state=seed)
        file_names_train, file_names_val = train_test_split(file_names_temp, train_size=train/(train+val), random_state=seed+1)

        print(f"Train: {len(file_names_train)} files, Val: {len(file_names_val)} files, Test: {len(file_names_test)} files")

        train_data = data_df[data_df['file_name'].isin(file_names_train)]
        val_data = data_df[data_df['file_name'].isin(file_names_val)]
        test_data = data_df[data_df['file_name'].isin(file_names_test)]
        print(f"Train: {len(train_data)} samples, Val: {len(val_data)} samples, Test: {len(test_data)} samples")
    else:
        train_data, test_data = train_test_split(data_df, test_size=test, random_state=seed)
        train_data, val_data = train_test_split(train_data, test_size=val/(train + val), random_state=seed + 1)


    target_column = 'ang_vel[deg/s]'
    numerical_columns = ['IRSKY_TEMP', 'TEMP', 'WINDSP', 'PRES', 'FWHM', 'RHUM', 'TAU0']


    def get_data(data: pd.DataFrame):
        images = []
        targets = torch.tensor(data[target_column].to_numpy(), dtype=torch.float, device=device)
        numerical = torch.tensor(data[numerical_columns].to_numpy(), dtype=torch.float, device=device)
        for _, row in data.iterrows():
            image = torch.tensor(np.load(utils.get_strip_file_path(row)), dtype=torch.float, device=device)
            images.append(image)
        return images, numerical, targets

    train_dataset = StreaksDataset(*get_data(train_data), img_width=min_width)

    val_dataset = StreaksDataset(   *get_data(val_data),
                                    images_mean=train_dataset.images_mean,
                                    images_std=train_dataset.images_std,
                                    numeric_features_mean=train_dataset.numeric_features_mean,
                                    numeric_features_std=train_dataset.numeric_features_std,
                                    targets_mean=train_dataset.targets_mean,
                                    targets_std=train_dataset.targets_std,
                                    img_width=min_width,
                                    eval=True
                                    )
    test_dataset = StreaksDataset(  *get_data(test_data),
                                    images_mean=train_dataset.images_mean,
                                    images_std=train_dataset.images_std,
                                    numeric_features_mean=train_dataset.numeric_features_mean,
                                    numeric_features_std=train_dataset.numeric_features_std,
                                    targets_mean=train_dataset.targets_mean,
                                    targets_std=train_dataset.targets_std,
                                    img_width=min_width,
                                    eval=True
                                    )

    train_df = train_data.reset_index(drop=True)
    val_df = val_data.reset_index(drop=True)
    test_df = test_data.reset_index(drop=True)

    return {"train": train_dataset, "val": val_dataset, "test": test_dataset}, {"train": train_df, "val": val_df, "test": test_df}

















