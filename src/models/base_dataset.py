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


class BaseDataset(Dataset):
    def __init__(self, images, targets, numeric_features=None, img_width=-1,
                 images_mean=None, images_std=None,
                 targets_mean=None, targets_std=None,
                 numeric_features_mean=None,
                 numeric_features_std=None, eval=False):

        self.img_width = img_width
        self.eval = eval

        images = [torch.log(image + 1) for image in images]
        images = [image.transpose(0, 1) for image in images]

        # targets = torch.log(targets + 10e-6)  # List of targets
        # images list of tensors of shape (,)
        # targets (num_images, 1)
        # images_mean, images_std, targets_mean, targets_std: Tensors of shape (1,)

        # Check if scaling data was provided
        scaling_data = [images_mean, images_std,
                        targets_mean, targets_std,
                        numeric_features_mean, numeric_features_std]

        provided_data = [x is not None for x in scaling_data]

        if any([images_mean, images_std,
                targets_mean, targets_std]) and not all([images_mean, images_std,
                                                         targets_mean, targets_std]):
            raise ValueError("If any of the scaling data is provided, all of them should be provided.")

        if all(provided_data):
            # Scaling data provided
            (self.images_mean, self.images_std,
             self.targets_mean, self.targets_std,
             self.numeric_features_mean, self.numeric_features_std) = scaling_data


        else:
            # Scaling data not provided, calculate it
            flat_images = torch.cat([image.flatten() for image in images])
            self.images_mean = flat_images.mean()
            self.images_std = flat_images.std()
            self.targets_mean = targets.mean()
            self.targets_std = targets.std()
            if numeric_features is not None:
                self.numeric_features_mean = numeric_features.mean(dim=0)
                self.numeric_features_std = numeric_features.std(dim=0)
            else:
                self.numeric_features_mean = None
                self.numeric_features_std = None

        # Scale data
        self.images = [(image - self.images_mean) / self.images_std for image in images]
        self.targets = (targets - self.targets_mean) / self.targets_std
        if numeric_features is not None:
            self.numeric_features = (numeric_features - self.numeric_features_mean) / self.numeric_features_std
        else:
            self.numeric_features = None

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image = self.images[idx]
        image_width = image.shape[0]

        if self.img_width > 0:
            start = 0
            if not self.eval:
                # Cut image to width at a random place
                start = np.random.randint(0, image_width - self.img_width + 1)
            image = image[start:start + self.img_width]

        if self.numeric_features is not None:
            return image, self.targets[idx], self.numeric_features[idx]
        else:
            return image, self.targets[idx]


def collate_fn_w_mask(batch):
    sequences, targets, *numeric_features = zip(*batch)
    padded_sequences, attention_mask = pad_sequences(sequences)
    targets = torch.stack(targets)
    if numeric_features:
        numeric_features = torch.stack(numeric_features[0])
    else:
        numeric_features = None

    return padded_sequences, attention_mask, targets, numeric_features


def collate_fn_wo_mask(batch):
    sequences, targets, *numeric_features, = zip(*batch)
    targets = torch.stack(targets)
    if numeric_features:
        numeric_features = torch.stack(numeric_features[0])
    else:
        numeric_features = None
    sequences = torch.stack(sequences)

    return sequences, None, targets, numeric_features


def split_data_into_datasets(data_df, train=0.8, val=0.1, test=0.1, seed=1104, device='cpu', no_samples=-1,
                             min_width=-1, width_column='width', target_column='frequency', split_column=None,
                             image_file_column='image_file', numerical_columns=None, image_directory: str = None
                             ):
    # target_column = 'ang_vel[deg/s]'
    # numerical_columns = ['IRSKY_TEMP', 'TEMP', 'WINDSP', 'PRES', 'FWHM', 'RHUM', 'TAU0']

    # Filter data by width if min_width is provided
    if width_column and min_width > 0:
        data_df = data_df[data_df[width_column] >= min_width]

    if no_samples > 0:
        data_df = data_df.head(no_samples)
    if split_column:
        # Split data by split_column
        split_data = data_df[split_column].unique()
        split_data_temp, split_data_test = train_test_split(split_data, train_size=train + val, random_state=seed)
        split_data_train, split_data_val = train_test_split(split_data_temp, train_size=train / (train + val),
                                                            random_state=seed + 1)

        train_data = data_df[data_df[split_column].isin(split_data_train)]
        val_data = data_df[data_df[split_column].isin(split_data_val)]
        test_data = data_df[data_df[split_column].isin(split_data_test)]
    else:
        train_data, test_data = train_test_split(data_df, test_size=test, random_state=seed)
        train_data, val_data = train_test_split(train_data, test_size=val / (train + val), random_state=seed + 1)

    print(f"Train: {len(train_data)} samples, Val: {len(val_data)} samples, Test: {len(test_data)} samples")

    def get_data(data: pd.DataFrame):
        images = []
        targets = torch.tensor(data[target_column].to_numpy(), dtype=torch.float, device=device)
        numerical = None
        if numerical_columns:
            numerical = torch.tensor(data[numerical_columns].to_numpy(), dtype=torch.float, device=device)
        for _, row in data.iterrows():
            image = torch.tensor(np.load(utils.path_from_proot(os.path.join(image_directory, row[image_file_column]))),
                                 dtype=torch.float, device=device)
            images.append(image)
        return images, targets, numerical

    train_dataset = BaseDataset(*get_data(train_data), img_width=min_width)

    val_dataset = BaseDataset(*get_data(val_data),
                              images_mean=train_dataset.images_mean,
                              images_std=train_dataset.images_std,
                              numeric_features_mean=train_dataset.numeric_features_mean,
                              numeric_features_std=train_dataset.numeric_features_std,
                              targets_mean=train_dataset.targets_mean,
                              targets_std=train_dataset.targets_std,
                              img_width=min_width,
                              eval=True
                              )
    test_dataset = BaseDataset(*get_data(test_data),
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

    #
    # # Split the data into training + validation and testing
    # train_val_df, test_df = train_test_split(data_df, test_size=test, random_state=seed)
    #
    # # Further split training + validation into separate training and validation sets
    # train_df, val_df = train_test_split(train_val_df, test_size=val/(train + val), random_state=seed + 1)
    #
    # images_folder_path = data_path
    # def get_data(data):
    #     images = [
    #         torch.tensor(np.load(os.path.join(images_folder_path, row['filename'])), dtype=torch.float, device=device) for (_, row) in data.iterrows()
    #     ]
    #     targets = torch.tensor(data['frequency'].to_numpy(), dtype=torch.float, device=device)
    #     return images, targets
    #
    #
    # train_dataset = ArtificialStreaksDatasetWithFixedWidth(*get_data(train_df), img_width=min_width)
    #
    # val_dataset = ArtificialStreaksDatasetWithFixedWidth(*get_data(val_df),
    #                                                      images_mean=train_dataset.images_mean,
    #                                                      images_std=train_dataset.images_std,
    #                                                      targets_mean=train_dataset.targets_mean,
    #                                                      targets_std=train_dataset.targets_std,
    #                                                      img_width=min_width,
    #                                                      eval=True)
    # test_dataset = ArtificialStreaksDatasetWithFixedWidth(*get_data(test_df),
    #                                                       images_mean=train_dataset.images_mean,
    #                                                       images_std=train_dataset.images_std,
    #                                                       targets_mean=train_dataset.targets_mean,
    #                                                       targets_std=train_dataset.targets_std,
    #                                                       img_width=min_width,
    #                                                       eval=True)
    #
    # train_df = train_df.reset_index(drop=True)
    # val_df = val_df.reset_index(drop=True)
    # test_df = test_df.reset_index(drop=True)

    return {"train": train_dataset, "val": val_dataset, "test": test_dataset}, {"train": train_df, "val": val_df,
                                                                                "test": test_df}


def split_data_into_datasets_synthetic_32(train=0.8, val=0.1, test=0.1, seed=1104, device='cpu',
                                          image_directory='src/datasets/synthetic_32/', no_samples=-1, min_width=500):
    data_df = pd.read_csv(utils.path_from_proot("src/datasets/synthetic_32.csv"))
    if min_width > 1000:
        raise ValueError("Minimum width should be less than 1000 for synthetic data.")

    return split_data_into_datasets(data_df, train=train, val=val, test=test, seed=seed, device=device,
                                    no_samples=no_samples, min_width=min_width, width_column=None, target_column='D',
                                    image_file_column='npy_path', image_directory=image_directory)


def split_data_into_datasets_strips_171124_lc(train=0.8, val=0.1, test=0.1, seed=1104, device='cpu',
                                              image_directory='src/datasets/strips_171124_lc/strips_171124_lc', no_samples=-1,
                                              min_width=500,
                                              numerical_columns=(
                                              'IRSKY_TEMP', 'TEMP', 'WINDSP', 'PRES', 'FWHM', 'RHUM', 'TAU0'),
                                              split_on_files=True):
    split_on_column = None
    if split_on_files:
        split_on_column = 'file_name'
    if numerical_columns:
        numerical_columns = list(numerical_columns)
    data_df = pd.read_csv(utils.path_from_proot("src/datasets/combined_lc.csv"))
    return split_data_into_datasets(data_df, train=train, val=val, test=test, seed=seed, device=device,
                                    no_samples=no_samples, min_width=min_width, width_column='width',
                                    target_column='ang_vel[deg/s]',
                                    image_file_column='image_name', image_directory=image_directory,
                                    numerical_columns=numerical_columns,
                                    split_column=split_on_column)


def split_data_into_datasets_artificial_wx():
    raise NotImplementedError("This function is not implemented yet.")


def split_data_into_datasets_artificial_600px():
    raise NotImplementedError("This function is not implemented yet.")
