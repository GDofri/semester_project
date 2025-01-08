import os

import pandas as pd
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from src import utils

from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split, KFold
from typing import Optional, Tuple, List, Dict, Union


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
                 numeric_features_mean=None, numeric_features_std=None,
                 eval=False,
                 augmentation_opts=None):
        """
        augmentation_opts: dict with optional augmentation parameters, e.g.
            {
                'horizontal_flip': True,
                'vertical_flip': True,
                'log_scale': True,
                'normalize_images': True
                'max_rotation': 10.0,       # degrees
                'max_translation': 5        # pixels
            }
        """
        self.img_width = img_width
        self.eval = eval
        self.augmentation_opts = augmentation_opts or {}

        if self.augmentation_opts.get('log_scale', True):
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
        if self.augmentation_opts.get('normalize_images', True):
            self.images = [(image - self.images_mean) / self.images_std for image in images]
        else:
            self.images = images

        self.targets = (targets - self.targets_mean) / self.targets_std
        if numeric_features is not None:
            self.numeric_features = (numeric_features - self.numeric_features_mean) / self.numeric_features_std
        else:
            self.numeric_features = None

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]
        numeric = self.numeric_features[idx] if self.numeric_features is not None else None

        image_width = image.shape[0]
        if self.img_width > 0:
            start = 0
            if not self.eval:
                # Cut image to width at a random place
                start = np.random.randint(0, image_width - self.img_width + 1)
            image = image[start:start + self.img_width]

        # Data Augmentations
        if not self.eval and self.augmentation_opts:
            image = self.apply_augmentations(image)

        # If image is not the correct height, cut it
        if image.shape[1] > 32:
            trim_size_top = (image.shape[1] - 32) // 2
            trim_size_bottom = image.shape[1] - 32 - trim_size_top
            image = image[:, trim_size_top:-trim_size_bottom]

        if numeric is not None:
            return image, target, numeric
        else:
            return image, target

    def apply_augmentations(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply random flips, rotation, and translation based on augmentation_opts.
        We assume `image` is a 2D tensor (H, W). Adjust dims for channels if needed.
        """
        opts = self.augmentation_opts

        # Random horizontal flip
        if opts.get('horizontal_flip', False) and np.random.rand() < 0.5:
            # Flip along width dimension
            image = torch.flip(image, dims=[-1])

        # Random vertical flip
        if opts.get('vertical_flip', False) and np.random.rand() < 0.5:
            # Flip along height dimension
            image = torch.flip(image, dims=[-2])

        # Random rotation
        max_rotation = opts.get('max_rotation', 0)
        if max_rotation > 0:
            image = self.rotate_tensor(image, max_rotation)

        # Random translation
        max_translation = opts.get('max_translation', 0)
        if max_translation > 0:
            ty = np.random.randint(-max_translation, max_translation + 1)
            image = self.translate_tensor(image, ty)

        return image

    def rotate_tensor(self, image: torch.Tensor, max_rotation: float) -> torch.Tensor:
        """
        Rotate the 2D `image` by `max_angle`.
        This is a minimal example using affine_grid + grid_sample or manual rotation.
        """
        rotation = transforms.RandomRotation(max_rotation, interpolation=transforms.InterpolationMode.BILINEAR)
        return rotation(image.unsqueeze(0)).squeeze(0)

    def translate_tensor(self, image: torch.Tensor, ty: int) -> torch.Tensor:
        """
        Translate the 2D `image` by (tx, ty). Minimal manual approach:
        """
        # We'll fill out-of-bounds with zeros
        w, h = image.shape
        # Create new empty
        translated = torch.zeros_like(image)
        # Compute valid bounding region
        y1_src, y1_dst = max(0, -ty), max(0, ty)
        y2_src, y2_dst = h - abs(ty), h - abs(ty)

        # Copy overlapping region
        translated[:, y1_dst:y1_dst + (y2_src - y1_src)] = image[:, y1_src:y2_src]
        return translated


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


def get_scaling_and_fixed_params(train_dataset: BaseDataset, img_width: int, eval_mode: bool = True) -> Dict[
    str, Union[torch.Tensor, int, bool]]:
    """
    Collects scaling parameters from the training dataset and adds fixed parameters.

    Args:
        train_dataset (BaseDataset): The training dataset containing scaling parameters.
        img_width (int): The image width to be used.
        eval_mode (bool): Whether the dataset is in evaluation mode.

    Returns:
        Dict[str, Union[torch.Tensor, int, bool]]: A dictionary of parameters.
    """
    params = {
        'images_mean': train_dataset.images_mean,
        'images_std': train_dataset.images_std,
        'numeric_features_mean': train_dataset.numeric_features_mean,
        'numeric_features_std': train_dataset.numeric_features_std,
        'targets_mean': train_dataset.targets_mean,
        'targets_std': train_dataset.targets_std,
        'img_width': img_width,
        'eval': eval_mode
    }
    return params


def get_data(
        data: pd.DataFrame,
        target_column: str,
        image_file_column: str,
        numerical_columns: Optional[List[str]],
        image_directory: str,
        device: str
) -> Tuple[List[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
    """
    Loads images, targets, and numerical features from the DataFrame.

    Args:
        data (pd.DataFrame): The data subset.
        target_column (str): The target column name.
        image_file_column (str): The image file column name.
        numerical_columns (Optional[List[str]]): List of numerical feature columns.
        image_directory (str): Directory where images are stored.
        device (str): Device to load tensors onto.

    Returns:
        Tuple[List[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]: Images, targets, numerical features.
    """
    images = []
    targets = torch.tensor(data[target_column].to_numpy(), dtype=torch.float, device=device)
    numerical = None
    if numerical_columns:
        numerical = torch.tensor(data[numerical_columns].to_numpy(), dtype=torch.float, device=device)
    for _, row in data.iterrows():
        image_path = utils.path_from_proot(os.path.join(image_directory, row[image_file_column]))
        image = torch.tensor(np.load(image_path), dtype=torch.float, device=device)
        images.append(image)
    return images, targets, numerical


def prepare_datasets(
        data_df: pd.DataFrame,
        train: float = 0.8,
        val: float = 0.1,
        test: float = 0.1,
        seed: int = 1104,
        device: str = 'cpu',
        no_samples: int = -1,
        min_width: int = -1,
        width_column: Optional[str] = 'width',
        target_column: str = 'frequency',
        split_column: Optional[str] = None,
        image_file_column: str = 'image_file',
        numerical_columns: Optional[List[str]] = None,
        image_directory: Optional[str] = None,
        augmentation_opts: Optional[Dict] = None,
        k_fold: int = 1  # Set to >1 for k-fold cross-validation
) -> Union[
    Tuple[Dict[str, BaseDataset], Dict[str, pd.DataFrame]],
    Tuple[Dict[str, List[Dict[str, BaseDataset]]], Dict[str, BaseDataset], Dict[str, BaseDataset], Dict[
        str, pd.DataFrame]]
]:
    """
    Splits the data into train, validation, and test sets, or into k folds for cross-validation.

    Args:
        data_df (pd.DataFrame): The complete dataset.
        train (float): Proportion of data for training.
        val (float): Proportion of data for validation.
        test (float): Proportion of data for testing.
        seed (int): Random seed for reproducibility.
        device (str): Device to load tensors onto.
        no_samples (int): Number of samples to use. -1 means all.
        min_width (int): Minimum width filter for data.
        width_column (Optional[str]): Column name for width filtering.
        target_column (str): Target variable column name.
        split_column (Optional[str]): Column name to split data uniquely (e.g., filenames).
        image_file_column (str): Column name for image file paths.
        numerical_columns (Optional[List[str]]): List of numerical feature columns.
        image_directory (Optional[str]): Directory where images are stored.
        augmentation_opts (Optional[Dict]): Augmentation options for training data.
        k_fold (int): Number of folds for cross-validation. Set to 1 for standard splitting.

    Returns:
        If k_fold == 1:
            Tuple containing:
                - A dictionary with 'train', 'val', and 'test' BaseDataset instances.
                - A dictionary with corresponding pandas DataFrames.
        If k_fold > 1:
            Tuple containing:
                - A list of dictionaries for each fold's 'train' and 'val' BaseDataset instances.
                - A separate 'test' BaseDataset instance.
                - A dictionary with the 'test' pandas DataFrame.
    """
    np.random.seed(seed)

    # Filter data by width if min_width is provided
    if width_column and min_width > 0:
        data_df = data_df[data_df[width_column] >= min_width]

    if no_samples > 0:
        data_df = data_df.head(no_samples)

    if k_fold < 1:
        raise ValueError("k_fold must be at least 1.")

    if k_fold == 1:
        # Standard train/val/test split

        split_data = data_df[split_column].unique() if split_column else data_df
        train_val_size = train + val
        train_val_split, test_split = train_test_split(
            split_data,
            train_size=train_val_size,
            random_state=seed,
            shuffle=True
        )
        # Further split train_val_data into train and val
        train_size_adjusted = train / (train + val)
        train_split, val_split = train_test_split(
            train_val_split,
            train_size=train_size_adjusted,
            random_state=seed + 1,
            shuffle=True
        )
        if split_column:
            test_data = data_df[data_df[split_column].isin(test_split)]
            train_data = data_df[data_df[split_column].isin(train_split)]
            val_data = data_df[data_df[split_column].isin(val_split)]
        else:
            test_data = test_split
            train_data = train_split
            val_data = val_split

        print(f"Train: {len(train_data)} samples, Val: {len(val_data)} samples, Test: {len(test_data)} samples")

        # Create datasets
        train_images, train_targets, train_numerical = get_data(
            train_data, target_column, image_file_column, numerical_columns, image_directory, device
        )
        val_images, val_targets, val_numerical = get_data(
            val_data, target_column, image_file_column, numerical_columns, image_directory, device
        )
        test_images, test_targets, test_numerical = get_data(
            test_data, target_column, image_file_column, numerical_columns, image_directory, device
        )

        train_dataset = BaseDataset(
            train_images,
            train_targets,
            numeric_features=train_numerical,
            img_width=min_width,
            augmentation_opts=augmentation_opts
        )

        scaling_and_fixed_params = get_scaling_and_fixed_params(train_dataset, img_width=min_width, eval_mode=True)

        val_dataset = BaseDataset(
            val_images,
            val_targets,
            numeric_features=val_numerical,
            **scaling_and_fixed_params  # Unpack scaling and fixed parameters
        )

        test_dataset = BaseDataset(
            test_images,
            test_targets,
            numeric_features=test_numerical,
            **scaling_and_fixed_params,  # Unpack scaling and fixed parameters
        )

        train_df = train_data.reset_index(drop=True)
        val_df = val_data.reset_index(drop=True)
        test_df = test_data.reset_index(drop=True)

        return (
            {"train": train_dataset, "val": val_dataset, "test": test_dataset},
            {"train": train_df, "val": val_df, "test": test_df}
        )

    else:
        # Split into test and train+val first
        split_data = data_df[split_column].unique() if split_column else data_df
        train_val_data, test_data = train_test_split(
            split_data,
            train_size=1-test,
            random_state=seed,
            shuffle=True
        )

        if split_column:
            train_val_df = data_df[data_df[split_column].isin(train_val_data)]
            test_df = data_df[data_df[split_column].isin(test_data)]
        else:
            train_val_df = train_val_data
            test_df = test_data

        # K-Fold Cross-Validation on Train+Validation data
        kf = KFold(n_splits=k_fold, shuffle=True, random_state=seed)
        folds_datasets = []

        if split_column:
            train_val_for_ksplit = train_val_df[split_column].unique()
        else:
            train_val_for_ksplit = train_val_df


        for fold, (train_indices, val_indices) in enumerate(kf.split(train_val_for_ksplit)):
            print(f"Fold {fold + 1}/{k_fold}")
            if split_column:
                train_file_names = train_val_for_ksplit[train_indices]
                val_file_names = train_val_for_ksplit[val_indices]
                train_subset = train_val_df[train_val_df[split_column].isin(train_file_names)]
                val_subset = train_val_df[train_val_df[split_column].isin(val_file_names)]
            else:
                train_subset = train_val_df.iloc[train_indices]
                val_subset = train_val_df.iloc[val_indices]

            # Create datasets
            train_images, train_targets, train_numerical = get_data(
                train_subset, target_column, image_file_column, numerical_columns, image_directory, device
            )
            val_images, val_targets, val_numerical = get_data(
                val_subset, target_column, image_file_column, numerical_columns, image_directory, device
            )

            train_dataset = BaseDataset(
                train_images,
                train_targets,
                numeric_features=train_numerical,
                img_width=min_width,
                augmentation_opts=augmentation_opts
            )

            scaling_and_fixed_params = get_scaling_and_fixed_params(train_dataset, img_width=min_width, eval_mode=True)

            val_dataset = BaseDataset(
                val_images,
                val_targets,
                numeric_features=val_numerical,
                **scaling_and_fixed_params  # Unpack scaling and fixed parameters
            )

            folds_datasets.append({"train": train_dataset, "val": val_dataset})
            print(f"Fold {fold + 1}: Train {len(train_subset)} samples, Val {len(val_subset)} samples")

        # Create a final train set:
        train_images, train_targets, train_numerical = get_data(
            train_val_df, target_column, image_file_column, numerical_columns, image_directory, device
        )
        train_dataset = BaseDataset(
            train_images,
            train_targets,
            numeric_features=train_numerical,
            img_width=min_width,
            augmentation_opts=augmentation_opts
        )
        scaling_and_fixed_params = get_scaling_and_fixed_params(train_dataset, img_width=min_width, eval_mode=True)

        # Create test dataset from test_df
        test_images, test_targets, test_numerical = get_data(
            test_df, target_column, image_file_column, numerical_columns, image_directory, device
        )

        test_dataset = BaseDataset(
            test_images,
            test_targets,
            numeric_features=test_numerical,
            **scaling_and_fixed_params,  # Unpack scaling and fixed parameters from the last fold's train_dataset
        )

        test_df = test_df.reset_index(drop=True)
        print(f"Test {len(test_df)} samples")
        return (
            folds_datasets,  # List of dicts for each fold's train and val datasets
            train_dataset,
            test_dataset,  # Separate test set
            test_df  # Separate test dataframe
        )


def split_data_into_datasets(data_df, train=0.8, val=0.1, test=0.1, seed=1104, device='cpu', no_samples=-1,
                             min_width=-1, width_column='width', target_column='frequency', split_column=None,
                             image_file_column='image_file', numerical_columns=None, image_directory: str = None,
                             augmentation_opts=None
                             ):
    # target_column = 'ang_vel[deg/s]'
    # numerical_columns = ['IRSKY_TEMP', 'TEMP', 'WINDSP', 'PRES', 'FWHM', 'RHUM', 'TAU0']
    np.random.seed(seed)
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

    train_dataset = BaseDataset(*get_data(train_data), img_width=min_width, augmentation_opts=augmentation_opts)

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

    return {"train": train_dataset, "val": val_dataset, "test": test_dataset}, {"train": train_df, "val": val_df,
                                                                                "test": test_df}


def split_data_into_datasets_synthetic_32(train=0.8, val=0.1, test=0.1, seed=1104, device='cpu',
                                          image_directory='src/datasets/synthetic_32/', no_samples=-1, min_width=500,
                                          augmentation_opts=None, k_fold=1):
    data_df = pd.read_csv(utils.path_from_proot("src/datasets/synthetic_32.csv"))
    if min_width > 1000:
        raise ValueError("Minimum width should be less than 1000 for synthetic data.")

    return prepare_datasets(data_df, train=train, val=val, test=test, seed=seed, device=device,
                                    no_samples=no_samples, min_width=min_width, width_column=None, target_column='D',
                                    image_file_column='npy_path', image_directory=image_directory,
                                    augmentation_opts=augmentation_opts, k_fold=k_fold)


def split_data_into_datasets_synthetic_50(train=0.8, val=0.1, test=0.1, seed=1104, device='cpu',
                                          image_directory='src/datasets/synthetic_50/', no_samples=-1, min_width=500,
                                          augmentation_opts=None, k_fold=1):
    # For now the synthetic_50.csv would be the same as synthetic_32.csv so we just use that here.
    data_df = pd.read_csv(utils.path_from_proot("src/datasets/synthetic_32.csv"))
    if min_width > 1000:
        raise ValueError("Minimum width should be less than 1000 for synthetic data.")

    return prepare_datasets(data_df, train=train, val=val, test=test, seed=seed, device=device,
                                    no_samples=no_samples, min_width=min_width, width_column=None, target_column='D',
                                    image_file_column='npy_path', image_directory=image_directory,
                                    augmentation_opts=augmentation_opts, k_fold=k_fold)


def split_data_into_datasets_strips_171124_lc(train=0.8, val=0.1, test=0.1, seed=1104, device='cpu',
                                              image_directory='src/datasets/strips_171124_lc/strips_171124_lc',
                                              no_samples=-1,
                                              min_width=500,
                                              numerical_columns=(
                                                      'IRSKY_TEMP', 'TEMP', 'WINDSP', 'PRES', 'FWHM', 'RHUM', 'TAU0'),
                                              split_on_files=True, augmentation_opts=None, k_fold=1):
    split_on_column = None
    if split_on_files:
        split_on_column = 'file_name'
    if numerical_columns:
        numerical_columns = list(numerical_columns)
    data_df = pd.read_csv(utils.path_from_proot("src/datasets/combined_lc.csv"))
    return prepare_datasets(data_df, train=train, val=val, test=test, seed=seed, device=device,
                                    no_samples=no_samples, min_width=min_width, width_column='width',
                                    target_column='ang_vel[deg/s]',
                                    image_file_column='image_name', image_directory=image_directory,
                                    numerical_columns=numerical_columns,
                                    split_column=split_on_column, augmentation_opts=augmentation_opts, k_fold=k_fold)


def split_data_into_datasets_strips_141224_lc_50(train=0.8, val=0.1, test=0.1, seed=1104, device='cpu',
                                                 image_directory='src/datasets/strips_141224_lc_50', no_samples=-1,
                                                 min_width=500,
                                                 numerical_columns=(
                                                         'IRSKY_TEMP', 'TEMP', 'WINDSP', 'PRES', 'FWHM', 'RHUM',
                                                         'TAU0'),
                                                 split_on_files=True, augmentation_opts=None, k_fold=1):
    split_on_column = None
    if split_on_files:
        split_on_column = 'file_name'
    if numerical_columns:
        numerical_columns = list(numerical_columns)
    data_df = pd.read_csv(utils.path_from_proot("src/datasets/combined_lc.csv"))
    return prepare_datasets(data_df, train=train, val=val, test=test, seed=seed, device=device,
                            no_samples=no_samples, min_width=min_width, width_column='width',
                            target_column='ang_vel[deg/s]',
                            image_file_column='image_name', image_directory=image_directory,
                            numerical_columns=numerical_columns,
                            split_column=split_on_column, augmentation_opts=augmentation_opts,
                            k_fold=k_fold)


def split_data_into_datasets_artificial_600px(train=0.8, val=0.1, test=0.1, seed=1104, device='cpu',
                                              image_directory='src/datasets/artificial_strips/w600px/', no_samples=-1,
                                              min_width=600, augmentation_opts=None, k_fold=1):

    data_df = pd.read_csv(utils.path_from_proot("src/datasets/artificial_strips/w600px/image_parameters.csv"))
    return prepare_datasets(data_df, train=train, val=val, test=test, seed=seed, device=device,
                            no_samples=no_samples, min_width=min_width, width_column='width',
                            target_column='frequency',
                            image_file_column='filename', image_directory=image_directory,
                            augmentation_opts=augmentation_opts,
                            k_fold=k_fold)

def split_data_into_datasets_artificial_wx():
    raise NotImplementedError("This function is not implemented yet.")


