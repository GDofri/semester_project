from src.models.dataset import StreaksDataset
from src.models.base_dataset import collate_fn_w_mask
from src.models.base_dataset import collate_fn_wo_mask
from src.models.base_model import BaseModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Tuple, List, Dict
import time


def train(
    num_epochs: int,
    model: BaseModel,
    train_dataset: StreaksDataset,
    val_dataset: StreaksDataset,
    test_dataset: Optional[StreaksDataset] = None,
    optimizer: Optional[optim.Optimizer] = None,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    patience: int = 5,
    verbose: bool = True
) -> Tuple[nn.Module, optim.Optimizer, List[float], List[float], float]:
    """
    Train the model.

    Args:
        num_epochs (int): Number of epochs to train.
        train_dataset (StreaksDataset): Training dataset.
        val_dataset (StreaksDataset): Validation dataset.
        test_dataset (StreaksDataset): Test dataset.
        model (Optional[nn.Module]): Model to train. If None, a new model is created.
        optimizer (Optional[optim.Optimizer]): Optimizer to use. If None, a new optimizer is created.
        batch_size (int): Batch size for data loaders.
        learning_rate (float): Learning rate for the optimizer.
        patience (int): Number of epochs with no improvement after which training will be stopped.

    Returns:
        Tuple[nn.Module, optim.Optimizer, List[float], List[float], List[float]]:
            Trained model, optimizer, training losses, validation losses, per epoch and test loss at the end.
    """

    print("Training model:", model.get_metadata())
    collate_fn = collate_fn_w_mask if model.input_requires_mask else collate_fn_wo_mask
    # Initialize data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    verbose_print("Data loaders initialized", verbose)
    sample_batch = next(iter(train_loader))
    expected_num_batch_items = 2 + int(model.input_requires_numerics) + int(model.input_requires_mask)
    if sum([1 for batch_item in sample_batch if batch_item is not None]) != expected_num_batch_items:
        print(f"WARNING: Expected {expected_num_batch_items} items in the batch, but got {len(sample_batch)}")


    # Initialize the model, optimizer, and loss function
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        print("Created new optimizer")
    else:
        print("Using existing optimizer")

    criterion = nn.MSELoss()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None

    train_losses = []
    val_losses   = []

    start = time.time()
    verbose_print("Starting training", verbose)
    for epoch in range(num_epochs):
        # Training
        avg_train_loss = run_epoch(train_loader, model, device, criterion, optimizer)
        train_losses.append(avg_train_loss)
        verbose_print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}", verbose)

        # Validation
        avg_val_loss = run_epoch(val_loader, model, device, criterion, optimizer=None)
        val_losses.append(avg_val_loss)
        verbose_print(f"Validation Loss: {avg_val_loss:.4f}", verbose)

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict()
            verbose_print(f"Validation loss improved to {best_val_loss:.4f}. Saving model.", verbose)
        else:
            epochs_without_improvement += 1
            verbose_print(f"No improvement in validation loss for {epochs_without_improvement} epoch(s).", verbose)
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # Load best validation model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model based on validation loss.")

    # Testing
    avg_test_loss = None
    if test_loader:
        avg_test_loss = run_epoch(test_loader, model, device, criterion, optimizer=None)
        print(f"Test Loss: {avg_test_loss:.4f}")

    print(f"Training complete in {time.time() - start:.0f} seconds")
    return model, optimizer, train_losses, val_losses, avg_test_loss


def run_epoch(
    loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    criterion: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
) -> float:
    """
    Runs a single epoch of training or evaluation on the given loader.
    If optimizer is provided, runs in training mode; otherwise in eval mode.
    Returns a per-sample average loss for the entire epoch.
    """
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        padded_sequences, attention_mask, targets, numeric_features = batch

        padded_sequences = padded_sequences.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if numeric_features is not None:
            numeric_features = numeric_features.to(device)
        targets = targets.to(device)

        if is_training:
            optimizer.zero_grad()

        # Forward pass
        if model.input_requires_mask and model.input_requires_numerics:
            outputs = model(padded_sequences, numeric_features, attention_mask).squeeze(-1)
        elif model.input_requires_mask:
            outputs = model(padded_sequences, attention_mask).squeeze(-1)
        elif model.input_requires_numerics:
            outputs = model(padded_sequences, numeric_features).squeeze(-1)
        else:
            outputs = model(padded_sequences).squeeze(-1)

        loss = criterion(outputs, targets)

        if is_training:
            loss.backward()
            optimizer.step()

        # Accumulate per-sample loss
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples

def test(
        test_dataset: StreaksDataset,
        model: BaseModel,
        batch_size: int = 8,
        criterion: Optional[nn.Module] = None
) -> float:
    """
    Evaluate the model on the test dataset.

    Args:
        test_dataset (StreaksDataset): Test dataset.
        model (BaseModel): Trained model.
        batch_size (int): Batch size for the data loader.
        criterion (Optional[nn.Module]): Loss function to use. If None, defaults to MSELoss.

    Returns:
        float: Average test loss.
    """
    print("Starting testing phase...")
    collate_fn = collate_fn_w_mask if model.input_requires_mask else collate_fn_wo_mask
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    if criterion is None:
        criterion = nn.MSELoss()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Run test epoch
    avg_test_loss = run_epoch(test_loader, model, device, criterion, optimizer=None)
    print(f"Test Loss: {avg_test_loss:.4f}")

    return avg_test_loss

def train_kfolds(
        num_epochs: int,
        model: BaseModel,
        folds: List[Dict[str, StreaksDataset]],
        test_dataset: Optional[StreaksDataset] = None,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        patience: int = 5,
        verbose: bool = True
) -> Dict[str, List[float]]:
    """
    Perform k-fold cross-validation training with model reinitialization using the input model's initial weights.

    Args:
        num_epochs (int): Number of epochs to train in each fold.
        model (BaseModel): The model instance to train.
        folds (List[Dict[str, StreaksDataset]]): List of dictionaries with 'train' and 'val' datasets for each fold.
        test_dataset (StreaksDataset): Test dataset for final evaluation.
        batch_size (int): Batch size for data loaders.
        learning_rate (float): Learning rate for the optimizer.
        patience (int): Number of epochs with no improvement after which training will stop.
        verbose: If true training prints train and val losses for each epoch.
    Returns:
        Dict[str, List[float]]: Dictionary with fold-wise training, validation, and test losses.
    """
    import numpy as np

    fold_train_losses = []
    fold_val_losses = []

    # Save initial model state for resetting
    initial_state = model.state_dict()

    for fold_idx, fold_data in enumerate(folds):
        print(f"\n=== Starting training for fold {fold_idx + 1}/{len(folds)} ===\n")

        # Reset model to its initial state
        model.load_state_dict(initial_state)

        # Create a new optimizer for this fold
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Get train and validation datasets for the current fold
        print(fold_data)
        train_dataset = fold_data["train"]
        val_dataset = fold_data["val"]

        # Train the model on the current fold
        model, _, train_losses, val_losses, _ = train(
            num_epochs=num_epochs,
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=None,  # Test separately after all folds
            optimizer=optimizer,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=patience,
            verbose=verbose
        )

        # Store the final training and validation loss for this fold
        fold_train_losses.append(train_losses[-1])  # Last epoch train loss
        fold_val_losses.append(val_losses[-1])  # Last epoch val loss

        print(f"Fold {fold_idx + 1} complete:")
        print(f"  Final Training Loss: {train_losses[-1]:.4f}")
        print(f"  Final Validation Loss: {val_losses[-1]:.4f}")

    # Final testing on the test dataset (if provided)
    test_loss = None
    if test_dataset:
        print("\nEvaluating on test dataset...")
        model.load_state_dict(initial_state)  # Reset model to initial state for testing
        test_loss = test(test_dataset, model, batch_size=batch_size)
        print(f"Final Test Loss: {test_loss:.4f}")

    # Print detailed summary
    print("\n=== K-Fold Cross-Validation Summary ===")
    for fold_idx, (train_loss, val_loss) in enumerate(zip(fold_train_losses, fold_val_losses)):
        print(f"Fold {fold_idx + 1}:")
        print(f"  Training Loss: {train_loss:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}")

    # Calculate and print mean and median
    train_mean = np.mean(fold_train_losses)
    train_median = np.median(fold_train_losses)
    val_mean = np.mean(fold_val_losses)
    val_median = np.median(fold_val_losses)

    print("\n=== Overall Performance ===")
    print(f"Training Losses: {fold_train_losses}")
    print(f"Validation Losses: {fold_val_losses}")
    print(f"  Mean Training Loss: {train_mean:.4f}")
    print(f"  Median Training Loss: {train_median:.4f}")
    print(f"  Mean Validation Loss: {val_mean:.4f}")
    print(f"  Median Validation Loss: {val_median:.4f}")
    print(f"  Mean Test Loss: {val_median:.4f}")


    # Return the losses for all folds
    return {
        "train_losses": fold_train_losses,
        "val_losses": fold_val_losses,
        "test_loss": test_loss,
        "train_mean": train_mean,
        "train_median": train_median,
        "val_mean": val_mean,
        "val_median": val_median,
    }

def verbose_print( string : str, verbose:bool):
    if verbose:
        print(string)