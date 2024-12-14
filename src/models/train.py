from src.models.dataset import StreaksDataset
from src.models.base_dataset import collate_fn_w_mask
from src.models.base_dataset import collate_fn_wo_mask
from src.models.base_model import BaseModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Tuple, List


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

def train(
    num_epochs: int,
    train_dataset: StreaksDataset,
    val_dataset: StreaksDataset,
    test_dataset: StreaksDataset,
    model: BaseModel,
    optimizer: Optional[optim.Optimizer] = None,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    patience: int = 5,
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
    print("Start of train()")
    print("Training model:", model.get_metadata())
    collate_fn = collate_fn_w_mask if model.input_requires_mask else collate_fn_wo_mask
    # Initialize data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    print("Data loaders initialized")
    sample_batch = next(iter(train_loader))
    expected_num_batch_items = 2 + int(model.input_requires_numerics) + int(model.input_requires_mask)
    if sum([1 for batch_item in sample_batch if batch_item is not None]) != expected_num_batch_items:
        raise ValueError(f"Expected {expected_num_batch_items} items in the batch, but got {len(sample_batch)}")


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


    print("Starting training")
    for epoch in range(num_epochs):
        # Training
        avg_train_loss = run_epoch(train_loader, model, device, criterion, optimizer)
        train_losses.append(avg_train_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")

        # Validation
        avg_val_loss = run_epoch(val_loader, model, device, criterion, optimizer=None)
        val_losses.append(avg_val_loss)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict()
            print(f"Validation loss improved to {best_val_loss:.4f}. Saving model.")
        else:
            epochs_without_improvement += 1
            print(f"No improvement in validation loss for {epochs_without_improvement} epoch(s).")
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # Load best validation model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model based on validation loss.")

    # Testing
    avg_test_loss = run_epoch(test_loader, model, device, criterion, optimizer=None)
    print(f"Test Loss: {avg_test_loss:.4f}")

    print("Training complete!")
    return model, optimizer, train_losses, val_losses, avg_test_loss
