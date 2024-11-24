from src.models.dataset import StreaksDataset
from src.models.dataset import collate_fn_w_mask
from src.models.simple_transformer import Transformer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Tuple, List

def train(
        num_epochs: int,
        train_dataset: StreaksDataset,
        val_dataset: StreaksDataset,
        test_dataset: StreaksDataset,
        model: Optional[nn.Module] = None,
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

    # Initialize data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_w_mask)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_w_mask)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_w_mask)
    print("Data loaders initialized")

    # Initialize the model, optimizer, and loss function
    if model is None:
        model = Transformer()
        print("Created new model")
    else:
        print("Using existing model")

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
    val_losses = []
    test_losses = []


    print("Starting training")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False)

        for padded_sequences, attention_mask, numeric_features, targets in train_loader_tqdm:
            padded_sequences = padded_sequences.to(device)
            attention_mask = attention_mask.to(device)
            numeric_features = numeric_features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(padded_sequences, numeric_features, attention_mask).squeeze(-1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_loader_tqdm.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.4f}")

        # Validation step
        model.eval()
        val_loss = 0
        val_error = 0

        val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", leave=False)
        total_elements = len(val_loader.dataset)
        with torch.no_grad():
            for padded_sequences, attention_mask, numeric_features, targets in val_loader_tqdm:
                padded_sequences = padded_sequences.to(device)
                attention_mask = attention_mask.to(device)
                numeric_features = numeric_features.to(device)
                targets = targets.to(device)

                outputs = model(padded_sequences, numeric_features, attention_mask).squeeze(-1)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * targets.size(0)
                val_error += (outputs - targets).sum().item()
                val_loader_tqdm.set_postfix(loss=loss.item())  # Update tqdm with the current loss

        avg_val_loss = val_loss / total_elements
        avg_val_error = val_error / total_elements
        val_losses.append(avg_val_loss)
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Mean Error: {avg_val_error:.4} ")

        # Early Stopping Check
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

    # Load the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model based on validation loss.")

    # Test step
    model.eval()
    test_loss = 0
    test_loader_tqdm = tqdm(test_loader, desc="Testing", leave=False)

    with torch.no_grad():
        for padded_sequences, attention_mask, numeric_features, targets in test_loader_tqdm:
            padded_sequences = padded_sequences.to(device)
            attention_mask = attention_mask.to(device)
            numeric_features = numeric_features.to(device)
            targets = targets.to(device)

            outputs = model(padded_sequences, numeric_features, attention_mask).squeeze(-1)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            test_loader_tqdm.set_postfix(loss=loss.item())

    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")

    print("Training complete!")
    return model, optimizer, train_losses, val_losses, avg_test_loss
