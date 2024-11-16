from src.models.dataset import StreaksDataset
from src.models.dataset import collate_fn
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
) -> Tuple[nn.Module, optim.Optimizer, List[float], List[float], List[float]]:
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

    Returns:
        Tuple[nn.Module, optim.Optimizer, List[float], List[float], List[float]]:
            Trained model, optimizer, training losses, validation losses, test losses per epoch.
    """
    print("Start of train()")

    # Initialize data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    print("Loaded datasets")

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

    # Lists to store losses
    train_losses = []
    val_losses = []
    test_losses = []


    print("Starting training")
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
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

            total_train_loss += loss.item()
            train_loader_tqdm.set_postfix(loss=loss.item())


        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")

        # Validation step
        model.eval()
        total_val_loss = 0
        total_val_err = 0
        val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", leave=False)

        with torch.no_grad():
            for padded_sequences, attention_mask, numeric_features, targets in val_loader_tqdm:
                padded_sequences = padded_sequences.to(device)
                attention_mask = attention_mask.to(device)
                numeric_features = numeric_features.to(device)
                targets = targets.to(device)

                outputs = model(padded_sequences, numeric_features, attention_mask).squeeze(-1)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()
                total_val_err = torch.mean(torch.abs(outputs - targets))
                val_loader_tqdm.set_postfix(loss=loss.item())

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_err = total_val_err / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Error: {avg_val_err:.4f}")

    # Test step
    model.eval()
    total_test_loss = 0
    test_loader_tqdm = tqdm(test_loader, desc="Testing", leave=False)

    with torch.no_grad():
        for padded_sequences, attention_mask, numeric_features, targets in test_loader_tqdm:
            padded_sequences = padded_sequences.to(device)
            attention_mask = attention_mask.to(device)
            numeric_features = numeric_features.to(device)
            targets = targets.to(device)

            outputs = model(padded_sequences, numeric_features, attention_mask).squeeze(-1)
            loss = criterion(outputs, targets)
            total_test_loss += loss.item()
            test_loader_tqdm.set_postfix(loss=loss.item())

    avg_test_loss = total_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)
    print(f"Test Loss: {avg_test_loss:.4f}")

    print("Training complete!")
    return model, optimizer, train_losses, val_losses, test_losses