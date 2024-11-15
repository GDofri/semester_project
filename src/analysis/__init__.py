import torch
from torch.utils.data import DataLoader
import numpy as np

def evaluate_model(model, test_dataset, test_df, batch_size=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Prepare DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Ensures alignment with test_df
        collate_fn=collate_fn
    )

    # Retrieve scaling factors from test_dataset
    targets_mean = test_dataset.targets_mean.to(device)
    targets_std = test_dataset.targets_std.to(device)

    # Lists to store results
    predicted_frequencies = []
    true_frequencies = []
    widths = []
    sample_mses = []

    with torch.no_grad():
        for idx, (padded_sequences, attention_mask, targets) in enumerate(test_loader):
            padded_sequences = padded_sequences.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)

            # Model prediction
            outputs = model(padded_sequences, attention_mask).squeeze(-1)

            # Rescale outputs and targets
            outputs_rescaled = outputs * targets_std + targets_mean
            targets_rescaled = targets * targets_std + targets_mean

            # Undo the logarithm transformation
            outputs_original = torch.exp(outputs_rescaled) - 1e-6
            targets_original = torch.exp(targets_rescaled) - 1e-6

            # Convert to CPU and numpy for compatibility
            predicted_frequency = outputs_original.cpu().numpy()
            true_frequency = targets_original.cpu().numpy()

            # Compute MSE for each sample in the batch
            mse = (predicted_frequency - true_frequency) ** 2

            # Get corresponding DataFrame rows
            batch_size_actual = targets.size(0)
            df_rows = test_df.iloc[idx * batch_size: idx * batch_size + batch_size_actual]
            width = df_rows['width'].values

            # Store results
            predicted_frequencies.extend(predicted_frequency)
            true_frequencies.extend(true_frequency)
            widths.extend(width)
            sample_mses.extend(mse)

    # Convert lists to numpy arrays
    predicted_frequencies = np.array(predicted_frequencies)
    true_frequencies = np.array(true_frequencies)
    widths = np.array(widths)
    sample_mses = np.array(sample_mses)

    return predicted_frequencies, true_frequencies, widths, sample_mses