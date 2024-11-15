import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from src.models.transformer_artifical_data import artificial_transformer
from src.models.transformer_artifical_data import artificial_dataset
#
# def initialize_csv(file_path, description):
#     """
#     Creates a new CSV file with a free-text description at the top.
#
#     Args:
#         file_path (str): The path to the CSV file.
#         description (str): A description of the experiment.
#     """
#     with open(file_path, 'w') as f:
#         # Write the description as a comment at the top
#         f.write(f"# Experiment Description: {description}\n")
#         f.write("# Tracking losses and metrics for each epoch\n")
#         # Write the headers for the epoch data
#         f.write("epoch,train_loss,val_loss\n")

def evaluate_model(model, test_dataset, test_df, batch_size=1, save=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    test_df_cp = test_df.copy()

    # Prepare DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Ensures alignment with test_df
        collate_fn=artificial_dataset.collate_fn
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

def analyze_results(predicted_frequencies, true_frequencies, widths):

    # Compute MSE for each sample in the batch
    sample_mses = (predicted_frequencies - true_frequencies) ** 2
    # Compute Pearson correlation between frequency and MSE
    freq_mse_corr, freq_mse_pval = pearsonr(true_frequencies, sample_mses)
    print(f"Correlation between True Frequency and MSE: {freq_mse_corr:.4f}, p-value: {freq_mse_pval:.4e}")

    # Compute Pearson correlation betwee<n width and MSE
    width_mse_corr, width_mse_pval = pearsonr(widths, sample_mses)
    print(f"Correlation between Width and MSE: {width_mse_corr:.4f}, p-value: {width_mse_pval:.4e}")

    def mean_error(predicted, true):
        return (predicted - true).mean()
    print(f"Mean error: {mean_error(predicted_frequencies, true_frequencies)}")

    # Plot True Frequency vs. MSE
    plt.figure(figsize=(8, 6))
    plt.scatter(true_frequencies, sample_mses)
    plt.xlabel('True Frequency')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('True Frequency vs. MSE')
    plt.show()

    # Plot Width vs. MSE
    plt.figure(figsize=(8, 6))
    plt.scatter(widths, sample_mses)
    plt.xlabel('Width')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Width vs. MSE')
    plt.show()

    # Additional Plot: True vs. Predicted Frequencies
    plt.figure(figsize=(8, 6))
    plt.scatter(true_frequencies, predicted_frequencies)
    plt.plot([true_frequencies.min(), true_frequencies.max()],
             [true_frequencies.min(), true_frequencies.max()], 'k--', lw=2)
    plt.xlabel('True Frequency')
    plt.ylabel('Predicted Frequency')
    plt.title('Predicted vs. True Frequency')
    plt.show()

if __name__ == '__main__':
    pass
    # # Assuming you have your trained model and test_dataset, test_df
    # # Load datasets and dataframes
    # datasets, dataframes = artificial_dataset.split_data_into_datasets()
    # test_dataset = datasets['test']
    # test_df = dataframes['test']
    #
    # # Load your trained model
    # # If you haven't trained the model yet, make sure to do so before this step
    # # model = ...  # Load or define your trained model
    #
    # # Evaluate the model on the test dataset
    # predicted_freqs, true_freqs, widths, mses = evaluate_model(model, test_dataset, test_df)
    #
    # # Analyze results
    # analyze_results(predicted_freqs, true_freqs, widths, mses)